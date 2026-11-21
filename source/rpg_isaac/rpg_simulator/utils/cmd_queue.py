import torch
import warp as wp

from rpg_simulator.kernels import cmd_queue as K
from rpg_simulator.kernels.registry import ensure_loaded
from rpg_simulator.utils.graph import LaunchGraph


class CmdQueue:
    """
    A high-performance, Warp-based circular buffer for RL commands.

    Optimizations:
    1. Fused 'Peek & Pop': Checks time thresholds and updates buffers in one kernel.
    2. Zero-Copy: Maintains internal state on GPU without Python overhead.
    3. Graph Capturable: The 'step' operation is fully graph-compatible using handle-based updates.
    """

    def __init__(self, num_envs, capacity, act_dim, device="cuda", dtype=torch.float32):
        self.num_envs = num_envs
        self.capacity = capacity
        self.act_dim = act_dim
        self.device = device

        # Initialize Warp
        wp.init()
        self.wp_dtype = wp.float32 if dtype == torch.float32 else wp.float64
        self.wp_int = wp.int32

        # -----------------------------------------------------------
        # 1. Internal Buffers (PyTorch allocation)
        # -----------------------------------------------------------
        # Buffer shape: (N, Capacity, Act_Dim + 1) -> Last index is timestamp
        self.buffer = torch.zeros(
            (num_envs, capacity, act_dim + 1), device=device, dtype=dtype
        )
        self.buffer[..., -1] = float("inf")  # Init times to infinity

        # Indices: Head, Tail, Size
        self.head = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.tail = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.size = torch.zeros(num_envs, device=device, dtype=torch.int32)

        # The "Current Command" Tensor (Persistent Zero-Order Hold)
        self.current_cmd = torch.zeros((num_envs, act_dim), device=device, dtype=dtype)

        # -----------------------------------------------------------
        # 2. Create Zero-Copy Warp Handles
        # -----------------------------------------------------------
        self.wp_buffer = wp.from_torch(self.buffer, dtype=self.wp_dtype)
        self.wp_head = wp.from_torch(self.head, dtype=self.wp_int)
        self.wp_tail = wp.from_torch(self.tail, dtype=self.wp_int)
        self.wp_size = wp.from_torch(self.size, dtype=self.wp_int)
        self.wp_current_cmd = wp.from_torch(self.current_cmd, dtype=self.wp_dtype)

        # Special Handle: A 1-element array to hold the changing time threshold
        # This allows us to update the value inside the graph without recompiling or updating args
        self.time_threshold_handle = wp.zeros(
            1, dtype=self.wp_dtype, device=self.device
        )

        # -----------------------------------------------------------
        # 3. Kernels
        # -----------------------------------------------------------
        kernels = K.KERNELS[dtype]
        self.push_kernel = kernels["push"]
        self.process_kernel = kernels["process"]
        ensure_loaded(self.device, dtype, self.push_kernel.module, self.process_kernel.module)

        # 4. Capture Graph Immediately
        self._graph = None
        self._capture_process_graph()

    def _capture_process_graph(self):
        """Capture the step graph once using internal persistent pointers."""
        self._graph = LaunchGraph(self.device, self._launch_process)

    def _launch_process(self):
        wp.launch(
            kernel=self.process_kernel,
            dim=self.num_envs,
            inputs=[
                self.wp_buffer,
                self.wp_head,
                self.wp_size,
                self.wp_current_cmd,
                self.time_threshold_handle,  # Pass the handle, not the value!
                self.capacity,
                self.act_dim,
            ],
            device=self.device,
        )

    def push(
        self,
        actions: torch.Tensor,
        timestamps: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        """
        Push new commands.
        Note: Usually not graph-captured as inference timing varies.
        """
        if env_ids is not None:
            # Subset push path (used by partial resets). Keep it tensor-native so
            # interface matches non-warp CmdQueue without requiring a second kernel.
            env_ids = env_ids.to(device=self.head.device, dtype=torch.long)
            tail = self.tail[env_ids]

            self.buffer[env_ids, tail, :-1] = actions.to(dtype=self.buffer.dtype)
            self.buffer[env_ids, tail, -1] = timestamps.to(dtype=self.buffer.dtype)

            new_tail = (tail + 1) % self.capacity
            self.tail[env_ids] = new_tail

            size = self.size[env_ids]
            is_full = size >= self.capacity
            self.size[env_ids] = torch.where(is_full, size, size + 1)

            head = self.head[env_ids]
            self.head[env_ids] = torch.where(is_full, (head + 1) % self.capacity, head)
            return

        wp.launch(
            kernel=self.push_kernel,
            dim=self.num_envs,
            inputs=[
                self.wp_buffer,
                self.wp_head,
                self.wp_tail,
                self.wp_size,
                wp.from_torch(actions, dtype=self.wp_dtype),
                wp.from_torch(timestamps, dtype=self.wp_dtype),
                self.capacity,
                self.act_dim,
            ],
            device=self.device,
        )

    def step(self, current_time: float, delay: float):
        """
        Fused check-and-pop step via CUDA Graph Replay.
        Updates self.current_cmd if new commands are ready.
        """
        # 1. Update the value inside the GPU memory handle
        # This is a fast async copy that prepares the data for the graph
        new_thresh = current_time - delay
        self.time_threshold_handle.fill_(new_thresh)

        # 2. Replay the Graph
        # The graph executes the kernel, which reads the new value from the handle
        self._graph()

        return self.current_cmd

    def reset(self, env_ids=None):
        """
        Clear queues for specific environments.
        """
        if env_ids is None:
            self.head.zero_()
            self.tail.zero_()
            self.size.zero_()
            self.buffer[..., -1] = float("inf")
            self.current_cmd.zero_()
        else:
            self.head[env_ids] = 0
            self.tail[env_ids] = 0
            self.size[env_ids] = 0
            self.buffer[env_ids, ..., -1] = float("inf")
            self.current_cmd[env_ids] = 0.0
