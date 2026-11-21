"""Capture-once / replay helper that degrades to a plain call on CPU."""

import warp as wp


def is_cuda(device) -> bool:
    return wp.get_device(str(device)).is_cuda


class LaunchGraph:
    """Wrap a launch sequence: on CUDA it is captured into a graph the first
    time and replayed afterwards; on CPU (no graph support) it is simply called."""

    def __init__(self, device, launch):
        self.device = str(device)
        self.launch = launch
        self.graph = None
        self.cuda = is_cuda(device)

    def __call__(self):
        if not self.cuda:
            self.launch()
            return
        if self.graph is None:
            wp.capture_begin(device=self.device)
            self.launch()
            self.graph = wp.capture_end(device=self.device)
        wp.capture_launch(self.graph)

    def reset(self):
        self.graph = None
