import torch


class LowPassFilter:
    def __init__(
        self,
        dim: int,
        cutoff_freq: torch.Tensor,
        sampling_freq: torch.Tensor,
        initial_value: torch.Tensor,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize a low-pass filter.

        Args:
            dim (int): Number of rows (dimensions) for the filter.
            cutoff_freq: Cutoff frequency of the filter in Hz.  (dim,)
            sampling_freq: Sampling frequency in Hz. (dim,)
            initial_value: Initial value for the filter. (num_env, dim)
        """
        self.dim = dim
        self.dtype = cutoff_freq.dtype  # Infer dtype from cutoff_freq tensor
        self.cutoff_freq = (
            cutoff_freq.repeat(self.dim) if cutoff_freq.numel() == 1 else cutoff_freq
        )
        self.cutoff_freq = self.cutoff_freq.reshape(1, dim).to(device)  # (1, dim)
        self.sampling_freq = (
            sampling_freq.repeat(self.dim)
            if sampling_freq.numel() == 1
            else sampling_freq
        )
        self.sampling_freq = self.sampling_freq.reshape(1, dim).to(device)  # (1, dim)
        self.device = device

        self.denominator = self.init_denominator().unsqueeze(0)  # (1, dim, 2)
        self.numerator = self.init_numerator().unsqueeze(0)  # (1, dim, 2)
        self.input = torch.stack([initial_value, initial_value], dim=-1).to(
            device
        )  # (num_env, dim, 2)
        self.output = torch.stack([initial_value, initial_value], dim=-1).to(
            device
        )  # (num_env, dim, 2)

    def init_denominator(self) -> torch.Tensor:
        K = torch.tan(torch.pi * self.cutoff_freq / self.sampling_freq)  # (1, dim)
        poly = (
            K * K
            + torch.sqrt(torch.tensor(2.0, device=self.device, dtype=self.dtype)) * K
            + 1.0
        )  # (1, dim)
        denominator = torch.concat(
            [
                2.0 * (K * K - 1.0) / poly,
                (
                    K * K
                    - torch.sqrt(
                        torch.tensor(2.0, device=self.device, dtype=self.dtype)
                    )
                    * K
                    + 1.0
                )
                / poly,
            ],
            dim=0,
        ).T  # (dim, 2)
        return denominator

    def init_numerator(self) -> torch.Tensor:
        K = torch.tan(torch.pi * self.cutoff_freq / self.sampling_freq)  # (1, dim)
        poly = (
            K * K
            + torch.sqrt(torch.tensor(2.0, device=self.device, dtype=self.dtype)) * K
            + 1.0
        )  # (1, dim)
        numerator = torch.concat(
            [K * K / poly, 2.0 * K * K / poly], dim=0
        ).T  # (dim, 2)
        return numerator

    def derivative(self):
        return (
            self.output[..., 0] - self.output[..., 1]
        ) * self.sampling_freq  # (num_env, dim)

    def add(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Add a new sample to the filter and compute the filtered output.

        Args:
            sample (torch.Tensor): New input sample. (num_env, dim)
        Returns:
            torch.Tensor: Filtered output. (num_env, dim)
        """
        x2 = self.input[..., 1].clone()  # (num_env, dim)
        self.input[..., 1] = self.input[..., 0]  # (num_env, dim)
        self.input[..., 0] = sample  # (num_env, dim)
        out = self.numerator[..., 0] * x2 + (
            self.numerator * self.input - self.denominator * self.output
        ).sum(
            dim=-1
        )  # (num_env, dim)
        self.output[..., 1] = self.output[..., 0]  # (num_env, dim)
        self.output[..., 0] = out
        return out
