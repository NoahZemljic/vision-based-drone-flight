import os
import numpy as np
import torch
import rpg_assets


class ThrustMap:
    def __init__(self, csv_path: str, device="cuda", dtype=torch.float32):
        """
        Initialize the ThrustMap with a given thrust map tensor.
        """
        full_path = os.path.join(os.path.dirname(rpg_assets.__file__), csv_path)
        assert os.path.isfile(full_path), f"Thrust map file not found: {full_path}"

        data = np.loadtxt(full_path, delimiter=",")
        self.size = data[0, 0].astype(int)
        thrust_min, thrust_max, voltage_min, voltage_max = (
            data[0, 1],
            data[0, 2],
            data[0, 3],
            data[0, 4],
        )
        voltage_max += 0.5 if (voltage_max - voltage_min) < 0.5 else 0.0

        # Convert scalar parameters to tensors with proper dtype to avoid precision loss
        self.thrust_min = torch.tensor(thrust_min, device=device, dtype=dtype)
        self.thrust_max = torch.tensor(thrust_max, device=device, dtype=dtype)
        self.voltage_min = torch.tensor(voltage_min, device=device, dtype=dtype)
        self.voltage_max = torch.tensor(voltage_max, device=device, dtype=dtype)

        self.thrust_map = torch.tensor(data[1:, :], device=device, dtype=dtype).reshape(
            self.size, self.size
        )
        self.n_row, self.n_col = (
            self.thrust_map.shape
        )  # row direction: voltage (axis 0), column direction: thrust (axis 1)

        assert torch.isfinite(
            self.thrust_map
        ).all(), "Thrust map contains non-finite values."
        assert np.isfinite(
            [thrust_min, thrust_max, voltage_min, voltage_max]
        ).all(), "Thrust map parameters contain non-finite values."

        self.delta_thrust = (self.thrust_max - self.thrust_min) / (self.n_col - 1)
        self.delta_voltage = (self.voltage_max - self.voltage_min) / (self.n_row - 1)

    def map(self, thrust: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        """
        Map desired thrust and voltage to motor command using bilinear interpolation.
        """
        i_thrust = torch.clamp(
            self.n_col
            * (thrust - self.thrust_min)
            / (self.thrust_max - self.thrust_min),
            0,
            self.n_col - 2,
        ).int()
        i_voltage = torch.clamp(
            self.n_row
            * (voltage - self.voltage_min)
            / (self.voltage_max - self.voltage_min),
            0,
            self.n_row - 2,
        ).int()

        rows = torch.stack([i_voltage, i_voltage, i_voltage + 1, i_voltage + 1], dim=-1)
        cols = torch.stack([i_thrust, i_thrust + 1, i_thrust, i_thrust + 1], dim=-1)

        W = self.thrust_map[rows, cols].reshape(-1, 2, 2)

        x0_thrust = i_thrust * self.delta_thrust + self.thrust_min
        x_thrust = (thrust - x0_thrust) / self.delta_thrust
        x0_voltage = i_voltage * self.delta_voltage + self.voltage_min
        x_voltage = (voltage - x0_voltage) / self.delta_voltage

        # Bilinear interpolation
        xv_thrust = torch.stack([1 - x_thrust, x_thrust], dim=-1).unsqueeze(
            -1
        )  # shape: (num_envs, 2, 1)
        xv_voltage = torch.stack([1 - x_voltage, x_voltage], dim=-1).unsqueeze(
            -2
        )  # shape: (num_envs, 1, 2)
        # motor_command = torch.bmm(xv_voltage, torch.bmm(W, xv_thrust)).squeeze(-1).squeeze(-1) # shape: (num_envs,)
        motor_command = torch.einsum("bik,bkj,bji->b", xv_voltage, W, xv_thrust)

        return motor_command
