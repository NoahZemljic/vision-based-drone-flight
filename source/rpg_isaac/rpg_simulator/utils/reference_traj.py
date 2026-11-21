import os
import csv
import torch

import rpg_simulator.utils.math_utils as math


class ReferenceTraj:
    def __init__(self, filename, device="cuda", dtype=torch.float32):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")

        with open(filename, "r") as file:
            reader = csv.reader(file)
            traj_data = [row for row in reader]
            self.data = torch.tensor(
                [[float(x) for x in row] for row in traj_data[1:]], dtype=dtype
            ).to(device)

        # reference slices into the data
        self.t = self.data[:, 0]
        self.pos = self.data[:, 1:4]
        self.q = self.data[:, 4:8]
        self.vel = self.data[:, 8:11]
        self.ome = self.data[:, 11:14]
        self.a_lin = self.data[:, 14:17]
        self.a_rot = self.data[:, 17:20]
        self.u = self.data[:, 20:24]  # individual rotor thrusts
        self.jerk = self.data[:, 24:27]
        self.snap = self.data[:, 27:30]

    def get_state_at_time(self, query_time):
        # Find the index where query_time would be inserted (upper bound)
        # Then subtract 1 to get the lower bound (last t <= query_time)
        idx = torch.searchsorted(self.t, query_time, right=True).item() - 1

        # Clamp to valid range [0, len-2] so idx+1 is always in bounds
        idx = int(max(0, min(idx, len(self.t) - 2)))

        # Interpolate between idx and idx+1
        t1, t2 = self.t[idx], self.t[idx + 1]
        alpha = (query_time - t1) / (t2 - t1)
        return self.interpolate_state(idx, idx + 1, alpha)

    def interpolate_state(self, idx1, idx2, alpha):
        # Linear interpolation for position, velocity, acceleration, jerk, snap, and rotor thrusts
        pos = (1 - alpha) * self.pos[idx1] + alpha * self.pos[idx2]
        vel = (1 - alpha) * self.vel[idx1] + alpha * self.vel[idx2]
        a_lin = (1 - alpha) * self.a_lin[idx1] + alpha * self.a_lin[idx2]
        # jerk = (1 - alpha) * self.jerk[idx1] + alpha * self.jerk[idx2]
        # snap = (1 - alpha) * self.snap[idx1] + alpha * self.snap[idx2]
        # u = (1 - alpha) * self.u[idx1] + alpha * self.u[idx2]

        # Slerp for orientation quaternion
        q1 = self.q[idx1]
        q2 = self.q[idx2]
        q_interp = math.quat_slerp(q1, q2, alpha)

        # Angular velocity interpolation
        # ome = (1 - alpha) * self.ome[idx1] + alpha * self.ome[idx2]
        action = torch.cat([pos, q_interp, vel, a_lin], dim=0).flatten()

        return action

    @property
    def total_time(self):
        return self.t[-1].item()
