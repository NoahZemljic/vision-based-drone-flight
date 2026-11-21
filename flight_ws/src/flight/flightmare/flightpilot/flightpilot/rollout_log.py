"""Deployment rollout logging in the flightpy ``Eval/`` layout, so the course's
streamlit viewer (``flightmare/flightplot``) reads a deployment like a training run::

    <run_dir>/rollout/<YYYYmmdd_HHMMSS>/Eval/traj_<n>.csv   n = number of logged steps
                                       /Eval/obs_<n>.csv    normalised policy input
                                       /config_tracking.yaml
"""
import csv
import datetime
import os

# quad_state_columns of the course rollout CSVs, minus what a vehicle cannot know.
TRAJ_COLS = [
    "t",
    "px", "py", "pz",
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz",
    "omex", "omey", "omez",
    "accx", "accy", "accz",
    "c", "wx", "wy", "wz",
]


def state_row(s, cmd):
    """flightplot row from an ``agiros_msgs/QuadState`` and the physical command."""
    p, q = s.pose.position, s.pose.orientation
    v, w, a = s.velocity.linear, s.velocity.angular, s.acceleration.linear
    return {
        "px": p.x, "py": p.y, "pz": p.z,
        "qw": q.w, "qx": q.x, "qy": q.y, "qz": q.z,
        "vx": v.x, "vy": v.y, "vz": v.z,
        "omex": w.x, "omey": w.y, "omez": w.z,
        "accx": a.x, "accy": a.y, "accz": a.z,
        "c": cmd[0], "wx": cmd[1], "wy": cmd[2], "wz": cmd[3],
    }


class RolloutLog:
    """Buffers rows in memory; ``save()`` writes one timestamped flightplot run."""

    def __init__(self, root, task, extra_cols=()):
        self.root = root
        self.task = task
        self.cols = TRAJ_COLS + [c for c in extra_cols if c not in TRAJ_COLS]
        self.rows = []
        self.obs = []
        self.t0 = None

    def append(self, t, row, obs=None):
        if self.t0 is None:
            self.t0 = t
        row["t"] = t - self.t0
        self.rows.append([float(row.get(c, 0.0)) for c in self.cols])
        if obs is not None:
            self.obs.append([float(x) for x in obs])

    def save(self):
        """Write and reset. Returns the run directory, or None if nothing was logged."""
        if not self.rows:
            return None
        run = os.path.join(self.root, f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
        ev = os.path.join(run, "Eval")
        os.makedirs(ev, exist_ok=True)
        n = len(self.rows)
        _write(os.path.join(ev, "traj_%05d.csv" % n), self.cols, self.rows)
        if self.obs:
            _write(os.path.join(ev, "obs_%05d.csv" % n), ["obs_%2i" % i for i in range(len(self.obs[0]))], self.obs)
        with open(os.path.join(run, "config_tracking.yaml"), "w") as f:  # flightplot reads rewards.components
            f.write("# written by flightpilot.rollout_log (ROS 2 deployment; not a flightpy config)\n")
            f.write("rewards:\n  components: []\n")
            f.write("main:\n  task: %s\n" % self.task)
        self.rows, self.obs, self.t0 = [], [], None
        return run


def _write(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)  # leading index column ("Unnamed: 0"), as pandas.to_csv writes
        for i, r in enumerate(rows):
            w.writerow([i] + r)
