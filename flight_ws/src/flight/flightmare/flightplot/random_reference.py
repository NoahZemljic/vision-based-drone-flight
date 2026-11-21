import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typing as T


class RandomReference:
    def __init__(self, n_points: int, bounds: T.List[T.List[float]]):
        self.bounds = bounds
        self.tmax = 30
        self.min_vel = 0
        self.max_vel = 8

        self.x, self.y, self.z = self.sample_points(bounds, self.min_vel, self.max_vel, self.tmax)
        self.n_points = len(self.x)

        self.t = np.linspace(0, self.tmax, self.n_points)
        self.t_interp = np.linspace(0, self.tmax, 1000)
        self.x_interp = self.interp_cubic(self.t, self.x, self.t_interp)
        self.y_interp = self.interp_cubic(self.t, self.y, self.t_interp)
        self.z_interp = self.interp_cubic(self.t, self.z, self.t_interp)
        self.r_interp = self.get_rotation(
            self.t_interp, self.x_interp, self.y_interp, self.z_interp
        )

    def interp_cubic(
        self, xv: np.ndarray, yv: np.ndarray, xq: T.Optional[np.ndarray] = None
    ) -> np.ndarray:
        n = len(xv)
        a = np.zeros((n))
        b = np.zeros((n))
        c = np.zeros((n))
        d = np.zeros((n))

        h = np.diff(xv)
        a[n - 1] = h[n - 2] / 6
        b[0] = h[0] / 3
        b[n - 1] = h[n - 2] / 3
        c[0] = h[0] / 6
        d[0] = (yv[1] - yv[0]) / h[0]
        d[n - 1] = (yv[n - 2] - yv[n - 1]) / h[n - 2]
        for i in range(1, n - 1):
            a[i] = h[i - 1] / 6
            b[i] = (h[i - 1] + h[i]) / 3
            c[i] = h[i] / 6
            d[i] = (yv[i + 1] - yv[i]) / h[i] - (yv[i] - yv[i - 1]) / h[i - 1]

        # Thomas algorithm
        c[0] = c[0] / b[0]
        d[0] = d[0] / b[0]
        for i in range(1, n):
            c[i] = c[i] / (b[i] - a[i] * c[i - 1])
            d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1])
        m = np.zeros(n)
        m[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            m[i] = d[i] - c[i] * m[i + 1]

        for i in range(0, n - 1):
            c[i] = (yv[i + 1] - yv[i]) / h[i] - h[i] * (m[i + 1] - m[i]) / 6
            d[i] = yv[i] - m[i] * h[i] ** 2 / 6

        # Interpolation
        if xq is None:
            return m, c, d

        yq = np.zeros((len(xq), 3))
        for i in range(0, len(xq)):
            for j in range(n - 1):
                if xq[i] >= xv[j] and xq[i] <= xv[j + 1]:
                    yq[i, 0] = (
                        m[j] * (xv[j + 1] - xq[i]) ** 3 / (6 * h[j])
                        + m[j + 1] * (xq[i] - xv[j]) ** 3 / (6 * h[j])
                        + c[j] * (xq[i] - xv[j])
                        + d[j]
                    )
                    yq[i, 1] = (
                        -((xv[j + 1] - xq[i]) ** 2) / (2 * h[j]) * m[j]
                        + (xq[i] - xv[j]) ** 2 / (2 * h[j]) * m[j + 1]
                        + c[j]
                    )
                    yq[i, 2] = m[j] * (xv[j + 1] - xq[i]) / h[j] + m[j + 1] * (xq[i] - xv[j]) / h[j]
                    break
        return yq

    def sample_points(
        self,
        bounds: T.List[T.List[float]],
        min_vel: float,
        max_vel: float,
        tmax: float,
    ) -> np.ndarray:

        # Expectation values for the distance between two points
        #  https://en.wikipedia.org/wiki/Triangular_distribution
        E_x = 1 / 3 * (bounds[0][1] - bounds[0][0])
        E_y = 1 / 3 * (bounds[1][1] - bounds[1][0])
        E_z = 1 / 3 * (bounds[2][1] - bounds[2][0])

        # Crude (and slightly wrong) way to calculate the expected points distance
        E_d = np.sqrt(E_x**2 + E_y**2 + E_z**2)
        n_points = int(0.5 * tmax / E_d * (max_vel + min_vel))
        x = np.random.uniform(bounds[0][0], bounds[0][1], n_points)
        y = np.random.uniform(bounds[1][0], bounds[1][1], n_points)
        z = np.random.uniform(bounds[2][0], bounds[2][1], n_points)
        return x, y, z

    def get_rotation(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        rot_mat = np.zeros((len(t), 3, 3))
        for i in range(len(t)):
            e_x = np.array([x[i, 1], y[i, 1], z[i, 1]])
            e_x /= np.linalg.norm(e_x)
            e_z = np.array([x[i, 2], y[i, 2], z[i, 2] + 9.81])
            e_z /= np.linalg.norm(e_z)
            e_y = np.cross(e_z, e_x)
            rot_mat[i] = np.array([e_x, e_y, e_z])
        return rot_mat

    def plot(self):
        col_left, col_right = st.columns(2)

        with col_left:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=self.x, y=self.y, z=self.z, mode="markers"))
            fig.add_trace(
                go.Scatter3d(
                    x=self.x_interp[:, 0],
                    y=self.y_interp[:, 0],
                    z=self.z_interp[:, 0],
                    mode="lines",
                )
            )
            fig.update_layout(
                scene=dict(
                    aspectmode="cube",
                    xaxis=dict(range=self.bounds[0]),
                    yaxis=dict(range=self.bounds[1]),
                    zaxis=dict(range=self.bounds[2]),
                ),
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Plot acceleration
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig.add_trace(
                go.Scatter(
                    x=self.t_interp,
                    y=np.sqrt(
                        self.x_interp[:, 2] ** 2
                        + self.y_interp[:, 2] ** 2
                        + self.z_interp[:, 2] ** 2
                    ),
                    name="Acceleration",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.t_interp,
                    y=np.sqrt(
                        self.x_interp[:, 1] ** 2
                        + self.y_interp[:, 1] ** 2
                        + self.z_interp[:, 1] ** 2
                    ),
                    name="Velocity",
                ),
                row=2,
                col=1,
            )
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Acceleration",
                yaxis2_title="Velocity",
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)
