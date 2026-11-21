import os
import pandas as pd
import numpy as np
import yaml

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
from ipywidgets import interactive


class TrackingPlotter:
    def __init__(self, policy_path: str, policy_folder: str, iteration: int):
        self.policy_path = policy_path
        self.policy_folder = policy_folder
        self.iteration = iteration

        self.eval_folder = os.path.join(policy_path, policy_folder, "Eval")
        self.traj_file = os.path.join(self.eval_folder, "traj_%05d.csv" % iteration)
        self.obs_file = os.path.join(self.eval_folder, "obs_%05d.csv" % iteration)
        self.data, self.obs = self._load()

        self.config_file = os.path.join(policy_path, policy_folder, "config_tracking.yaml")
        self.config = self._load_config()

        col_left, col_right = st.columns([0.7, 0.3])
        with col_left:
            st.title(f"Policy {policy_folder} at iteration {iteration}")

        with col_right:
            selected_plot = st.radio(
                "Select plot",
                ["2D Plots", "3D Plots", "Reward Plot", "Obs Plot", "Plot Anything"],
                key="plot_selector",
                horizontal=True,
            )

        if selected_plot == "2D Plots":
            self.plot_2d()
        elif selected_plot == "3D Plots":
            self.plot_3d()
        elif selected_plot == "Reward Plot":
            self.plot_reward()
        elif selected_plot == "Obs Plot":
            self.plot_obs()
        elif selected_plot == "Plot Anything":
            self.plot_anything()

    def _load(self):
        # Built-in caching
        if self.traj_file in st.session_state["data"]:
            return st.session_state["data"][self.traj_file], st.session_state["data"][self.obs_file]
        else:
            df_eval = pd.read_csv(self.traj_file)
            df_eval["idx"] = df_eval["Unnamed: 0"].astype(np.float64)
            st.session_state["data"][self.traj_file] = df_eval

            if os.path.exists(self.obs_file):
                df_obs = pd.read_csv(self.obs_file)
                df_obs.drop(columns=["Unnamed: 0"], inplace=True)
            else:
                df_obs = pd.DataFrame()
            st.session_state["data"][self.obs_file] = df_obs
            return df_eval, df_obs

    def _load_config(self):
        with open(self.config_file, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def _get_G10_color(self, idx: int) -> str:
        return px.colors.qualitative.G10[idx % 10]

    def plot_2d(self):
        # Make 4x3 subplot matrix with shared x-axis
        fig = make_subplots(
            rows=4,
            cols=12,
            specs=[
                [
                    {"colspan": 4},
                    None,
                    None,
                    None,
                    {"colspan": 4},
                    None,
                    None,
                    None,
                    {"colspan": 4},
                    None,
                    None,
                    None,
                ]
            ]
            * 3
            + [
                [
                    {"colspan": 3},
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                    {"colspan": 3},
                    None,
                    None,
                ]
            ],
            subplot_titles=9 * [None] + ["Thrust", "wx", "wy", "wz"],
            shared_xaxes=True,
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            column_titles=["X", "Y", "Z"],
        )

        # Add traces position, velocity, acceleration for ego drone
        for row_idx, quantity in enumerate(["p", "v", "acc"]):
            for axis_idx, axis_name in enumerate("xyz"):
                trace = go.Scatter(
                    y=self.data[f"{quantity}{axis_name}"],
                    mode="lines",
                    line=dict(color=self._get_G10_color(0)),
                    legendgroup="drone",
                    name="Ego Drone",
                    showlegend=row_idx + axis_idx == 0,
                )
                fig.add_trace(trace, row=row_idx + 1, col=4 * axis_idx + 1)

        # COMMENT BELOW TO PLOT THE POSITION OF THE ACTOR DRONE
        # BUT FOR THIS YOU MUST ALSO INCLUDE IT IN THE OUTPUT
        # FOR EXAMPLE, THROUGH THE EXTRA INFO MECHANISM.
        # Add traces for position of other drone
        #  for axis_idx, axis_name in enumerate("xyz"):
            #  trace = go.Scatter(
                #  y=self.data[f"other_pos_{axis_name}"],
                #  mode="lines",
                #  line=dict(color=self._get_G10_color(1)),
                #  legendgroup="other",
                #  name="Other Drone",
                #  showlegend=axis_idx == 0,
            #  )
            #  fig.add_trace(trace, row=1, col=4 * axis_idx + 1)

        # Add traces for control
        for axis_idx, axis_name in enumerate(["c", "wx", "wy", "wz"]):
            trace = go.Scatter(
                y=self.data[axis_name],
                mode="lines",
                line=dict(color=self._get_G10_color(2)),
                legendgroup="control",
                name="Command",
                showlegend=axis_idx == 0,
            )
            fig.add_trace(trace, row=4, col=3 * axis_idx + 1)

        fig.update_layout(
            yaxis_title="Position",
            yaxis4_title="Velocity",
            yaxis7_title="Acceleration",
            yaxis10_title="Control",
            showlegend=True,
            height=900,
        )
        st.plotly_chart(fig, width="stretch")  # use_container_width= removed after streamlit 1.4x

    def plot_3d(self):
        # Convert dataframe to suitable format for plotly
        traj_length = 10
        ego_df = pd.DataFrame(columns=["idx", "px", "py", "pz", "drone", "size"])
        ego_df["idx"] = (self.data["idx"] // traj_length).astype(int)
        ego_df["px"] = self.data["px"]
        ego_df["py"] = self.data["py"]
        ego_df["pz"] = self.data["pz"]
        ego_df["drone"] = "ego"
        ego_df["size"] = self.data["idx"] % traj_length + 1

        # COMMENT BELOW TO PLOT THE POSITION OF THE ACTOR DRONE
        # BUT FOR THIS YOU MUST ALSO INCLUDE IT IN THE OUTPUT
        # FOR EXAMPLE, THROUGH THE EXTRA INFO MECHANISM.
        #  other_df = pd.DataFrame(columns=["idx", "px", "py", "pz", "drone", "size"])
        #  other_df["idx"] = (self.data["idx"] // traj_length).astype(int)
        #  other_df["px"] = self.data["other_pos_x"]
        #  other_df["py"] = self.data["other_pos_y"]
        #  other_df["pz"] = self.data["other_pos_z"]
        #  other_df["drone"] = "other"
        #  other_df["size"] = self.data["idx"] % traj_length + 1
        #  df = pd.concat([ego_df, other_df])
        df = ego_df

        range_x = [df["px"].min() - 0.2, df["px"].max() + 0.2]
        range_y = [df["py"].min() - 0.2, df["py"].max() + 0.2]
        range_z = [df["pz"].min() - 0.2, df["pz"].max() + 0.2]

        fig = px.scatter_3d(
            df,
            x="px",
            y="py",
            z="pz",
            animation_frame="idx",
            animation_group="drone",
            size="size",
            color="drone",
            color_discrete_map={"ego": "blue", "other": "red"},
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
        )
        fig.add_trace(
            go.Scatter3d(
                x=self.data["px"],
                y=self.data["py"],
                z=self.data["pz"],
                mode="lines",
                line=dict(color="blue", width=1.0),
                name="Ego Drone",
            )
        )
        #  fig.add_trace(
            #  go.Scatter3d(
                #  x=self.data["other_pos_x"],
                #  y=self.data["other_pos_y"],
                #  z=self.data["other_pos_z"],
                #  mode="lines",
                #  line=dict(color="red", width=1.0),
                #  name="Other Drone",
            #  )
        #  )
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube",
                xaxis=dict(showspikes=False),
                yaxis=dict(showspikes=False),
                zaxis=dict(showspikes=False),
            ),
            showlegend=True,
            hovermode=False,
            height=800,
            margin=dict(l=20, r=20, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")  # use_container_width= removed after streamlit 1.4x

    def plot_reward(self):
        reward_names = self.config["rewards"]["components"]
        use_log_scale = st.toggle("Use log scale", False)

        fig = go.Figure()
        for reward_name in reward_names:
            if use_log_scale:
                fig.add_trace(
                    go.Scatter(
                        x=self.data["idx"],
                        y=np.abs(self.data[reward_name]),
                        mode="lines",
                        name=reward_name,
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=self.data["idx"], y=self.data[reward_name], mode="lines", name=reward_name,
                    )
                )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
            height=600,
        )
        fig.update_yaxes(type="log" if use_log_scale else "linear")
        st.plotly_chart(fig, width="stretch")  # use_container_width= removed after streamlit 1.4x

    def plot_obs(self):
        # Make nx2 subplots, such that always 6 curves are there per plot
        # plot all columns of self.obs

        n_plots = len(self.obs.columns) // 6 + 1
        fig = make_subplots(
            rows=int(np.ceil(n_plots/2)),
            cols=2,
            shared_xaxes=True,
            subplot_titles=self.obs.columns,
            vertical_spacing=0.1,
        )

        for i, col in enumerate(self.obs.columns):
            fig.add_trace(
                go.Scatter(x=self.obs.index, y=self.obs[col], mode="lines", name=col),
                row=i // 12 + 1,
                col=(i // 6) % 2 + 1,
            )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
            height=800,
        )
        st.plotly_chart(fig, width="stretch")  # use_container_width= removed after streamlit 1.4x



    def plot_anything(self):
        selected_x = st.selectbox(
            "Select x-axis", self.data.columns, index=len(self.data.columns) - 1
        )
        selected_cols = st.multiselect("Select columns", self.data.columns)

        fig = go.Figure()
        for col in selected_cols:
            fig.add_trace(
                go.Scatter(x=self.data[selected_x], y=self.data[col], mode="lines", name=col)
            )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True,
            height=600,
        )
        st.plotly_chart(fig, width="stretch")  # use_container_width= removed after streamlit 1.4x
