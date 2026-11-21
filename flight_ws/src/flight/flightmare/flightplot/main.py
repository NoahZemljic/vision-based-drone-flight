import os
import re
import argparse
import typing as T

import streamlit as st

from tracking_plotter import TrackingPlotter
from random_reference import RandomReference

@st.cache_resource
def get_all_policies(policy_path: str) -> T.List[str]:
    # flightpy runs are named PPO_%i; rpg_isaac sb3 runs (logs/sb3/<task>/<date>_<name>) also
    # carry an Eval/ folder (eval_vbdf_policy.py --flightplot). Accept any folder with Eval/.
    folders = [
        folder
        for folder in os.listdir(policy_path)
        if os.path.isdir(os.path.join(policy_path, folder, "Eval"))
    ]
    ppo = sorted([f for f in folders if re.fullmatch(r"PPO_\d+", f)], key=lambda x: int(x.split("_")[1]))
    return ppo + sorted(f for f in folders if f not in ppo)


def get_all_iterations(policy_path: str, policy_folder: int) -> T.List[int]:
    # Manually cached
    if "available_iters" in st.session_state:
        if policy_folder in st.session_state["available_iters"]:
            return st.session_state["available_iters"][policy_folder]
    else:
        st.session_state["available_iters"] = {}

    eval_iters = sorted(
        [
            int(re.search(r"\d+", filename).group(0))
            for filename in os.listdir(os.path.join(policy_path, policy_folder, "Eval"))
            if re.search(r"traj_\d+.csv", filename)
        ]
    )
    st.session_state["available_iters"][policy_folder] = eval_iters
    return eval_iters


def make_policy_selector(policy_path: str) -> bool:
    policies = get_all_policies(policy_path)
    with st.sidebar:
        st.title("Policy Selection")
        policy = st.selectbox(
            "Select policy", policies, index=len(policies) - 1, key="policy_selector"
        )

        if policy is None:
            return False

        iteration = st.select_slider(
            "Select iteration",
            options=get_all_iterations(policy_path, policy),
            key="iteration_selector",
        )

        if not iteration:
            return False

    return True


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default=None)
    args = parser.parse_args()

    make_plots = make_policy_selector(policy_path=args.policy_path)

    if "data" not in st.session_state:
        st.session_state["data"] = {}

    #  rref = RandomReference(n_points=10, bounds = [[-5, 15], [-6, 6], [1, 2]])
    #  rref.plot()

    TrackingPlotter(
        policy_path=args.policy_path,
        policy_folder=st.session_state["policy_selector"],
        iteration=st.session_state["iteration_selector"],
    )
