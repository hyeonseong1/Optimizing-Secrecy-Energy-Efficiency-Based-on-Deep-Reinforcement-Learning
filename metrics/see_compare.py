#!/usr/bin/env python3
# see_compare.py

"""
usage:
python3 metrics/see_compare.py --paths data/storage/SIMBA/simba_see_10 data/storage/T5D/td3_see data/storage/DDPG/ddpg_see --labels "Proposed" "T5D" "TDDRL" --ep-num 300 --out plots/comparison_result.png
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ----------------------------
# UAV Energy model
# ----------------------------
P_i    = 790.6715
P_0    = 580.65
U2_tip = 200**2
s      = 0.05
d_0    = 0.3
p      = 1.225
A      = 0.79
delta_time = 0.1

m = 1.3
g = 9.81
T = m * g
v_0 = (T / (2 * A * p))**0.5

def get_energy_consumption(v_t):
    energy_1 = P_0 \
        + 3 * P_0 * (abs(v_t))**2 / U2_tip \
        + 0.5 * d_0 * p * s * A * (abs(v_t))**3

    energy_2 = P_i * ((
        (1 + (abs(v_t)**4) / (4*(v_0**4)))**0.5
        - (abs(v_t)**2) / (2*(v_0**2))
    )**0.5)

    return delta_time * (energy_1 + energy_2)


# -------------------------------------------------
# Calculate SEE curve from single scratch dataset
# -------------------------------------------------
def compute_average_see(mat_folder, ep_num):
    all_ssr = []
    all_energy = []

    for i in range(ep_num):
        fn = os.path.join(mat_folder, f"simulation_result_ep_{i}.mat")
        data = loadmat(fn)
        struct = data[f"result_{i}"][0][0]

        # 1) secure_capacity → (timesteps, user_num) or (timesteps,)
        raw_sec = struct["secure_capacity"]
        sec = np.squeeze(raw_sec)
        # sec.ndim == 2 → 여러 사용자, ==1 → 사용자 1명
        if sec.ndim == 2:
            ssr = sec.sum(axis=1)
        elif sec.ndim == 1:
            ssr = sec
        else:
            raise ValueError(f"Unexpected dimension of secure_capacity: {sec.shape}")
        all_ssr.append(ssr)

        # 2) UAV moving
        movt_raw = struct[-1]
        movt = movt_raw
        # movt_raw.ndim == 3 → [1,1,N,2]
        if movt.ndim == 3:
            movt = movt[0][0]
        # movt = (timesteps, 2)
        eps_energy = []
        for dx, dy in movt:
            v_t = np.hypot(dx, dy)
            eps_energy.append(get_energy_consumption(v_t / delta_time))
        all_energy.append(eps_energy)

    # 3) Calculate Average SEE
    avg_see = []
    for ssr_eps, en_eps in zip(all_ssr, all_energy):
        L = min(len(ssr_eps), len(en_eps))
        if L == 0:
            avg_see.append(0.0)
        else:
            see = ssr_eps[:L] / np.array(en_eps[:L])
            avg_see.append(np.mean(see))
    return np.array(avg_see) * 1000  # bits/s/Hz per kJ


# ------------------------
# Compare other algorithms
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SEE across multiple algorithms result folders"
    )
    parser.add_argument(
        "--paths", nargs="+", required=True,
        help="folders to compare (simulation_result_ep_*.mat)"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="labels for plot each folders (must match orders with paths)"
    )
    parser.add_argument(
        "--ep-num", type=int, default=300,
        help="episode num (default: 300)"
    )
    parser.add_argument(
        "--out", type=str, default="SIMBA_PPO.png",
        help="image name to save"
    )
    args = parser.parse_args()

    if len(args.paths) != len(args.labels):
        raise ValueError("--paths and --labels are not matched")

    plt.figure(figsize=(8,6))
    for path, label in zip(args.paths, args.labels):
        see_curve = compute_average_see(path, args.ep_num)
        plot_label = "Proposition" if label == "SIMBA" else label
        plt.plot(range(len(see_curve)), see_curve, label=plot_label)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.xlabel("Episodes")
    plt.ylabel("Average SEE (bits/s/Hz/kJ)")
    plt.legend()
    plt.ylim(0, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved comparison plot to {args.out}")