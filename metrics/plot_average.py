#!/usr/bin/env python3
# see_compare.py (multi-run under each algo)

"""
usage examples:

# 1) 각 알고리즘 폴더의 모든 하위 폴더를 런으로 간주(최대 5개까지)
python3 metrics/plot_average.py \
  --paths data/storage/DDPG data/storage/TD3 data/storage/PPO data/storage/SIMBA\
  --labels "TDDRL" "T5D" "DPPO" "LSPPO(Ours)" \
  --ep-num 300 --out plots/comparison_result.png
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ----------------------------
# UAV Energy model (원본 유지)
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
# Calculate SEE curve from a single run folder
# -------------------------------------------------
def compute_average_see(mat_folder, ep_num):
    all_ssr = []
    all_energy = []

    for i in range(ep_num):
        fn = os.path.join(mat_folder, f"simulation_result_ep_{i}.mat")
        if not os.path.exists(fn):
            # 파일이 없으면 0 처리
            all_ssr.append(np.array([]))
            all_energy.append([])
            continue

        data = loadmat(fn)
        struct = data.get(f"result_{i}")
        if struct is None:
            all_ssr.append(np.array([]))
            all_energy.append([])
            continue
        struct = struct[0][0]

        # 1) secure_capacity → (timesteps, user_num) or (timesteps,)
        raw_sec = struct["secure_capacity"]
        sec = np.squeeze(raw_sec)
        if sec.ndim == 2:
            ssr = sec.sum(axis=1)
        elif sec.ndim == 1:
            ssr = sec
        else:
            raise ValueError(f"Unexpected dimension of secure_capacity: {sec.shape}")
        all_ssr.append(ssr)

        # 2) UAV moving (원 코드 관성 유지: 마지막 필드)
        movt_raw = struct[-1]
        movt = movt_raw
        if movt.ndim == 3:
            movt = movt[0][0]
        # movt = (timesteps, 2)
        eps_energy = []
        for dx, dy in movt:
            v_t = np.hypot(dx, dy)
            eps_energy.append(get_energy_consumption(v_t / delta_time))
        all_energy.append(eps_energy)

    # 3) Calculate Average SEE per episode
    avg_see = []
    for ssr_eps, en_eps in zip(all_ssr, all_energy):
        L = min(len(ssr_eps), len(en_eps))
        if L == 0:
            avg_see.append(0.0)
        else:
            see = ssr_eps[:L] / np.array(en_eps[:L])
            avg_see.append(np.mean(see))
    return np.array(avg_see) * 1000.0  # bits/s/Hz per kJ


# -------------------------------------------------
# Find run folders under an algorithm base path
# -------------------------------------------------
def find_run_dirs(algo_base_dir, run_glob=None, max_runs=5):
    """
    algo_base_dir 하위의 직속 디렉토리를 '런(=시드)'으로 간주.
    - run_glob이 주어지면 해당 패턴에 매칭되는 하위 디렉토리만 사용
    - 사전순 정렬 후 최대 max_runs 개까지 사용
    - 하위 디렉토리를 못 찾으면 algo_base_dir 자체를 단일 런으로 간주
    """
    if not os.path.isdir(algo_base_dir):
        return []

    if run_glob:
        candidates = sorted(
            d for d in glob.glob(os.path.join(algo_base_dir, run_glob))
            if os.path.isdir(d)
        )
    else:
        # 직속 하위 디렉토리 전부
        candidates = sorted(
            os.path.join(algo_base_dir, d)
            for d in os.listdir(algo_base_dir)
            if os.path.isdir(os.path.join(algo_base_dir, d))
        )

    runs = candidates[:max_runs]
    if not runs:
        # fallback: .mat가 바로 들어있는 단일 런 구조
        return [algo_base_dir]
    return runs


# ------------------------
# Compare algorithms (multi-run)
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare SEE across multiple algorithms (aggregate multiple runs per algorithm)"
    )
    parser.add_argument(
        "--paths", nargs="+", required=True,
        help="algorithm base folders (each contains run subfolders like simba_see, simba_see_1, ...)"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="labels for plot (must match orders with paths)"
    )
    parser.add_argument(
        "--ep-num", type=int, default=300,
        help="episode num (default: 300)"
    )
    parser.add_argument(
        "--out", type=str, default="SIMBA_PPO.png",
        help="image name to save"
    )
    parser.add_argument(
        "--max-runs", type=int, default=5,
        help="max number of runs per algorithm to aggregate (default: 5)"
    )
    parser.add_argument(
        "--run-glob", type=str, default=None,
        help='glob pattern to select run dirs under each algo path (e.g., "simba_see*")'
    )
    parser.add_argument(
        "--show-runs", action="store_true",
        help="plot individual run curves as thin lines"
    )
    args = parser.parse_args()

    if len(args.paths) != len(args.labels):
        raise ValueError("--paths and --labels are not matched")

    plt.figure(figsize=(8, 6))
    x = np.arange(args.ep_num)

    for algo_path, label in zip(args.paths, args.labels):
        run_dirs = find_run_dirs(algo_path, args.run_glob, args.max_runs)
        if len(run_dirs) == 0:
            print(f"[WARN] No valid runs found in: {algo_path}")
            continue

        run_curves = []
        for run_dir in run_dirs:                      # 1중: 알고리즘별
            see_curve = compute_average_see(run_dir, args.ep_num)  # 2중: 런(폴더)별
            run_curves.append(see_curve)

        run_curves = np.stack(run_curves, axis=0)  # (R, E)
        mean_curve = run_curves.mean(axis=0)
        std_curve  = run_curves.std(axis=0)

        # 개별 런 곡선(얇게)
        if args.show_runs and len(run_dirs) > 1:
            for rc in run_curves:
                plt.plot(x, rc, linewidth=1, alpha=0.35)

        # 라벨 치환(원 코드 유지)
        plot_label = "Proposition" if label in ("SIMBA", "Proposed") else label

        # 평균 곡선 + 표준편차 음영
        plt.plot(x, mean_curve, label=f"{plot_label}", linewidth=2.2)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.xlabel("Episodes")
    plt.ylabel("Average SEE (bits/s/Hz/kJ)")
    plt.legend()
    plt.ylim(0, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
