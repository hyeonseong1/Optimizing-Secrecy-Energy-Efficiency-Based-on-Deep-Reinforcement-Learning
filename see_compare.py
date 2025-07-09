#!/usr/bin/env python3
# see_compare.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ----------------------------
# UAV 에너지 모델 (원본과 동일)
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
# 한 폴더의 .mat 파일만 읽어 SEE 커브 계산
# -------------------------------------------------
def compute_average_see(mat_folder, ep_num):
    all_ssr = []
    all_energy = []

    for i in range(ep_num):
        fn = os.path.join(mat_folder, f"simulation_result_ep_{i}.mat")
        data = loadmat(fn)
        struct = data[f"result_{i}"][0][0]

        # 1) secure_capacity 꺼내기 → (timesteps, user_num) 또는 (timesteps,)
        raw_sec = struct["secure_capacity"]
        sec = np.squeeze(raw_sec)
        # sec.ndim == 2 → 여러 사용자, ==1 → 사용자 1명
        if sec.ndim == 2:
            ssr = sec.sum(axis=1)
        elif sec.ndim == 1:
            ssr = sec
        else:
            raise ValueError(f"secure_capacity 의 차원이 예상과 다릅니다: {sec.shape}")
        all_ssr.append(ssr)

        # 2) UAV 이동 → 마지막 필드
        movt_raw = struct[-1]
        movt = movt_raw
        # movt_raw.ndim == 3 이면 [1,1,N,2] 형태일 수 있으니[0,0]
        if movt.ndim == 3:
            movt = movt[0][0]
        # 이제 movt 은 (timesteps, 2)
        eps_energy = []
        for dx, dy in movt:
            v_t = np.hypot(dx, dy)
            eps_energy.append(get_energy_consumption(v_t / delta_time))
        all_energy.append(eps_energy)

    # 3) 평균 SEE 계산
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
# 메인: 여러 알고리즘 비교
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare SEE across multiple algorithms result folders"
    )
    parser.add_argument(
        "--paths", nargs="+", required=True,
        help="비교할 결과 폴더들 (simulation_result_ep_*.mat 이 있는 디렉터리)"
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="그래프에 표시할 각 폴더의 라벨 (paths 순서와 일치해야 함)"
    )
    parser.add_argument(
        "--ep-num", type=int, default=300,
        help="에피소드 수 (default: 300)"
    )
    parser.add_argument(
        "--out", type=str, default="SIMBA_PPO.png",
        help="저장할 출력 이미지 파일명"
    )
    args = parser.parse_args()

    if len(args.paths) != len(args.labels):
        raise ValueError("--paths 와 --labels 의 개수가 같아야 합니다")

    plt.figure(figsize=(8,6))
    for path, label in zip(args.paths, args.labels):
        see_curve = compute_average_see(path, args.ep_num)
        # SIMBA는 Proposal로 이름 변경
        plot_label = "Proposal" if label == "SIMBA" else label
        plt.plot(range(len(see_curve)), see_curve, label=plot_label)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    plt.xlabel("Episodes (Ep)")
    plt.ylabel("Average SEE (bits/s/Hz/kJ)")
    plt.title("Comparison of Secrecy Energy Efficiency")
    plt.legend()
    plt.ylim(0, 60)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved comparison plot to {args.out}")
