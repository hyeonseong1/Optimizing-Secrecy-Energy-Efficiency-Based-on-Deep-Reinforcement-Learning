# from metrics.compute_simpl.ddpg import compute_simpl_ddpg
# from metrics.compute_simpl.simba import compute_simpl_simba
# from metrics.compute_simpl.td3 import compute_simpl_td3
#
# if __name__ == "__main__":
#     compute_simpl_simba()
#     compute_simpl_td3()
#     compute_simpl_ddpg()


import io
import re
import contextlib
from statistics import mean, stdev

from metrics.compute_simpl.ddpg import compute_simpl_ddpg
from metrics.compute_simpl.simba import compute_simpl_simba
from metrics.compute_simpl.td3 import compute_simpl_td3

# ---- 유틸: stdout 캡처 ----
def capture_stdout(func):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = func()
    return ret, buf.getvalue()

# ---- 파서: 각 알고리즘 출력에서 숫자 추출 ----
def parse_simba(text: str):
    actor = float(re.search(r"SimBa\s*Actor\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    critic = float(re.search(r"SimBa\s*Critic\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    return actor, critic

def parse_td3(text: str):
    actor = float(re.search(r"TD3\s*Actor\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    c1 = float(re.search(r"TD3\s*Critic1\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    c2 = float(re.search(r"TD3\s*Critic2\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    critic = (c1 + c2) / 2.0  # <-- 표에서는 단일 'Critic' 값으로 쓰기 위해 평균
    # 만약 Critic1/2를 분리해서 보고 싶다면 위 한 줄 대신 각각 리스트에 따로 저장하세요.
    return actor, critic

def parse_ddpg(text: str):
    actor = float(re.search(r"DDPG\s*Actor\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    critic = float(re.search(r"DDPG\s*Critic\s*simplicity\s*:\s*([0-9.]+)", text, re.I).group(1))
    return actor, critic

# ---- N회 반복 실행 ----
def run_n_times(n=10):
    res = {
        "Simba": {"Actor": [], "Critic": []},
        "TD3":   {"Actor": [], "Critic": []},
        "DDPG":  {"Actor": [], "Critic": []},
    }
    for _ in range(n):
        _, t = capture_stdout(compute_simpl_simba)
        a, c = parse_simba(t)
        res["Simba"]["Actor"].append(a)
        res["Simba"]["Critic"].append(c)

        _, t = capture_stdout(compute_simpl_td3)
        a, c = parse_td3(t)
        res["TD3"]["Actor"].append(a)
        res["TD3"]["Critic"].append(c)

        _, t = capture_stdout(compute_simpl_ddpg)
        a, c = parse_ddpg(t)
        res["DDPG"]["Actor"].append(a)
        res["DDPG"]["Critic"].append(c)
    return res

# ---- 포맷 & 출력 ----
def fmt(values):
    return f"{mean(values):.2f}±{stdev(values):.2f}"

def print_table(stats):
    headers = ["", "Simba", "TD3", "DDPG"]
    actor_row  = ["Actor",  fmt(stats["Simba"]["Actor"]), fmt(stats["TD3"]["Actor"]), fmt(stats["DDPG"]["Actor"])]
    critic_row = ["Critic", fmt(stats["Simba"]["Critic"]), fmt(stats["TD3"]["Critic"]), fmt(stats["DDPG"]["Critic"])]

    rows = [headers, actor_row, critic_row]
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def render(row):
        return " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row)))

    print(render(headers))
    print("-+-".join("-" * w for w in col_widths))
    print(render(actor_row))
    print(render(critic_row))

if __name__ == "__main__":
    stats = run_n_times(n=10)
    print_table(stats)
