
#!/usr/bin/env python3
"""Parameter count table for DDPG, TD3, and SimBa models.

Usage:
  python metrics/total_params.py --repo-path /home/hs/codes/UAV-RIS-SIMBA-Journal/algorithms/

You can customize input/action dims and hidden sizes with CLI flags.
"""
import sys
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description='Count parameters of common RL models and print as a table.')
    p.add_argument('--repo-path', type=str, default='algorithms/', help='Path to repository root containing ddpg.py, td3.py, simba.py')
    p.add_argument('--input-dim', type=int, default=27)
    p.add_argument('--n-actions', type=int, default=2)
    p.add_argument('--ddpg-hidden', type=int, nargs=4, default=[400,300,256,128],
                   metavar=('FC1','FC2','FC3','FC4'),
                   help='Four hidden sizes for DDPG/TD3 networks')
    p.add_argument('--simba-hidden', type=int, default=256, help='Hidden size for SimBa Actor/Critic')
    p.add_argument('--simba-actor-blocks', type=int, default=2)
    p.add_argument('--simba-critic-blocks', type=int, default=4)
    p.add_argument('--csv', type=str, default='model_param_counts.csv', help='Optional CSV output path (set empty to skip)')
    return p.parse_args()

def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    non_trainable = total - trainable
    buffers = sum(b.numel() for b in module.buffers())
    return total, trainable, non_trainable, buffers

def fmt_table(rows, headers):
    # simple monospace table
    col_w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i,h in enumerate(headers)]
    def fmt_row(r): return ' | '.join(str(v).rjust(col_w[i]) for i,v in enumerate(r))
    sep = '-+-'.join('-'*w for w in col_w)
    out = [fmt_row(headers), sep]
    for r in rows: out.append(fmt_row(r))
    return '\n'.join(out)

def main():
    args = parse_args()
    repo = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo))

    try:
        import torch
    except Exception as e:
        print('ERROR: torch is required:', e)
        raise SystemExit(1)

    # Import model modules
    try:
        ddpg = __import__('ddpg')
    except Exception as e:
        print('ERROR: failed to import ddpg.py from', repo, '\n', e)
        raise SystemExit(1)
    try:
        td3 = __import__('td3')
    except Exception as e:
        print('ERROR: failed to import td3.py from', repo, '\n', e)
        raise SystemExit(1)
    try:
        ppo = __import__('ppo')
    except Exception as e:
        print('ERROR: failed to import ppo.py from', repo, '\n', e)
        raise SystemExit(1)
    try:
        simba = __import__('simba')
    except Exception as e:
        print('ERROR: failed to import simba.py from', repo, '\n', e)
        raise SystemExit(1)

    input_dims = args.input_dim
    n_actions = args.n_actions
    fc1, fc2, fc3, fc4 = args.ddpg_hidden
    simba_hidden = args.simba_hidden
    nblock_actor = args.simba_actor_blocks
    nblock_critic = args.simba_critic_blocks

    # Instantiate models with typical defaults
    ddpg_actor = ddpg.ActorNetwork(alpha=1e-3, input_dims=input_dims, fc1_dims=fc1, fc2_dims=fc2,
                                   fc3_dims=fc3, fc4_dims=fc4, n_actions=n_actions, name='Actor_DDPG')
    ddpg_critic = ddpg.CriticNetwork(beta=1e-3, input_dims=input_dims, fc1_dims=fc1, fc2_dims=fc2,
                                     fc3_dims=fc3, fc4_dims=fc4, n_actions=n_actions, name='Critic_DDPG')

    td3_actor = td3.ActorNetwork(alpha=1e-3, input_dims=input_dims, fc1_dims=fc1, fc2_dims=fc2,
                                 fc3_dims=fc3, fc4_dims=fc4, n_actions=n_actions, name='Actor_TD3')
    td3_critic = td3.CriticNetwork(beta=1e-3, input_dims=input_dims, fc1_dims=fc1, fc2_dims=fc2,
                                   fc3_dims=fc3, fc4_dims=fc4, n_actions=n_actions, name='Critic_TD3')

    ppo_actor = ppo.Actor(alpha=1e-3, input_dim=input_dims, hidden1_dim=fc3, hidden2_dim=fc4, n_action=n_actions)
    ppo_critic = ppo.Critic(beta=1e-3, input_dim=input_dims, hidden1_dim=fc3, hidden2_dim=fc4, n_action=n_actions)

    simba_actor = simba.Actor(alpha=1e-3, input_dim=input_dims, hidden_dim=simba_hidden,
                              n_action=n_actions, num_block=nblock_actor)
    simba_critic = simba.Critic(beta=1e-3, input_dim=input_dims, hidden_dim=simba_hidden,
                                num_block=nblock_critic)

    models = [
        ('DDPG', 'Actor',  ddpg_actor),
        ('DDPG', 'Critic', ddpg_critic),
        ('TD3',  'Actor',  td3_actor),
        ('TD3',  'Critic', td3_critic),
        ('PPO', 'Actor', ppo_actor),
        ('PPO', 'Critic', ppo_critic),
        ('SimBa(PPO)', 'Actor',  simba_actor),
        ('SimBa(PPO)', 'Critic', simba_critic),
    ]

    rows = []
    for algo, comp, m in models:
        total, trainable, non_trainable, buffers = count_params(m)
        rows.append([algo, comp, total, trainable, non_trainable, buffers, round(total/1_000_000, 3)])

    headers = ['Algorithm', 'Component', 'Total Params', 'Trainable', 'Non-trainable', 'Buffers', 'Params (M)']
    print(fmt_table(rows, headers))

    # Optional CSV output
    if args.csv:
        try:
            import csv
            with open(args.csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(headers)
                for r in rows: w.writerow(r)
            print(f"\nSaved CSV to: {args.csv}")
        except Exception as e:
            print('WARNING: failed to write CSV:', e)

if __name__ == '__main__':
    main()