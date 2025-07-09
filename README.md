# SimBa Duet for Maximizing SEE (Security Energy Efficiency)
Multi-agent scenario for uav agent and ris agent in communication when eavesdropper exist


----


## Prerequisites
Before running the script, ensure the following dependencies installed:
- Python 3.10.x
- Requirements libraries

You can install the dependencies to run:

```bash
$ pip install -r requirements.txt
```
Then you can easily train agents
```bash
$ python3 main_train.py 
```
Use metrics to see performance of your trained model 
```bash
$ python3 load_and_plot.py --path data/storage/[Algorithm]/[your data path] --ep-num [your episode]
$ python see_compare.py \
  --paths data/storage/SIMBA/simba_see_300_10_7 \
  data/storage/PPO/ppo_see_300_10 \
  data/storage/T5D/td3_see/ \
  data/storage/DDPG/ddpg_see/ \
  --labels "SIMBA" "PPO" "TTD3" "TDDRL" \
  --ep-num 300 \
  --out plot/4.png
```
---
## Reference
1. Baseline: https://github.com/yjwong1999/Twin-TD3
2. PPO: https://arxiv.org/abs/1707.06347
3. PPO implementation: https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/12-implementation-ppo.html#id4
4. SimBa: https://arxiv.org/abs/2410.09754
