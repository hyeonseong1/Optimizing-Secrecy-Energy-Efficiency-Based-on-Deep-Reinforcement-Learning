# Optimizing Secrecy Energy Efficiency Based on Deep Reinforcement Learning
Multi-agent scenario for uav agent and ris agent in communication when eavesdropper exist

### Current Process: Prepairing a paper to submit.

----

## Prerequisites
Before running the script, ensure the following dependencies installed:
- Python 3.10.x
- requirements.txt

You can install the dependencies to run:

```commandline
conda create -n uavris python=3.10
conda activate uavris
pip3 install -r requirements.txt
```
(Optional) You can install different version of torch with compatible cuda version. 
```commandline
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Then train multi-agent with 4 types of algorithms:
```commandline
##### LSPPO #####
python3 main_train.py --algo SIMBA
##### PPO #####
python3 main_train.py --algo PPO
##### TTD3 #####
python3 main_train.py --algo TD3
##### TDDRL #####
python3 main_train.py --algo DDPG
```
Use metrics to see performances or parameters of your trained model 
```commandline
python3 metrics/load_and_plot.py --path data/storage/[algorithm]/[your data path]
python3 metrics/plot_average.py   --paths data/storage/DDPG data/storage/TD3 data/storage/PPO data/storage/SIMBA  --labels "TDDRL" "T5D" "DPPO" "LSPPO(Ours)"   --ep-num 300 --out plots/comparison_result.png

python3 calculate_simplicity.py
python3 metrics/total_params.py 
```
---
## Reference
1. Baseline: https://github.com/yjwong1999/Twin-TD3
2. PPO: https://arxiv.org/abs/1707.06347
3. PPO codes reference: https://hiddenbeginner.github.io/Deep-Reinforcement-Learnings/book/Chapter2/12-implementation-ppo.html#id4
4. SimBa: https://arxiv.org/abs/2410.09754
5. Neural Redshift: https://arxiv.org/abs/2403.02241
