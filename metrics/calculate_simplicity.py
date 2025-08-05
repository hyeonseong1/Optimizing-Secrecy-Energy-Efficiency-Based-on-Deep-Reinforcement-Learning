from metrics.compute_simpl.actor_critic import compute_simpl_AC
from metrics.compute_simpl.simba import compute_simpl_simba
from metrics.compute_simpl.td3 import compute_simpl_td3

if __name__ == "__main__":
    compute_simpl_simba()
    compute_simpl_td3()
    compute_simpl_AC()

# TODO: Plot boxplot of 3 models
# return format: dict
#     return {
#         "simplicity_actor": simp_actor,
#         "simplicity_critic": simp_critic,
#         "c_values_actor": c_actor,
#         "c_values_critic": c_critic,
#     }