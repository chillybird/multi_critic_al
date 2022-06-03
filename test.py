import numpy
import torch.nn as nn
import torch.types

from envs.mt_env import MultiTaskWrapper
from envs.minitaur_env import BulletEnv

env_build_args = {
    # 'render': True,
    'use_signal_in_observation': True,
    'use_angle_in_observation': True,
}
env_names = ["minitaur_reactive_env", "minitaur_trotting_env"]
env = MultiTaskWrapper([BulletEnv(env_name).build_env(**env_build_args) for env_name in env_names])

print(env.action_space)
print(env.observation_space)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, r_dim, has_continuous_action_space, action_std_init, device=None):
        super(ActorCritic, self).__init__()


device = torch.device('cpu')
policy = ActorCritic(1, 2, 3, 4, 5, device=6).to(device)
print(policy)
