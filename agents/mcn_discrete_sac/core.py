import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from copy import deepcopy
import gym


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class CategoricalMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.logits_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        logits = self.logits_layer(net_out)

        action_probs = F.softmax(logits, dim=-1)
        log_action_probs = F.log_softmax(logits, dim=-1)

        # Pre-squash distribution and sample
        pi_distribution = Categorical(logits=logits)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = torch.argmax(logits, dim=-1)
        else:
            pi_action = pi_distribution.sample()

        return pi_action, action_probs, log_action_probs


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q(obs)
        return q


class MultiMLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, r_dim, hidden_sizes, activation):
        super().__init__()
        self.q = nn.ModuleList()
        for _ in range(r_dim):
            self.q.append(MLPQFunction(obs_dim, act_dim, hidden_sizes, activation))

    def forward(self, obs):
        q = torch.stack([q(obs) for q in self.q],-1)
        return q


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, r_dim, hidden_sizes=(256,256), activation=nn.ReLU):
        super().__init__()

        assert len(observation_space.shape) == 1, (
            "Obs-space seems to be a matrix. You should probably use a CNNActorCritic "
            f"with observations of type {observation_space}\n"
            "(you are using `MLPActorCritic`)\n"
        )

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = CategoricalMLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1_logits = MultiMLPQFunction(obs_dim, act_dim, r_dim, hidden_sizes, activation)
        self.q2_logits = MultiMLPQFunction(obs_dim, act_dim, r_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic)
            return a.numpy()

