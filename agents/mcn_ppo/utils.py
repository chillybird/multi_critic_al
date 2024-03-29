import numpy as np
import torch
from scipy import signal


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# https://vimsky.com/examples/usage/python-scipy.signal.lfilter.html
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2, 1 + discount * x2, x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MCPPOBuffer:
    """
        A buffer for storing trajectories experienced by a PPO agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
        for calculating the advantages of state-action pairs.
        """

    def __init__(self, obs_dim, act_dim, size, r_dim, gamma=0.99, lam=0.95, device=torch.device('cpu')):
        """
        buffer initialize
        :param obs_dim:
        :param act_dim:
        :param size: buffer transition size
        :param r_dim: reward signal size
        :param gamma:
        :param lam:
        """
        self.r_dim = r_dim
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(combined_shape(size, r_dim), dtype=np.float32)
        self.ret_buf = np.zeros(combined_shape(size, r_dim), dtype=np.float32)
        self.val_buf = np.zeros(combined_shape(size, r_dim), dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        :param obs:
        :param act:
        :param rew: state-action reward
        :param val: critic value function estimation
        :param logp: action log probability
        :return:
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val[None], axis=0)
        vals = np.append(self.val_buf[path_slice], last_val[None], axis=0)
        obs = self.obs_buf[path_slice]

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = ((rews[:-1] + self.gamma * vals[1:] - vals[:-1]) * obs[:, :self.r_dim]).sum(-1)
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        self.adv_buf = torch.tensor(self.adv_buf, dtype=torch.float32).to(self.device)
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-7)
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class RolloutBuffer:
    """
    回放缓冲区
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]