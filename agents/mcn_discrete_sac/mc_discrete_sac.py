from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

import gym
import time
import spinup.algos.pytorch.mcn_discrete_sac.core as core
from agents.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, r_dim, size):
        self.r_dim = r_dim
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, r_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

"""
Store the observations in ring buffer type array of size m
"""
class StateBuffer:
    def __init__(self,m):
        self.m = m

    def init_state(self, init_obs):
        self.current_state = np.concatenate([init_obs]*self.m, axis=0)
        return self.current_state

    def append_state(self, obs):
        new_state = np.concatenate( (self.current_state, obs), axis=0)
        self.current_state = new_state[obs.shape[0]:]
        return self.current_state

"""
Process features of the environment
"""
def process_observation(o, obs_dim, observation_type):
    if observation_type == 'Discrete':
        o = np.eye(obs_dim)[o]
    return o

def process_action(a, act_dim):
    one_hot_a = np.eye(act_dim)[a]
    return one_hot_a

"""
Linear annealing from start to stop value based on current step and max_steps
"""
def linear_anneal(current_step, start=0.1, stop=1.0, steps=1e6):
    if current_step<=steps:
        eps = stop + (start - stop) * (1 - current_step/steps)
    else:
        eps=start
    return eps


def mc_discrete_sac(env_fn, ac_kwargs=dict(), seed=0, 
                    steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
                    polyak=0.995, lr=1e-3, batch_size=256, start_steps=10000, 
                    update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
                    target_entropy_start=0.3, target_entropy_stop=0.3, target_entropy_steps=1e5, alpha='auto',
                    logger_kwargs=dict(), save_freq=1, retro=0):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    if retro > 0:
        test_env = env
    else:
        test_env = env_fn()

    _ = env.reset()
    _,_,_,e_info = env.step(env.action_space.sample())
    if 'RewardBreakdown' in e_info.keys():
        n_rewards = len(e_info['RewardBreakdown'])

    # Create actor-critic module and target networks
    if len(obs_dim) == 1:
        img_in = False
        actor_critic = core.MLPActorCritic
    else:
        img_in = True
        actor_critic = core.CNNActorCritic
    ac = actor_critic(env.observation_space, env.action_space, n_rewards, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1_logits.parameters(), ac.q2_logits.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, r_dim=n_rewards)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1_logits, ac.q2_logits])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # alpha and entropy setup
    max_target_entropy = torch.log(torch.tensor(act_dim, dtype=torch.float32))
    log_alpha = torch.zeros(1, requires_grad=True)
    if alpha == 'auto':
        _alpha = lambda : log_alpha.exp()
    elif isinstance(alpha, float):
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        _alpha = lambda : alpha_tensor
    else:
        raise ValueError('Alpha is weird')

    def compute_loss_alpha(data, t=None):
        o, a = data['obs'], data['act']
        with torch.no_grad():
            _, action_probs, log_action_probs = ac.pi(o)
            if t is None:
                t = 0
            target_entropy = linear_anneal(t, target_entropy_start, target_entropy_stop, target_entropy_steps) * max_target_entropy
            pi_entropy = -(action_probs * log_action_probs).sum(dim=-1)
            alpha_backup = target_entropy - pi_entropy

        loss_alpha = -(log_alpha * alpha_backup).mean()

        alpha_info = dict(Alpha=_alpha().detach().numpy(), PiEntropy=pi_entropy.detach().numpy(), TargetEntropy=target_entropy.detach().numpy())

        return loss_alpha, alpha_info

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1_logits = ac.q1_logits(o)
        q2_logits = ac.q2_logits(o)

        W = o[:,:n_rewards]
        w = torch.transpose(W[...,None], -1, -2)

        q1_a = ((q1_logits * w) * a[...,None]).sum(dim=-2)
        q2_a = ((q2_logits * w) * a[...,None]).sum(dim=-2)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            _, p_a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_logits_pi_targ = ac_targ.q1_logits(o2)
            q2_logits_pi_targ = ac_targ.q2_logits(o2)
            q_logits_pi_targ = torch.min(q1_logits_pi_targ, q2_logits_pi_targ)

            backup = r + gamma * (1 - d[...,None]) * (p_a2[...,None] * (q_logits_pi_targ - (_alpha() * logp_a2)[...,None])).sum(dim=-2)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(W*q1_a, W*backup)
        loss_q2 = F.mse_loss(W*q2_a, W*backup)
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1_a.detach().numpy(),
                      Q2Vals=q2_a.detach().numpy(),
                      LossQ1=loss_q1.detach().numpy(),
                      LossQ2=loss_q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, p_pi, logp_pi = ac.pi(o)
        q1_logits_pi = ac.q1_logits(o)
        q2_logits_pi = ac.q2_logits(o)
        q_logits_pi = torch.min(q1_logits_pi, q2_logits_pi)

        w = torch.transpose(o[:,:n_rewards,None], -1, -2)
        Q_logits_pi = (q_logits_pi * w).sum(-1)

        # Entropy-regularized policy loss
        pi_backup = (p_pi * (_alpha() * logp_pi - Q_logits_pi)).sum(dim=-1)
        loss_pi = pi_backup.mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr, eps=1e-4)
    q_optimizer = Adam(q_params, lr=lr, eps=1e-4)
    alpha_optimizer = Adam([log_alpha], lr=lr, eps=1e-4)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, t=None):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Freeze Actor network 
        for p_a in ac.pi.parameters():
            p_a.requires_grad = False

        # Alplha optimizer
        alpha_optimizer.zero_grad()
        loss_alpha, alpha_info = compute_loss_alpha(data, t)
        loss_alpha.backward()
        alpha_optimizer.step()

        # Unfreeze Actor Network
        for p_a in ac.pi.parameters():
            p_a.requires_grad = True

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        logger.store(LossAlpha=loss_alpha.item(), **alpha_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)


    def get_action(o, deterministic=False):
        if not img_in:
            o_step = torch.as_tensor(o, dtype=torch.float32)
        else:
            o_step = torch.as_tensor(o[None,...], dtype=torch.float32)
        return ac.act(o_step, deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, _, d, info = test_env.step(get_action(o, True))
                r = info['RewardBreakdown'].copy()
                ep_ret += (o[:n_rewards] * r).sum()
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len, reward_breakdown = env.reset(), 0, 0, np.zeros(n_rewards)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, _, d, info = env.step(a)
        a = process_action(a, act_dim)
        r = info['RewardBreakdown'].copy()
        reward_breakdown += r
        ep_ret += (r * o[:n_rewards]).sum()
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # Update handling
            if t >= update_after:
                for j in range(ep_len):
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch, t=t)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            for i in range(n_rewards):
                exec("logger.store(Rew%d=%f)" % (i, reward_breakdown[i]))
            o, ep_ret, ep_len, reward_breakdown = env.reset(), 0, 0, np.zeros(n_rewards)


        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            if retro > 0:
                o, ep_ret, ep_len = env.reset(), 0, 0

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            if n_rewards:
                for i in range(n_rewards):
                    exec("logger.log_tabular('Rew%d', with_min_and_max=True)" % (i))
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TargetEntropy', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from agents.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    mc_discrete_sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
                    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                    gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                    logger_kwargs=logger_kwargs)
