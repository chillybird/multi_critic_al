import numpy as np
import gym
from gym import spaces


class MultiTaskWrapper(gym.Wrapper):
    def __init__(self, envs):
        assert isinstance(envs, list)
        super(MultiTaskWrapper, self).__init__(envs[0])
        self.envs = envs
        self.env_num = len(envs)
        # current choose env
        self.env_choice = 0
        observation_space = self.env.observation_space
        # print("observation space: ", observation_space)
        self.observation_space = spaces.Box(low=np.concatenate(([0]*self.env_num, observation_space.low)),
                                            high=np.concatenate(([1.]*self.env_num, observation_space.high)),
                                            dtype=np.float32)

    def reset(self, selection=None):
        if selection is None:
            selection = np.random.randint(self.env_num)
        self.env_choice = selection
        indicator = [0]*self.env_num
        indicator[self.env_choice] = 1
        self.env = self.envs[self.env_choice]
        obs = self.env.reset().copy()
        return np.concatenate((indicator, obs))

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if 'RewardBreakdown' in info.keys():
            info['RewardBreakdown'] = np.concatenate((np.eye(self.env_num)[self.env_choice]*rew,
                                                      info['RewardBreakdown']))
        else:
            info['RewardBreakdown'] = np.eye(self.env_num)[self.env_choice]*rew
        indicator = [0]*self.env_num
        indicator[self.env_choice] = 1
        obs_return = np.concatenate((indicator, obs))
        return obs_return, rew, done, info

    def set_direction(self, direction, **kwargs):
        self.env.set_direction(direction, **kwargs)
