import gym
import numpy as np


class Env(object):
    """
    # 环境中的智能体
    """
    def __init__(self, i):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 14  # 设置智能体的观测纬度
        self.action_dim = 5  # 设置智能体的动作纬度，这里假定为一个五个纬度的

        self._action_space = []
        self._observ_space = []

    @property
    def action_space(self):

        for i in range(self.agent_num):
            self._action_space.append(
                gym.spaces.Box(-np.inf * np.ones(self.action_dim), np.inf * np.ones(self.action_dim), dtype=np.float32))

        return self._action_space

    @property
    def observation_space(self):

        for i in range(self.agent_num):
            self._observ_space.append(
                gym.spaces.Box(-np.ones(self.obs_dim), np.ones(self.obs_dim), dtype=np.float32))

        return self._observ_space

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


if __name__ == '__main__':
    env = Env(1)
    for line in env.step(0):
        print(line)