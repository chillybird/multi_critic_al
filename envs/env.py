# -*- coding:utf-8 -*-
# @Time : 2022/3/3 19:23
# @Author: zhcode
# @File : env.py

from envs.minitaur_env import BulletMultiEnv


class Env(object):
    def __init__(self, idx, args):
        self.idx = idx
        self.args = args

        if isinstance(args.env_name, str):
            base_env = BulletMultiEnv(args.env_name)
        else:
            base_env = BulletMultiEnv(args.env_name[idx % len(args.env_name)])
        if args.eval_env:
            print(f"use eval env {args.env_name if isinstance(args.env_name, str) else args.env_name[0]}.")

        env_args = {
            'render': True if (idx == 0 and args.eval_env and args.use_render) else False,
            # 'render': True if args.eval_env and args.use_render else False,
            'random_start': args.random_start,
            'urdf_version': args.urdf_version,
            'max_length': args.max_length,
            'multi_task': args.multi_task,
            'use_signal_in_observation': args.use_signal_in_observation,
            'use_angle_in_observation': args.use_angle_in_observation,
        }
        self._env = base_env.build_env(**env_args)
        self.agent_num = len(self._env.action_space)

        # print("action space: ", self._env.action_space)
        # print("observation_space:", self._env.observation_space)
        # print("share_observation_space: ", self._env.share_observation_space)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step(self, actions):

        sub_agent_obs, sub_agent_share_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_avail_action = self._env.step(actions)
        # 修改返回奖励的结构
        sub_agent_reward = sub_agent_reward.reshape(self.agent_num, 1)
        sub_agent_done = [sub_agent_done for i in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_share_obs, sub_agent_reward, sub_agent_done, sub_agent_info, sub_avail_action]

