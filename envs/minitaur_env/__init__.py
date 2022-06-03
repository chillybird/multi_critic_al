import os
import yaml
import importlib
from envs.minitaur_env import wrappers
from envs.minitaur_env import minitaur


class BulletEnv:
    # 加载指定的bullet环境
    env_pakg = "envs.minitaur_env.minitaur"

    def __init__(self, env_name):
        self.args = {}
        self.config = {}
        self.max_length = None
        self.env_name = env_name
        print(f"build env {env_name}")
        self.env_class_name = "".join(list(map(lambda x: x.capitalize(), env_name.split("_"))))

    def build_env(self, **kwargs):
        env_module = importlib.import_module(name=f'.{self.env_name}', package=self.env_pakg)
        env_class = getattr(env_module, self.env_class_name)
        self.config = self.load_params_from_yaml()
        max_length_config = self.config.pop('max_length', None)
        max_length_kwargs = kwargs.pop('max_length', None)
        self.max_length = max_length_kwargs if max_length_kwargs else max_length_config
        self.random_max_steps = self.config.pop('random_max_steps', None)
        self.random_start = kwargs.pop('random_start', False)
        self.multi_task = kwargs.pop('multi_task', False)
        self.args.update(kwargs)
        self.config.update(kwargs)
        # print("load env params: ", self.config)
        return self._create_environment(env_class(**self.config))

    def load_params_from_yaml(self):
        """
        从yaml配置文件加载环境的参数
        :param env_name_name:
        :return:
        """
        if not os.path.exists(os.path.join(os.path.dirname(__file__), f'{self.env_name}.yaml')):
            raise Exception("env is not exist.")

        env_config = {}
        with open(os.path.join(os.path.dirname(__file__), f'{self.env_name}.yaml'), 'r') as file:
            env_config = yaml.safe_load(file)

        return env_config

    def _create_environment(self, env):
        """Constructor for an instance of the environment.
        Args:
          config: Object providing configurations via attributes.

        Returns:
          Wrapped OpenAI Gym environment.
        """
        # 从上到下是从环境到训练的网络的方向
        if self.max_length is not None:
            env = wrappers.LimitDuration(env, self.max_length)
        env = wrappers.RangeNormalize(env)
        env = wrappers.ClipAction(env)
        env = wrappers.ConvertTo32Bit(env)
        return env


class BulletMultiEnv(BulletEnv):

    def _create_environment(self, env):
        """Constructor for an instance of the environment.
        Args:
          config: Object providing configurations via attributes.

        Returns:
          Wrapped OpenAI Gym environment.
        """
        if self.max_length is not None:
            print("Limit duration.")
            env = wrappers.LimitDuration(env, self.max_length)

        if self.random_max_steps is not None and self.random_start:
            print(f"Random start with random start step {self.random_max_steps}.")
            env = wrappers.RandomStart(env, self.random_max_steps)
        # auto reset done env
        env = wrappers.AutoReset(env)
        env = wrappers.RangeNormalize(env)
        env = wrappers.ClipAction(env)
        # 将环境封装为多智能体环境
        env = wrappers.ConvertToMultiEnv(env, **{
            'env_name': self.env_name,
            'multi_task': self.multi_task,
            'use_signal_in_observation': self.args['use_signal_in_observation'],
            'use_angle_in_observation': self.args['use_angle_in_observation']
        })
        env = wrappers.ConvertTo32Bit(env)
        return env


