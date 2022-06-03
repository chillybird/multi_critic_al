from envs.mpe.environment import MultiAgentEnv
from envs.mpe.scenarios import load
import argparse


class Env(object):
    def __init__(self, idx, args):
        # load scenario from script
        scenario = load(args.scenario_name + ".py").Scenario()
        # create world
        world = scenario.make_world(args)
        # create multiagent environment
        self._env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation, scenario.info)
        # print(self._env.__dict__)

    def __getattr__(self, item):
        return getattr(self._env, item)


if __name__ == '__main__':
    args = {
        "use_valuenorm": False,
        "use_popart": False,
        "env_name": "MPE",
        "algorithm_name": "rmappo",
        "experiment_name": "check",
        "scenario_name": "simple_spread",
        "num_agents": 3,
        "num_landmarks": 3,
        "seed": 0,
        "n_training_threads": 1,
        "n_rollout_threads": 128,
        "num_mini_batch": 1,
        "episode_length": 25,
        "num_env_steps": 20000000,
        "ppo_epoch": 10,
        "use_ReLU": False,
        "gain": 0.01,
        "lr": 7e-4,
        "critic_lr": 7e-4,
        "wandb_name": "zoeyuchao",
        "user_name": "zoeyuchao"
    }
    env = Env(0, argparse.Namespace(**args))
    print(env.__dict__)
