"""Implements the gym environment of minitaur moving with trotting style.
"""
import math

from gym import spaces
import numpy as np
from envs.minitaur_env.minitaur.minitaur_gait_env import MinitaurGaitEnv

from envs.minitaur_env.minitaur import minitaur_gym_env

# TODO(tingnan): These constants should be moved to minitaur/minitaur_gym_env.
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS


class MinitaurTrottingEnv(MinitaurGaitEnv):
        def _signal(self, t):
            """Generates the trotting gait for the robot.

            Args:
              t: Current time in simulation.

            Returns:
              A numpy array of the reference leg positions.
            """
            # Generates the leg trajectories for the two digonal pair of legs.
            ext_first_pair, sw_first_pair = self._gen_signal(t, 0)
            ext_second_pair, sw_second_pair = self._gen_signal(t, math.pi)

            trotting_signal = np.array([
                sw_first_pair, sw_second_pair, sw_second_pair, sw_first_pair, ext_first_pair,
                ext_second_pair, ext_second_pair, ext_first_pair
            ])
            signal = np.array(self._init_pose) + trotting_signal
            return signal
