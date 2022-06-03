# -*- coding:utf-8 -*-
# @Time : 2022/3/25 18:04
# @Author: zhcode
# @File : minitaur_pacing_env.py
import math
import numpy as np
from envs.minitaur_env.minitaur.minitaur_gait_env import MinitaurGaitEnv

# TODO(tingnan): These constants should be moved to minitaur/minitaur_gym_env.
NUM_LEGS = 4
NUM_MOTORS = 2 * NUM_LEGS


class MinitaurPacingEnv(MinitaurGaitEnv):
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

        pacing_signal = np.array([
            sw_first_pair, sw_first_pair, sw_second_pair, sw_second_pair,
            ext_first_pair, ext_first_pair, ext_second_pair, ext_second_pair
        ])
        return np.array(self._init_pose) + pacing_signal
