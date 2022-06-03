# -*- coding:utf-8 -*-
# @Time : 2022/5/26 19:51
# @Author: zhcode
# @File : utils.py
import math


def get_forward_reward(pre_position, next_position):
    """
    compute reward when  quadruped robot approach to target position.
    Args:
        pre_position: previous position
        next_position: next position

    Returns:
        forward reward
    """
    # modify first value to adjust yaw punishment
    pseude_target = [0.04, 0.]
    pre_position = list(pre_position)[0:-1]
    next_position = list(next_position)[0:-1]
    next_position[0] = next_position[0] - pre_position[0]
    pre_position[0] = 0.

    def compute_dis(target_pos, pos):
        return math.sqrt((target_pos[0] - pos[0])**2 + (target_pos[1] - pos[1])**2)

    return compute_dis(pseude_target, pre_position) - compute_dis(pseude_target, next_position)


if __name__ == '__main__':
    pre_pos = (3.001102012889821, -0.021187675343078548, 0.24205244100237533)
    next_pos = (3.005720900674606, -0.021174663979786932, 0.24240420514164612)
    print(get_forward_reward(pre_pos, next_pos))