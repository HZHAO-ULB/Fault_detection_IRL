from gibson2.reward_functions.reward_function_base import BaseRewardFunction
import pybullet as p
import numpy as np


class DoorStatusReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(DoorStatusReward, self).__init__(config)
        self.potential_reward_weight = self.config.get(
            'potential_reward_weight', 1.0
        )

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = task.get_door_handler_distance(env)
        self.angular_potential = np.max(p.getJointState(env.door.body_id, env.door_axis_link_id)[0],0)
        self.previous_task_status = env.stage

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = task.get_door_handler_distance(env)
        new_angular_potential = np.max(p.getJointState(env.door.body_id, env.door_axis_link_id)[0],0)

        if env.stage == 0:
            reward = self.potential - new_potential
        else:
            reward = 0
        rewarda = new_angular_potential - self.angular_potential
        reward *= self.potential_reward_weight
        rewarda *= self.potential_reward_weight
        if env.stage == 1 and self.previous_task_status == 0:
            rewardb = 1
        else:
            rewardb = 0
        self.previous_task_status = env.stage

        self.potential = new_potential
        self.angular_potential = new_angular_potential
        return reward+rewardb+rewarda