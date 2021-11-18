from gibson2.reward_functions.reward_function_base import BaseRewardFunction
from gibson2.utils.utils import l2_distance
import pybullet as p
import numpy as np

class DoorAngleGoalReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(DoorAngleGoalReward, self).__init__(config)
        self.success_reward = self.config.get(
            'success_reward', 10.0
        )
        self.angle_tol = self.config.get('angle_tol', 10)

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        success = p.getJointState(env.door.body_id, env.door_axis_link_id)[0] > np.pi/2
        reward = self.success_reward if success else 0.0
        return reward
