from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
import pybullet as p
import numpy as np

class DoorAngle(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(DoorAngle, self).__init__(config)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = p.getJointState(env.door.body_id, env.door_axis_link_id)[0] > np.pi/2
        success = done
        if done:
            print("SUCCESS")
        return done, success
