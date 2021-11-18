from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition


class robotDeath(BaseTerminationCondition):
    """
    MaxCollision used for navigation tasks
    Episode terminates if the robot has collided more than
    max_collisions_allowed times
    """

    def __init__(self, config):
        super(robotDeath, self).__init__(config)
        self.death_z_thresh = self.config.get('death_z_thresh', 0.1)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot has collided more than self.max_collisions_allowed times

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """

        done = env.robots[0].get_position()[2] > self.death_z_thresh
        success = False
        if done:
            print("DEATH")
        return done, success
