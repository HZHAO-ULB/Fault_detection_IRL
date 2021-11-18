from gibson2.envs.door_opening_env import doorOpeningEnv
from time import time
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging
import pybullet as p
from gibson2.objects.object_base import Object
import numpy as np
from gibson2.utils.utils import parse_config, rotate_vector_3d, l2_distance, quatToXYZW
from transforms3d.euler import euler2quat
import cv2

def main():
    config_filename = os.path.join(gibson2.example_config_path, 'door_opening_stadium.yaml')
    env = doorOpeningEnv(config_file=config_filename, mode='gui')

    print(env.robots[0])

    for j in range(10):
        env.reset()
        for i in range(100000000):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                #print('sampled action: ', action)
                state, reward, done, info = env.step(0.0)
                print('enmv observation', env.observation_space)
                frame = cv2.cvtColor(
                    state["exo_rgb"], cv2.COLOR_RGB2BGR)
                cv2.imshow('1',frame)
    env.close()


if __name__ == "__main__":
    main()
