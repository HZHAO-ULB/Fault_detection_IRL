import os
from time import time
from collections import deque
import random
import numpy as np
import sys
import argparse
import math
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.envs.toy_env.toy_env import ToyEnv
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *
import matplotlib.pyplot as plt

def main():
    #load parameters
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    #manual input of arguments tobe removed finally.
    args.eval_only=True
    args.experiment_folder = '/media/zhq/A4AB-0E37/igibson_FDD_exp'
    #with open(os.path.join(args.experiment_folder, 'fdd_sas_with_rgb.pkl'), 'rb') as handle:
    #    collected_fdd_list = pickle.load(handle)
    with open(os.path.join(args.experiment_folder, 'res_fdd_sas_with_rgb.pkl'), 'rb') as handle:
        segmentation_result = pickle.load(handle)
    #print(collected_fdd_list[-1][0][-1])
    print(segmentation_result[-1])
    import matplotlib.pyplot as plt
    '''
    w = 50
    h = 50
    fig = plt.figure(figsize=(8, 8))
    columns = 11
    rows = 1
    for i in range(1, columns * rows + 1):
        img = collected_fdd_list[-1][(i-1)*5][-1]
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(str((i-1)*5+1))
        plt.axis('off')
        plt.imshow(img)
    '''
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('Time step', fontsize=16)
    ax.set_ylabel('Stage (Forward passing)', fontsize=16)
    ax.plot(segmentation_result[-1])

    plt.show()
if __name__ == "__main__":
    main()