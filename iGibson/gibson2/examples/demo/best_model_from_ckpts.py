import os
from time import time
from collections import deque
import random
import numpy as np
import sys
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.envs.toy_env.toy_env import ToyEnv
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

import gibson2
from gibson2.envs.door_opening_env import doorOpeningEnv
from gibson2.envs.parallel_env import ParallelNavEnv

def evaluate(envs,
            ckpt_idx,
             actor_critic,
             action_mask_choices,
             hidden_size,
             num_eval_episodes,
             device,
             writer,
             update=0,
             count_steps=0,
             eval_only=False):
    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(envs._num_envs, 1, device=device)
    episode_collision_steps = torch.zeros(envs._num_envs, 1, device=device)

    episode_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs._num_envs, 1, device=device)

    recurrent_hidden_states = torch.zeros(envs._num_envs, hidden_size, device=device)
    masks = torch.zeros(envs._num_envs, 1, device=device)

    while episode_counts.sum() < num_eval_episodes:
        with torch.no_grad():
            _, actions, _, recurrent_hidden_states = actor_critic.act(
                batch,
                recurrent_hidden_states,
                masks,
                deterministic=True,
                update=0,
            )
        action_masks = action_mask_choices.index_select(0, torch.tensor([x['task_obs'][3] for x in observations],
                                                                        device=device).to(torch.int))
        actions_masked = actions * action_masks
        actions_np = actions_masked.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )
        success_masks = torch.tensor(
            [[1.0] if done and "success" in info and info["success"] else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        lengths = torch.tensor(
            [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        collision_steps = torch.tensor(
            [[float(info["collision_step"])] if done and "collision_step" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_success_rates += success_masks
        episode_lengths += lengths
        episode_collision_steps += collision_steps
        episode_counts += 1 - masks
        current_episode_reward *= masks

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (episode_success_rates.sum() / episode_counts.sum()).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()
    episode_collision_step_mean = (episode_collision_steps.sum() / episode_counts.sum()).item()

    if eval_only:
        print("EVAL: checkpoint: {}\tnum_eval_episodes: {}\treward: {:.3f}\t"
              "success_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}\t".format(ckpt_idx,
            num_eval_episodes, episode_reward_mean, episode_success_rate_mean, episode_length_mean,
            episode_collision_step_mean
        ))

    return episode_reward_mean
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
    args.num_train_processes = 18
    args.num_eval_processes = 1
    args.env_mode = "gui"
    current_reward = -1
    best_reward = 0
    args.checkpoint_index=[12590, 12600, 12610]
    ckptpath=[]
    for ckpt_idx in args.checkpoint_index:
        ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, ckpt_idx)
        ckptpath.append(ckpt_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    logger.add_filehandler(log_file)

    if not args.eval_only:
        writer = SummaryWriter(log_dir=summary_folder)
    else:
        writer = None

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    config_file = os.path.join(gibson2.example_config_path, 'door_opening_stadium.yaml')

    assert os.path.isfile(config_file), "config file does not exist: {}".format(config_file)

    for (k, v) in parse_config(config_file).items():
        logger.info("{}: {}".format(k, v))

    #env = doorOpeningEnv(config_file=config_filename, mode='gui')
    def load_env(env_mode, device_idx):
        return doorOpeningEnv(config_file=config_file,
                              device_idx=device_idx,
                              automatic_reset=True,
                                          mode=env_mode)

    sim_gpu_id = [0]
    env_id_to_which_gpu = np.linspace(0,
                                      len(sim_gpu_id),
                                      num=args.num_train_processes + args.num_eval_processes,
                                      dtype=np.int,
                                      endpoint=False)
    train_envs = [lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env("headless", device_idx)
                  for env_id in range(args.num_train_processes)]
    train_envs = ParallelNavEnv(train_envs, blocking=False)
    #train and eval in the same env

    # (output_channel, kernel_size, stride, padding)
    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]

    actor_critic = Policy(
        observation_space=train_envs.observation_space,
        action_space=train_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        use_clipped_value_loss=True
    )

    for idx, ckpt_pth in enumerate(ckptpath):
        ckpt = torch.load(ckpt_pth, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoing: {}".format(ckpt_pth))

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in agent.parameters())
            )
        )

        action_dim = train_envs.action_space.shape[0]

        action_mask_choices = torch.zeros(2, action_dim, device=device)
        action_mask_choices[0, 0:2] = 1.0
        action_mask_choices[1, :] = 1.0

        if args.eval_only:
            evaluate(train_envs,
                     args.checkpoint_index[idx],
                     actor_critic,
                     action_mask_choices,
                     args.hidden_size,
                     args.num_eval_episodes,
                     device,
                     writer,
                     update=0,
                     count_steps=0,
                     eval_only=True)


if __name__ == "__main__":
    main()