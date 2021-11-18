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

import gibson2
from gibson2.envs.door_opening_env import doorOpeningEnv
from gibson2.envs.parallel_env import ParallelNavEnv


def reset_seed(success_seed):
    # return values: new_seed, current_seed, success_seed_lst
    if success_seed:
        current_seed = success_seed.pop()
        return current_seed, success_seed
    else:
        current_seed = random.randint(0, int(1e5))
        return current_seed, success_seed


def collect(envs,
            ckpt_idx,
            actor_critic,
            action_mask_choices,
            hidden_size,
            num_eval_episodes,
            device,
            fault_id=-1,
            intermediate_fault_inj=True,
            with_rgb=False):
    # seed management
    current_seed = 0
    random.seed(current_seed)
    np.random.seed(current_seed)
    observations = envs.reset()
    batch = batch_obs(observations)
    fdd_obs = batch["fdd_obs"].squeeze().tolist()
    sas_fdd = []
    sas_fdd_all_episodes = []
    # number of each type of episodes to remember
    num_ff = 50
    num_fs = 50
    num_ss = 100
    num_ff_count = 0
    num_fs_count = 0
    num_ss_count = 0

    # ep_step_count = 0
    # parameters and variables for intermediate injection of failures
    ep_step = 0
    fault_step = -1
    has_fault = True
    if fault_id != -1 and intermediate_fault_inj:
        fault_step = min(max(int(np.random.normal(20, 20)), 0), 50)
        has_fault = False

    repeat_seed = False
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
    # one constant action fault injection mask for each episode
    update_mask = True
    # init non responsive masks and max troque mask
    action_dim = envs.action_space.shape[0]
    action_max_idx = np.ones(action_dim)
    while np.sum(action_max_idx) == action_dim - 3:
        action_max_idx = np.random.randint(2, size=action_dim - 3)
    action_mask = np.ones(action_dim)
    while np.sum(action_mask) == action_dim:
        # Randomly disable some of the joints.
        action_mask = np.random.randint(2, size=action_dim)
        action_mask[-1] = 1
        action_mask[0:2] = 1
    action_masks = torch.from_numpy(action_mask)
    # termination condition
    term_cond = False

    while term_cond == False:
        with torch.no_grad():
            _, actions, log_probs, recurrent_hidden_states = actor_critic.act(
                batch,
                recurrent_hidden_states,
                masks,
                deterministic=False,
                update=0,
            )

        # print("action_logprobs: ",actions, math.exp(log_probs.item()))

        if fault_id == -1 or repeat_seed == False:  # no fault injection
            action_masks = action_mask_choices.index_select(0, torch.tensor([x['task_obs'][3] for x in observations],
                                                                            device=device).to(torch.int))
            actions_masked = actions * action_masks
        elif fault_id != -1 and ep_step<fault_step:
            action_masks = action_mask_choices.index_select(0, torch.tensor([x['task_obs'][3] for x in observations],
                                                                            device=device).to(torch.int))
            actions_masked = actions * action_masks
            has_fault=False
        elif fault_id == 1:
            action_dim = envs.action_space.shape[0]
            if update_mask:
                action_mask = np.ones(action_dim)
                # at least one fault
                while np.sum(action_mask) == action_dim or np.sum(action_mask[2:(action_dim - 1)]) == 0:
                    # Randomly disable some of the joints.
                    action_mask = np.random.randint(2, size=action_dim)
                    action_mask[-1] = 1
                    action_mask[0:2] = 1
                action_masks = torch.from_numpy(action_mask)

            action_masks = action_masks.to(device)
            # print(action_masks)
            update_mask = False
            actions_masked = actions * action_masks
            has_fault=True
        elif fault_id == 2:
            # random troque fault
            action_dim = envs.action_space.shape[0]
            if update_mask:
                action_max_idx = np.ones(action_dim - 2)
                action_max_idx_values = np.zeros(action_dim)
                # generate problematic motors list
                while np.sum(action_max_idx) == action_dim - 2 or np.sum(action_max_idx) == 0:
                    action_max_idx = np.random.randint(2, size=action_dim - 2)
                for idx, res in enumerate(action_max_idx):
                    if res == 1:
                        action_max_idx_values[idx + 2] = np.random.rand() * 10
            actions_masked = actions
            for idx, res in enumerate(actions_masked[0]):
                if res > 0 and action_max_idx_values[idx] != 0:  # motor active
                    actions_masked[0][idx] = np.random.rand() * 10
            update_mask = False
            has_fault=True

        actions_np = actions_masked.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)
        next_fdd_obs = batch["fdd_obs"].squeeze().tolist()

        if with_rgb:
            sas_fdd.append((fdd_obs[:6],
                            actions.squeeze().detach().cpu().tolist(),
                            math.exp(log_probs),
                            next_fdd_obs[:6],
                            fdd_obs[6:],
                            next_fdd_obs[6:],
                            batch["rgb"].squeeze().tolist(),
                            has_fault))
        else:
            sas_fdd.append((fdd_obs[:6],
                            actions.squeeze().detach().cpu().tolist(),
                            math.exp(log_probs),
                            next_fdd_obs[:6],
                            fdd_obs[6:],
                            next_fdd_obs[6:],
                            has_fault))

        # print(sas_fdd)
        fdd_obs = next_fdd_obs
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
        ep_step+=1

        # if termination of episode:
        if (1 - masks).sum().item() == 1:
            term_cond = True
            ep_step=0

        if (1 - masks).sum().item() == 1 and repeat_seed == False:
            if success_masks.sum().item() == 1:
                repeat_seed = True
                print("successful episode, keep seed ", current_seed)
            if num_ss_count < num_ss:
                if len(sas_fdd) < 150 and success_masks.sum().item() == 1:
                    sas_fdd_all_episodes.append((sas_fdd, 1))
                    num_ss_count += 1
                    print("successful episode, append to record: ", len(sas_fdd), len(sas_fdd_all_episodes),
                          num_ss_count)
                term_cond = False
            sas_fdd = []
        elif (
                1 - masks).sum().item() == 1 and success_masks.sum().item() != 1 and fault_id != -1 and repeat_seed == True:
            update_mask = True
            repeat_seed = False
            if num_ff_count < num_ff:
                sas_fdd_all_episodes.append((sas_fdd, 0))
                print("failure episode, append to record: ", len(sas_fdd), len(sas_fdd_all_episodes), num_ff_count)
                num_ff_count += 1
                term_cond = False
            sas_fdd = []
        elif (
                1 - masks).sum().item() == 1 and success_masks.sum().item() == 1 and fault_id != -1 and repeat_seed == True:
            repeat_seed = False
            update_mask = True
            if num_fs_count < num_fs:
                sas_fdd_all_episodes.append((sas_fdd, 2))
                print("success episode with failure, append to record: ", len(sas_fdd), len(sas_fdd_all_episodes),
                      num_fs_count)
                num_fs_count += 1
                term_cond = False
            sas_fdd = []
        elif (1 - masks).sum().item() == 1 and repeat_seed == True:
            repeat_seed = False
            sas_fdd = []
        # RENEW SEED
        if (1 - masks).sum().item() == 1:
            if repeat_seed == False:
                current_seed = random.randint(0, int(1e5))
            if num_fs_count < num_fs or num_ff_count < num_ff or num_ss_count < num_ss:
                term_cond = False
            print('next seed: ', current_seed)
            random.seed(current_seed)
            np.random.seed(current_seed)
            observations = envs.reset()
            if fault_id != -1 and intermediate_fault_inj:
                fault_step =  min(max(int(np.random.normal(20, 20)), 0), 50)
                print("renew fault injection step: ", fault_step)
                has_fault = False

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (episode_success_rates.sum() / episode_counts.sum()).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()
    episode_collision_step_mean = (episode_collision_steps.sum() / episode_counts.sum()).item()
    return sas_fdd_all_episodes


def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    # manual input of arguments tobe removed finally.
    args.eval_only = True
    args.experiment_folder = '/media/zhq/A4AB-0E37/igibson_FDD_exp_lowdim_onestage'
    args.num_eval_processes = 1
    args.env_mode = "gui"
    args.checkpoint_index = 16340
    fault_id_lst = [1, 2]
    config_file = os.path.join(gibson2.example_config_path, 'door_opening_stadium.yaml')

    # create environments
    def load_env(env_mode, device_idx):
        return doorOpeningEnv(config_file=config_file,
                              device_idx=device_idx,
                              automatic_reset=False,
                              mode=env_mode)

    sim_gpu_id = [0]
    env_id_to_which_gpu = np.linspace(0,
                                      len(sim_gpu_id),
                                      num=args.num_train_processes + args.num_eval_processes,
                                      dtype=np.int,
                                      endpoint=False)

    eval_envs = [lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env("gui", device_idx)
                 for env_id in range(args.num_eval_processes, args.num_eval_processes + args.num_eval_processes - 1)]
    eval_envs += [lambda: load_env(args.env_mode, sim_gpu_id[env_id_to_which_gpu[-1]])]
    eval_envs = ParallelNavEnv(eval_envs, blocking=False)

    ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, args.checkpoint_index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    # creat A2C
    # (output_channel, kernel_size, stride, padding)
    cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]

    actor_critic = Policy(
        observation_space=eval_envs.observation_space,
        action_space=eval_envs.action_space,
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

    ckpt = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(ckpt["state_dict"])

    # action masks:
    action_dim = eval_envs.action_space.shape[0]

    action_mask_choices = torch.zeros(2, action_dim, device=device)
    action_mask_choices[0, :] = 1.0
    action_mask_choices[1, :] = 1.0
    # init storage
    rollouts = RolloutStorage(
        args.num_steps,
        eval_envs._num_envs,
        eval_envs.observation_space,
        eval_envs.action_space,
        args.hidden_size,
    )
    # start data collection
    for fault_id in fault_id_lst:
        collected_fdd_list = collect(eval_envs,
                                     args.checkpoint_index,
                                     actor_critic,
                                     action_mask_choices,
                                     args.hidden_size,
                                     args.num_eval_episodes,
                                     device,
                                     fault_id=fault_id,
                                     with_rgb=False)

        with open(os.path.join(args.experiment_folder, 'bfdd_sas_without_rgb_intm_inj_' + str(fault_id) + '.pkl'),
                  'wb') as handle:
            pickle.dump(collected_fdd_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
