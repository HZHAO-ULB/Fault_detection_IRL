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
    not_failed = True
    while episode_counts.sum() < num_eval_episodes and not_failed:
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
        recurrent_hidden_states = masks.to(device) * recurrent_hidden_states
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
        if (1-masks).sum().item() == 1 and success_masks.sum().item() == 0:
            not_failed = False

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (episode_success_rates.sum() / episode_counts.sum()).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()
    episode_collision_step_mean = (episode_collision_steps.sum() / episode_counts.sum()).item()

    if eval_only:
        print("EVAL: num_eval_episodes: {}\treward: {:.3f}\t"
              "success_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}\t"
              "total_energy_cost: {:.3f}\tavg_energy_cost: {:.3f}\t"
              "stage_open_door: {:.3f}\tstage_to_target: {:.3f}".format(
            num_eval_episodes, episode_reward_mean, episode_success_rate_mean, episode_length_mean,
            episode_collision_step_mean
        ))
    else:
        logger.info("EVAL: num_eval_episodes: {}\tupdate: {}\t"
                    "reward: {:.3f}\tsuccess_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}".format(
            num_eval_episodes, update, episode_reward_mean, episode_success_rate_mean, episode_length_mean,
            episode_collision_step_mean))
        writer.add_scalar("eval/updates/reward", episode_reward_mean, global_step=update)
        writer.add_scalar("eval/updates/success_rate", episode_success_rate_mean, global_step=update)
        writer.add_scalar("eval/updates/episode_length", episode_length_mean, global_step=update)
        writer.add_scalar("eval/updates/collision_step", episode_collision_step_mean, global_step=update)

        writer.add_scalar("eval/env_steps/reward", episode_reward_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/success_rate", episode_success_rate_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/episode_length", episode_length_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/collision_step", episode_collision_step_mean, global_step=count_steps)
    return episode_reward_mean
def main():
    #load parameters
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    #manual input of arguments tobe removed finally.
    args.eval_only=False
    args.experiment_folder = '/media/zhq/A4AB-0E37/igibson_FDD_exp_lowdim_onestage_retrain'
    args.num_train_processes = 15
    args.num_eval_processes = 1
    args.env_mode = "gui"
    current_reward = -1
    best_reward = 0

    ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, args.checkpoint_index)

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
    if args.env_mode == "gui":
        eval_envs = [lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env("headless", device_idx)
                    for env_id in range(args.num_train_processes, args.num_train_processes + args.num_eval_processes - 1)]
        eval_envs += [lambda: load_env(args.env_mode, sim_gpu_id[env_id_to_which_gpu[-1]])]
        eval_envs = ParallelNavEnv(eval_envs, blocking=False)

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

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoing: {}".format(ckpt_path))

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )

    action_dim = train_envs.action_space.shape[0]

    action_mask_choices = torch.zeros(2, action_dim, device=device)
    action_mask_choices[0, :] = 1.0
    action_mask_choices[1, :] = 1.0

    if args.eval_only:
        evaluate(train_envs,
                 actor_critic,
                 action_mask_choices,
                 args.hidden_size,
                 args.num_eval_episodes,
                 device,
                 writer,
                 update=0,
                 count_steps=0,
                 eval_only=True)
        return

    observations = train_envs.reset()

    batch = batch_obs(observations)

    rollouts = RolloutStorage(
        args.num_steps,
        train_envs._num_envs,
        train_envs.observation_space,
        train_envs.action_space,
        args.hidden_size,
    )
    for sensor in rollouts.observations:
        #print(batch[sensor], sensor)
        rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)

    episode_rewards = torch.zeros(train_envs._num_envs, 1)
    episode_success_rates = torch.zeros(train_envs._num_envs, 1)
    episode_lengths = torch.zeros(train_envs._num_envs, 1)
    episode_collision_steps = torch.zeros(train_envs._num_envs, 1)
    episode_counts = torch.zeros(train_envs._num_envs, 1)
    current_episode_reward = torch.zeros(train_envs._num_envs, 1)

    window_episode_reward = deque()
    window_episode_success_rates = deque()
    window_episode_lengths = deque()
    window_episode_collision_steps = deque()
    window_episode_counts = deque()

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = start_env_step

    for update in range(start_epoch, args.num_updates):
        update_lr(agent.optimizer, args.lr, update, args.num_updates, args.use_linear_lr_decay, 0)

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        # collect num_steps tuples for each environment
        for step in range(args.num_steps):
            t_sample_action = time()
            # sample actions
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }
                # values: [num_processes, 1]
                # actions: [num_processes, 1]
                # actions_log_probs: [num_processes, 1]
                # recurrent_hidden_states: [num_processes, hidden_size]
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                    update=update,
                )
            pth_time += time() - t_sample_action
            t_step_env = time()


            # outputs:
            # [
            #     (observation, reward, done, info),
            #     ...
            #     ...
            #     (observation, reward, done, info),
            # ]
            # len(outputs) == num_processes
            action_masks = action_mask_choices.index_select(0, torch.tensor([x['task_obs'][3] for x in observations],device=device).to(torch.int))

            actions_masked = actions * action_masks
            actions_np = actions_masked.cpu().numpy()
            outputs = train_envs.step(actions_np)
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

            print("this rewards: ", rewards)
            #print("this action: ", actions_np)
            env_time += time() - t_step_env

            t_update_stats = time()
            batch = batch_obs(observations)
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)
            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )

            #only enable success mask, length counter and collision steps
            success_masks = torch.tensor(
                [[1.0] if done and "success" in info and info["success"] else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float
            )
            lengths = torch.tensor(
                [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
            )
            collision_steps = torch.tensor(
                [[float(info["collision_step"])] if done and "collision_step" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
            )

            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_success_rates += success_masks
            episode_lengths += lengths
            episode_collision_steps += collision_steps
            episode_counts += 1 - masks
            current_episode_reward *= masks
            recurrent_hidden_states = masks.to(device) * recurrent_hidden_states

            #print(recurrent_hidden_states)
            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )

            count_steps += train_envs._num_envs
            pth_time += time() - t_update_stats

        if len(window_episode_reward) == args.perf_window_size:
            window_episode_reward.popleft()
            window_episode_success_rates.popleft()
            window_episode_lengths.popleft()
            window_episode_collision_steps.popleft()
            window_episode_counts.popleft()
        window_episode_reward.append(episode_rewards.clone())
        window_episode_success_rates.append(episode_success_rates.clone())
        window_episode_lengths.append(episode_lengths.clone())
        window_episode_collision_steps.append(episode_collision_steps.clone())
        window_episode_counts.append(episode_counts.clone())

        t_update_model = time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        # V(s_t+num_steps) - next_value: [num_processes, 1]
        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts, update=update)
        print("num updates: ", update, value_loss, action_loss, dist_entropy)
        rollouts.after_update()
        pth_time += time() - t_update_model

        if update > 0 and update % args.checkpoint_interval == 0:
            checkpoint = {"state_dict": agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    ckpt_folder,
                    "ckpt.{}.pth".format(update),
                ),
            )
        if update > 500 and update % args.eval_interval == 0 and args.env_mode == "gui":
            current_reward = evaluate(eval_envs,
                     actor_critic,
                     action_mask_choices,
                     args.hidden_size,
                     args.num_eval_episodes,
                     device,
                     writer,
                     update=update,
                     count_steps=count_steps,
                     eval_only=False)
        elif update > 0 and update % int(1000/args.num_steps) == 0 and args.env_mode != "gui":
            current_reward = evaluate(train_envs,
                     actor_critic,
                     action_mask_choices,
                     args.hidden_size,
                     args.num_eval_episodes,
                     device,
                     writer,
                     update=update,
                     count_steps=count_steps,
                     eval_only=False)
        if current_reward>best_reward:
            best_reward = current_reward
            checkpoint = {"state_dict": agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    ckpt_folder,
                    "best_ckpt.{}.pth".format(update),
                ),
            )
if __name__ == "__main__":
    main()