import copy
import os
import time
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import torch

from algorithms.ppo import PPO
from algorithms.storage import RolloutStorage
from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule, update_exponential_schedule
from vae_motion.models import PoseVAEController, PoseVAEPolicy


def make_gym_environment(args):

    pose_vae_path = os.path.join(current_dir, args.vae_path)

    env = gym.make(
        "{}:{}".format(args.env_module, args.env_name),
        num_parallel=args.num_parallel,
        device=args.device,
        pose_vae_path=pose_vae_path,
        rendered=False,
        frame_skip=args.frame_skip,
    )
    env.seed(args.seed)

    return env


class StatsLogger:
    def __init__(self, csv_path):
        self.start = time.time()
        self.logger = CSVLogger(log_path=csv_path)

    def preprocess(self, args, data):
        rewards = data["ep_info"].get("reward")
        if rewards is not None:
            data["mean_rew"] = float(rewards.mean())
            data["median_rew"] = float(rewards.median())
            data["min_rew"] = float(rewards.min())
            data["max_rew"] = float(rewards.max())
            del data["ep_info"]

        update = data["update"]
        total_num_steps = (update + 1) * args.num_parallel * args.num_steps_per_rollout
        data["num_steps"] = total_num_steps

    def log_stats(self, args, data):
        now = time.time()
        self.preprocess(args, data)
        self.logger.log_epoch(data)

        print(
            (
                "Updates {}, num timesteps {}, FPS {}, "
                "mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}, "
                "entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}"
            ).format(
                data["update"],
                data["num_steps"],
                int(data["num_steps"] / (now - self.start)),
                data["mean_rew"],
                data["median_rew"],
                data["min_rew"],
                data["max_rew"],
                data["dist_entropy"],
                data["value_loss"],
                data["action_loss"],
            ),
            flush=True,
        )


def main():
    # setup parameters
    args = SimpleNamespace(
        env_module="environments",
        env_name="TargetEnv-v0",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_parallel=100,
        vae_path="models/",
        frame_skip=1,
        seed=16,
        load_saved_model=False,
    )

    args.num_parallel *= args.frame_skip
    env = make_gym_environment(args)

    # env parameters
    args.action_size = env.action_space.shape[0]
    args.observation_size = env.observation_space.shape[0]

    # other configs
    args.save_path = os.path.join(current_dir, "con_" + args.env_name + ".pt")

    # sampling parameters
    args.num_frames = 10e7
    args.num_steps_per_rollout = env.unwrapped.max_timestep
    args.num_updates = int(
        args.num_frames / args.num_parallel / args.num_steps_per_rollout
    )

    # learning parameters
    args.lr = 3e-5
    args.final_lr = 1e-5
    args.eps = 1e-5
    args.lr_decay_type = "exponential"
    args.mini_batch_size = 1000
    args.num_mini_batch = (
        args.num_parallel * args.num_steps_per_rollout // args.mini_batch_size
    )

    # ppo parameters
    use_gae = True
    entropy_coef = 0.0
    value_loss_coef = 1.0
    ppo_epoch = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    max_grad_norm = 1.0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])

    if args.load_saved_model:
        actor_critic = torch.load(args.save_path, map_location=args.device)
        print("Loading model:", args.save_path)
    else:
        controller = PoseVAEController(env)
        actor_critic = PoseVAEPolicy(controller)

    actor_critic = actor_critic.to(args.device)
    actor_critic.env_info = {"frame_skip": args.frame_skip}

    agent = PPO(
        actor_critic,
        clip_param,
        ppo_epoch,
        args.num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=max_grad_norm,
    )

    rollouts = RolloutStorage(
        args.num_steps_per_rollout,
        args.num_parallel,
        obs_shape,
        args.action_size,
        actor_critic.state_size,
    )
    obs = env.reset()
    rollouts.observations[0].copy_(obs)
    rollouts.to(args.device)

    log_path = os.path.join(current_dir, "log_ppo_progress-{}".format(args.env_name))
    logger = StatsLogger(csv_path=log_path)

    for update in range(args.num_updates):

        ep_info = {"reward": []}
        ep_reward = 0

        if args.lr_decay_type == "linear":
            update_linear_schedule(
                agent.optimizer, update, args.num_updates, args.lr, args.final_lr
            )
        elif args.lr_decay_type == "exponential":
            update_exponential_schedule(
                agent.optimizer, update, 0.99, args.lr, args.final_lr
            )

        for step in range(args.num_steps_per_rollout):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.act(
                    rollouts.observations[step]
                )

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            end_of_rollout = info.get("reset")
            masks = (~done).float()
            bad_masks = (~(done * end_of_rollout)).float()

            if done.any():
                ep_info["reward"].append(ep_reward[done].clone())
                ep_reward *= (~done).float()  # zero out the dones
                reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                obs = env.reset(reset_indices)

            if end_of_rollout:
                obs = env.reset()

            rollouts.insert(
                obs, action, action_log_prob, value, reward, masks, bad_masks
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        torch.save(copy.deepcopy(actor_critic).cpu(), args.save_path)

        ep_info["reward"] = torch.cat(ep_info["reward"])
        logger.log_stats(
            args,
            {
                "update": update,
                "ep_info": ep_info,
                "dist_entropy": dist_entropy,
                "value_loss": value_loss,
                "action_loss": action_loss,
            },
        )


if __name__ == "__main__":
    main()
