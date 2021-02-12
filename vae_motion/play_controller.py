import argparse
import multiprocessing as mp
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import numpy as np
import torch

from common.misc_utils import EpisodeRunner, POSE_CSV_HEADER

FOOT2METER = 0.3048
FOOT2CM = FOOT2METER * 100

env_module = "environments"


def get_model_paths(args):
    pwd = os.getcwd()
    controller_path = None
    pose_vae_path = None

    if args.dir is not None:
        from glob import glob

        base_dir = os.path.join(pwd, args.dir)
        candidate_controller_paths = glob(base_dir + "/con*.pt")
        candidate_pose_vae_paths = glob(base_dir + "/posevae*.pt")

        if len(candidate_controller_paths) == 0 or len(candidate_pose_vae_paths) == 0:
            print("Controller or VAE file not found in ", base_dir)
            exit(0)

        controller_path = candidate_controller_paths[0]
        for path in candidate_controller_paths:
            if "con_" + args.env in path:
                controller_path = path

        pose_vae_path = candidate_pose_vae_paths[0]

    if args.con is not None:
        controller_path = os.path.join(pwd, args.con)

    if args.vae is not None:
        pose_vae_path = os.path.join(pwd, args.vae)

    return controller_path, pose_vae_path


def visualize_rl_controller_replay(args):
    device = "cpu"

    controller_path, pose_vae_path = get_model_paths(args)

    actor_critic = torch.load(controller_path, map_location=device)
    if hasattr(actor_critic, "env_info"):
        frame_skip = actor_critic.env_info["frame_skip"]
    else:
        frame_skip = 1
    controller = actor_critic.actor

    env = gym.make(
        "{}:{}".format(env_module, args.env),
        num_parallel=args.num,
        device=device,
        pose_vae_path=pose_vae_path,
        rendered=True,
        use_params=args.gui,
        camera_tracking=args.track,
        frame_skip=frame_skip,
    )
    print("Loaded:", controller_path)

    obs = env.reset()
    ep_reward = 0

    # overwrite if necessary
    if args.frame != -1:
        env.reset_initial_frames(args.frame)

    with EpisodeRunner(env, save=args.save, max_steps=args.len, csv=args.csv) as runner:

        while not runner.done:
            with torch.no_grad():
                action = controller(obs)

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if done.any():
                print("--- Episode reward: %2.4f" % float(ep_reward[done].mean()))
                ep_reward *= (~done).float()
                reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                obs = env.reset(reset_indices)

            if info.get("reset"):
                print("--- Episode reward: %2.4f" % float(ep_reward.mean()))
                ep_reward = 0
                obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="TimedTargetEnv-v0",
        required=False,
        help="Envrionment name",
    )
    parser.add_argument(
        "--con",
        type=str,
        default=None,
        required=False,
        help="Path to trained RL controller network file",
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        required=False,
        help="Path to VAE associated with the environment",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing both VAE and controller files",
    )
    parser.add_argument(
        "--num", type=int, default=1, help="Number of characters to simulate"
    )
    parser.add_argument("--gui", type=int, default=1, help="Show parameters in GUI")
    parser.add_argument(
        "--track",
        type=int,
        default=1,
        help="1 - camera tracks character | 0 - static camera",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Initial frame for random walk (-1 for random)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save video recorded from camera",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        required=False,
        help="CSV path to dump trajectory",
    )
    parser.add_argument(
        "--len",
        type=int,
        default=None,
        help="Length of video to save in number of frames",
    )
    args = parser.parse_args()

    ps = [
        mp.Process(target=visualize_rl_controller_replay, args=(args,)),
    ]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
