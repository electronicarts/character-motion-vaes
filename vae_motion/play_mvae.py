import argparse
import multiprocessing as mp
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import torch

from common.misc_utils import EpisodeRunner

FOOT2METER = 0.3048
FOOT2CM = FOOT2METER * 100

env_module = "environments"


def test_vae_replay_full_motion(args):
    device = "cpu"

    num_characters = args.num
    pose_vae_path = os.path.join(os.getcwd(), args.vae)

    is_rendered = True
    env = gym.make(
        "{}:{}".format(env_module, args.env),
        num_parallel=num_characters,
        device=device,
        pose_vae_path=pose_vae_path,
        rendered=is_rendered,
        use_params=args.gui,
        camera_tracking=args.track,
        frame_skip=args.skip,
    )

    env.reset()
    mocap_data = env.mocap_data
    num_future_predictions = env.pose_vae_model.num_future_predictions

    latent_size = env.action_space.shape[0]
    action_shape = (num_characters, latent_size)
    action = torch.empty(action_shape).to(device)

    # overwrite if necessary
    if args.frame != -1:
        env.reset_initial_frames(args.frame)

    alpha = torch.ones(num_characters).float()
    with EpisodeRunner(env, save=args.save, max_steps=args.len, csv=args.csv) as runner:

        while not runner.done:

            action.normal_(0, 1.0)
            frames = env.get_vae_next_frame(action)

            for i in range(env.frame_skip):
                frame_indices = (
                    env.start_indices
                    + env.pose_vae_model.num_condition_frames
                    + env.timestep
                    + env.substep * env.frame_skip
                )

                # testing only, need to overwrite history in env
                alpha_ = env.viewer.controller_autonomy if is_rendered else 1.0
                alpha.fill_(alpha_)
                prediction_range = (
                    frame_indices.repeat((num_future_predictions, 1)).t()
                    + torch.arange(0, num_future_predictions).long()
                ).remainder_(mocap_data.shape[0])
                if args.mocap:
                    alpha[0] = 0
                frames = (
                    alpha.view(-1, 1, 1) * frames
                    + alpha.mul(-1).add(1).view(-1, 1, 1) * mocap_data[prediction_range]
                )

                frame = frames[:, i]
                _, _, done, info = env.calc_env_state(frame)

                if done.any():
                    reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                    env.reset(reset_indices)

                if info.get("reset"):
                    env.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="RandomWalkEnv-v0",
        required=False,
        help="Envrionment name",
    )
    parser.add_argument("--vae", type=str, help="Path to VAE file", required=True)
    parser.add_argument(
        "--num", type=int, default=1, help="Number of characters to simulate"
    )
    parser.add_argument("--gui", type=int, default=1, help="Show parameters in GUI")
    parser.add_argument(
        "--mocap",
        action="store_true",
        default=False,
        help="Play first character with pure mocap",
    )
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
        "--skip",
        type=int,
        default=1,
        help="Number of internal steps (minus 1) per action",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save video recorded from camera",
    )
    parser.add_argument(
        "--len",
        type=int,
        default=None,
        help="Length of video to save in number of frames",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        required=False,
        help="CSV path to dump trajectory",
    )
    args = parser.parse_args()

    ps = [
        mp.Process(target=test_vae_replay_full_motion, args=(args,)),
    ]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
