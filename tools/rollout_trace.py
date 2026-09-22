"""Record a deterministic rollout as a pose trajectory, and optionally frames.

Used to prove a change leaves model behaviour untouched: run before and
after, then compare the saved arrays for exact equality.

    python tools/rollout_trace.py --env TargetEnv-v0 \
        --controller vae_motion/models/con_TargetEnv-v0.safetensors \
        --out trace.npy --shots shots/
"""
import argparse
import os
import sys

import numpy as np
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def build_env(env_id, models_dir, rendered, frame_skip):
    import gym

    import environments  # noqa: F401  (registers the env ids)

    return gym.make(
        env_id,
        num_parallel=1,
        device="cpu",
        pose_vae_path=models_dir,
        rendered=rendered,
        use_params=False,
        camera_tracking=False,
        frame_skip=frame_skip,
    )


def capture(env, shots_dir, name):
    import pybullet as pb
    from imageio import imwrite

    p = env.viewer._p
    lookat = list(np.asarray(env.viewer.root_xyzs[0], dtype=float))
    view = p.computeViewMatrixFromYawPitchRoll(lookat, 4.5, 60, -20, 0, upAxisIndex=2)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=4 / 3, nearVal=0.01, farVal=1000)
    w, h, rgb, _, _ = p.getCameraImage(
        640, 480, viewMatrix=view, projectionMatrix=proj, renderer=pb.ER_TINY_RENDERER
    )
    image = np.reshape(np.array(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]
    os.makedirs(shots_dir, exist_ok=True)
    imwrite(os.path.join(shots_dir, name + ".png"), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", required=True, help="Registered env id, e.g. TargetEnv-v0"
    )
    parser.add_argument(
        "--controller", required=True, help="Path to a .safetensors controller"
    )
    parser.add_argument(
        "--models", default=None, help="Directory holding the VAE checkpoint"
    )
    parser.add_argument("--out", required=True, help="Where to write the .npy trajectory")
    parser.add_argument("--shots", default=None, help="Directory for PNG frames")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=1)
    args = parser.parse_args()

    if args.shots:
        # Render offscreen so this works without a display.
        import pybullet as pb

        import environments.mocap_renderer as mocap_renderer

        mocap_renderer.pb.GUI = pb.DIRECT
        import common.bullet_utils as bullet_utils

        bullet_utils.Camera.wait = lambda self: None

    from vae_motion.models import load_model

    models_dir = args.models or os.path.dirname(os.path.abspath(args.controller))
    env = build_env(args.env, models_dir, args.shots is not None, args.frame_skip)
    controller = load_model(args.controller, "cpu").actor

    # Seed last, after every model is built. load_model constructs the module
    # before overwriting its tensors, so it draws from the global RNG; seeding
    # here keeps the rollout independent of how much init consumed.
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    obs = env.reset()
    poses = []
    for step in range(args.steps):
        with torch.no_grad():
            action = controller(obs)
        obs, _reward, _done, _info = env.step(action)
        poses.append(env.history[:, 0].clone().cpu().numpy()[0])
        if args.shots and step % 20 == 19:
            capture(env, args.shots, "%s_%04d" % (args.env, step))

    trace = np.asarray(poses, dtype=np.float32)
    np.save(args.out, trace)
    print(
        "%s shape=%s checksum=%.6f -> %s"
        % (args.env, trace.shape, float(trace.sum()), args.out)
    )


if __name__ == "__main__":
    main()
