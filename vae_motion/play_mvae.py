#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import os
import sys
import gym
import torch
import numpy as np

# --- repo path setup ---
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

from common.misc_utils import EpisodeRunner

FOOT2METER = 0.3048
FOOT2CM = FOOT2METER * 100
env_module = "environments"


# =========================
# 267D -> 630D 어댑터 유틸
# =========================

def compute_stats(npz_path):
    """
    학습에 사용한 mocap.npz에서 X의 (μ,σ)을 계산하여 정규화 역변환에 사용.
    """
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)               # [1, D]
    sd = X.std(axis=0, keepdims=True) + 1e-6         # [1, D]
    return mu, sd

def denorm(feat, mu, sd):
    return feat * sd + mu

def rot6_reorth(rot6_flat):
    """
    rot6 (J*6,) -> (J,3,3), 정규화/재직교화 포함
    """
    J = rot6_flat.shape[-1] // 6
    r = rot6_flat.reshape(J, 6).reshape(J, 3, 2)             # [J,3,2]
    b1 = r[..., 0]
    b1 = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    v2 = r[..., 1] - (b1 * (r[..., 1] * b1).sum(-1, keepdims=True))
    b2 = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)                      # [J,3,3]
    # 다시 rot6 평면으로 투영(수치적 안정 위해)
    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return np.concatenate([col1, col2], axis=-1).reshape(-1) # (J*6,)

def expand_267_to_630(frame267, keep_idx_44, prev630=None):
    """
    입력:
      - frame267: (267,) = [trans(3), 44*rot6(264)]
      - keep_idx_44: 길이 44인 numpy 배열 (0-based, SMPL-H 52 기준)
      - prev630: (630,) 이전 프레임(Δ 계산용). 없으면 Δ는 0으로.
    출력:
      - frame630: (630,) = [trans(3), (52*6), dtrans(3), d(52*6)]
    """
    assert frame267.shape[-1] == 267
    trans = frame267[:3].astype(np.float32)                  # [3]
    rot6_44 = frame267[3:].astype(np.float32)                # [264] = 44*6

    # 52개 관절 rot6로 확장, 누락 8개는 항등(I) -> rot6 = [1,0,0, 0,1,0]
    rot6_full = np.zeros((52, 6), dtype=np.float32)
    rot6_full[:] = np.array([1,0,0, 0,1,0], dtype=np.float32) # identity
    rot6_full[keep_idx_44] = rot6_44.reshape(44, 6)

    # 수치 안정화를 위해 재직교화
    rot6_full = rot6_reorth(rot6_full.reshape(-1)).reshape(52, 6)

    # Δ 계산 (없으면 0)
    if prev630 is not None:
        prev_trans = prev630[:3]
        prev_rot6_full = prev630[3:3+52*6]
        dtrans = (trans - prev_trans).astype(np.float32)
        drot6 = (rot6_full.reshape(-1) - prev_rot6_full).astype(np.float32)
    else:
        dtrans = np.zeros(3, dtype=np.float32)
        drot6  = np.zeros(52*6, dtype=np.float32)

    frame630 = np.concatenate([
        trans,
        rot6_full.reshape(-1),  # 52*6
        dtrans,
        drot6
    ], axis=0).astype(np.float32)  # (630,)
    return frame630


def adapt_batch_267_to_630(frames267_bt, keep_idx_44, prev630_b=None):
    """
    배치 변환: 
      frames267_bt: [B, Tskip, 267]
      prev630_b   : [B, 630] 또는 None
    반환:
      frames630_bt: [B, Tskip, 630]
      last630_b   : [B, 630]  (다음 스텝의 prev로 사용)
    """
    B, Tskip, D = frames267_bt.shape
    out = np.zeros((B, Tskip, 630), dtype=np.float32)
    last = np.zeros((B, 630), dtype=np.float32)
    for b in range(B):
        prev = None if prev630_b is None else prev630_b[b]
        for t in range(Tskip):
            out[b, t] = expand_267_to_630(frames267_bt[b, t], keep_idx_44, prev)
            prev = out[b, t]
        last[b] = prev
    return out, last


# =================================
# 메인 실행 로직 (원본에 어댑터 삽입)
# =================================

def test_vae_replay_full_motion(args):
    device = "cpu"

    num_characters = args.num
    pose_vae_path = os.path.join(os.getcwd(), args.vae)

    # 267D 데이터/모델을 그대로 사용 (추가 변환 없음)

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
    mocap_data = env.mocap_data                       # NOTE: 현재 이는 267D일 가능성이 큼
    num_future_predictions = env.pose_vae_model.num_future_predictions

    latent_size = env.action_space.shape[0]
    action_shape = (num_characters, latent_size)
    action = torch.empty(action_shape).to(device)

    # overwrite if necessary
    if args.frame != -1:
        env.reset_initial_frames(args.frame)

    alpha = torch.ones(num_characters).float()

    # prev630: Δ 계산용 이전 프레임(캐릭터별)
    prev630_b = None

    with EpisodeRunner(env, save=args.save, max_steps=args.len, csv=args.csv) as runner:

        while not runner.done:

            action.normal_(0, 1.0)
            # frames: torch tensor [B, Tskip, D]; D=267
            frames = env.get_vae_next_frame(action)

            for i in range(env.frame_skip):
                frame_indices = (
                    env.start_indices
                    + env.pose_vae_model.num_condition_frames
                    + env.timestep
                    + env.substep * env.frame_skip
                )

                alpha_ = env.viewer.controller_autonomy if is_rendered else 1.0
                alpha.fill_(alpha_)
                prediction_range = (
                    frame_indices.repeat((num_future_predictions, 1)).t()
                    + torch.arange(0, num_future_predictions).long()
                ).remainder_(mocap_data.shape[0])

                if args.mocap:
                    alpha[0] = 0

                # 정규화 공간에서 바로 혼합
                frames_mixed = (
                    alpha.view(-1, 1, 1) * frames
                    + (1.0 - alpha).view(-1, 1, 1) * mocap_data[prediction_range]
                )

                frame = frames_mixed[:, i]
                _, _, done, info = env.calc_env_state(frame)

                if done.any():
                    reset_indices = env.parallel_ind_buf.masked_select(done.squeeze())
                    env.reset(reset_indices)

                if info.get("reset"):
                    env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="RandomWalkEnv-v0", required=False, help="Environment name")
    parser.add_argument("--vae", type=str, help="Path to VAE file", required=True)
    parser.add_argument("--num", type=int, default=1, help="Number of characters to simulate")
    parser.add_argument("--gui", type=int, default=1, help="Show parameters in GUI")
    parser.add_argument("--mocap", action="store_true", default=False, help="Play first character with pure mocap")
    parser.add_argument("--track", type=int, default=1, help="1 - camera tracks character | 0 - static camera")
    parser.add_argument("--frame", type=int, default=-1, help="Initial frame for random walk (-1 for random)")
    parser.add_argument("--skip", type=int, default=1, help="Number of internal steps (minus 1) per action")
    parser.add_argument("--save", action="store_true", default=False, help="Save video recorded from camera")
    parser.add_argument("--len", type=int, default=None, help="Length of video to save in number of frames")
    parser.add_argument("--csv", type=str, default=None, required=False, help="CSV path to dump trajectory")

    # --- 새로 추가된 옵션 ---
    parser.add_argument("--mocap_npz", type=str, default="environments/mocap.npz",
                        help="Path to the mocap.npz used for training (for denorm μ,σ)")
    parser.add_argument("--keep_joints_mode", type=str, choices=["range", "csv"], default="range",
                        help="How to specify 44 kept joints (SMPL-H indices)")
    parser.add_argument("--keep_lo", type=int, default=0, help="If mode=range, low (inclusive)")
    parser.add_argument("--keep_hi", type=int, default=43, help="If mode=range, high (inclusive)")
    parser.add_argument("--keep_joints_csv", type=str,
                        default=",".join(str(i) for i in range(44)),
                        help="If mode=csv, a comma-separated list of exactly 44 indices")

    args = parser.parse_args()

    ps = [mp.Process(target=test_vae_replay_full_motion, args=(args,)),]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
