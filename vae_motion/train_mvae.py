import copy
import os
import time
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule
from vae_motion.models import (
    PoseVAE,
    PoseVQVAE,
    PoseMixtureVAE,
    PoseMixtureSpecialistVAE,
)
from mvae_dataloader_accad import MVAEACCADDataset


class StatsLogger:
    def __init__(self, args, csv_path):
        self.start = time.time()
        self.logger = CSVLogger(log_path=csv_path)
        self.num_epochs = args.num_epochs
        self.progress_format = None

    def time_since(self, ep):
        now = time.time()
        elapsed = now - self.start
        estimated = elapsed * self.num_epochs / ep
        remaining = estimated - elapsed

        em, es = divmod(elapsed, 60)
        rm, rs = divmod(remaining, 60)

        if self.progress_format is None:
            time_format = "%{:d}dm %02ds".format(int(np.log10(rm) + 1))
            perc_format = "%{:d}d %5.1f%%".format(int(np.log10(self.num_epochs) + 1))
            self.progress_format = f"{time_format} (- {time_format}) ({perc_format})"

        return self.progress_format % (em, es, rm, rs, ep, ep / self.num_epochs * 100)

    def log_stats(self, data):
        self.logger.log_epoch(data)

        ep = data["epoch"]
        ep_recon_loss = data["ep_recon_loss"]
        ep_kl_loss = data["ep_kl_loss"]
        ep_perplexity = data["ep_perplexity"]

        print(
            "{} | Recon: {:.3e} | KL: {:.3e} | PP: {:.3e}".format(
                self.time_since(ep), ep_recon_loss, ep_kl_loss, ep_perplexity
            ),
            flush=True,
        )


def feed_vae(pose_vae, ground_truth, condition, future_weights):
    condition = condition.flatten(start_dim=1, end_dim=2)
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)

    output_shape = (-1, pose_vae.num_future_predictions, pose_vae.frame_size)

    if isinstance(pose_vae, PoseVQVAE):
        vae_output, vq_loss, perplexity = pose_vae(flattened_truth, condition)
        vae_output = vae_output.view(output_shape)

        # recon_loss = F.mse_loss(vae_output, ground_truth)
        recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
        recon_loss = recon_loss.mul(future_weights).sum()

        return (vae_output, perplexity), (recon_loss, vq_loss)

    elif isinstance(pose_vae, PoseMixtureSpecialistVAE):
        vae_output, mu, logvar, coefficient = pose_vae(flattened_truth, condition)

        recon_loss = (vae_output - ground_truth).pow(2).mean(dim=2).mul(-0.5).exp()
        recon_loss = (recon_loss * coefficient).sum(dim=1).log().mul(-1).mean()

        # Sample a next frame from experts
        indices = torch.distributions.Categorical(coefficient).sample()
        # was (expert, batch, feature), after select is (batch, feature)
        vae_output = vae_output[torch.arange(vae_output.size(0)), indices]
        vae_output = vae_output.view(output_shape)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        return (vae_output, mu, logvar), (recon_loss, kl_loss)

    else:
        # PoseVAE and PoseMixtureVAE
        vae_output, mu, logvar = pose_vae(flattened_truth, condition)
        vae_output = vae_output.view(output_shape)

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
        recon_loss = recon_loss.mul(future_weights).sum()

        return (vae_output, mu, logvar), (recon_loss, kl_loss)


def main():
    env_path = os.path.join(parent_dir, "environments")

    # setup parameters
    args = SimpleNamespace(
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        mocap_file=os.path.join(env_path, "mocap.npz"),
        norm_mode="zscore",
        latent_size=32,
        num_embeddings=12,
        num_experts=6,
        num_condition_frames=1,
        num_future_predictions=1,
        num_steps_per_rollout=8,
        kl_beta=1.0,
        load_saved_model=False,
    )

    # learning parameters
    teacher_epochs = 20
    ramping_epochs = 20
    student_epochs = 100
    args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    # 데이터셋 로딩 교체
    ds = MVAEACCADDataset(npz_path=args.mocap_file)
    dl = DataLoader(ds, batch_size=args.mini_batch_size, shuffle=True, drop_last=True, num_workers=2)
    frame_size = ds.dim
    
    # 정규화 정보 설정
    normalization = {
        "mode": args.norm_mode,
        "avg": torch.from_numpy(ds.mean).float().to(args.device),
        "std": torch.from_numpy(ds.std).float().to(args.device),
    }

    pose_vae = PoseMixtureVAE(
        frame_size,
        args.latent_size,
        args.num_condition_frames,
        args.num_future_predictions,
        normalization,
        args.num_experts,
    ).to(args.device)

    if isinstance(pose_vae, PoseVAE):
        pose_vae_path = "posevae_c{}_l{}.pt".format(
            args.num_condition_frames, args.latent_size
        )
    elif isinstance(pose_vae, PoseMixtureVAE):
        pose_vae_path = "posevae_c{}_e{}_l{}.pt".format(
            args.num_condition_frames, args.num_experts, args.latent_size
        )
    elif isinstance(pose_vae, PoseMixtureSpecialistVAE):
        pose_vae_path = "posevae_c{}_s{}_l{}.pt".format(
            args.num_condition_frames, args.num_experts, args.latent_size
        )
    elif isinstance(pose_vae, PoseVQVAE):
        pose_vae_path = "posevae_c{}_n{}_l{}.pt".format(
            args.num_condition_frames, args.num_embeddings, args.latent_size
        )

    if args.load_saved_model:
        pose_vae = torch.load(pose_vae_path, map_location=args.device)
    pose_vae.train()

    vae_optimizer = optim.Adam(pose_vae.parameters(), lr=args.initial_lr)

    sample_schedule = torch.cat(
        (
            # First part is pure teacher forcing
            torch.zeros(teacher_epochs),
            # Second part with schedule sampling
            torch.linspace(0.0, 1.0, ramping_epochs),
            # last part is pure student
            torch.ones(student_epochs),
        )
    )

    # future_weights = torch.softmax(
    #     torch.linspace(1, 0, args.num_future_predictions), dim=0
    # ).to(args.device)

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    # DataLoader를 사용하므로 history buffer는 필요 없음

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    for ep in range(1, args.num_epochs + 1):
        ep_recon_loss = 0
        ep_kl_loss = 0
        ep_perplexity = 0

        update_linear_schedule(
            vae_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        )

        num_mini_batch = 0
        for batch_idx, (prev_frames, curr_frames) in enumerate(dl):
            num_mini_batch += 1
            
            # 데이터를 GPU로 이동
            prev_frames = prev_frames.to(args.device)
            curr_frames = curr_frames.to(args.device)
            
            # 배치 크기 조정
            batch_size = prev_frames.size(0)
            if batch_size != args.mini_batch_size:
                continue
                
            # condition과 ground_truth 설정
            condition = prev_frames.unsqueeze(1)  # (batch_size, 1, frame_size)
            ground_truth = curr_frames.unsqueeze(1)  # (batch_size, 1, frame_size)
            
            # VAE forward pass
            if isinstance(pose_vae, PoseVQVAE):
                (vae_output, perplexity), (recon_loss, kl_loss) = feed_vae(
                    pose_vae, ground_truth, condition, future_weights
                )
                ep_perplexity += float(perplexity)
            else:
                # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE
                (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
                    pose_vae, ground_truth, condition, future_weights
                )

            vae_optimizer.zero_grad()
            (recon_loss + args.kl_beta * kl_loss).backward()
            vae_optimizer.step()

            ep_recon_loss += float(recon_loss)
            ep_kl_loss += float(kl_loss)

        if num_mini_batch > 0:
            avg_ep_recon_loss = ep_recon_loss / num_mini_batch
            avg_ep_kl_loss = ep_kl_loss / num_mini_batch
            avg_ep_perplexity = ep_perplexity / num_mini_batch
        else:
            avg_ep_recon_loss = 0
            avg_ep_kl_loss = 0
            avg_ep_perplexity = 0

        logger.log_stats(
            {
                "epoch": ep,
                "ep_recon_loss": avg_ep_recon_loss,
                "ep_kl_loss": avg_ep_kl_loss,
                "ep_perplexity": avg_ep_perplexity,
            }
        )

        torch.save(copy.deepcopy(pose_vae).cpu(), pose_vae_path)


if __name__ == "__main__":
    main()
