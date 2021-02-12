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

from common.logging_utils import CSVLogger
from common.misc_utils import update_linear_schedule
from vae_motion.models import (
    PoseVAE,
    PoseVQVAE,
    PoseMixtureVAE,
    PoseMixtureSpecialistVAE,
)


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
        load_saved_model=True,
    )

    # learning parameters
    teacher_epochs = 20
    ramping_epochs = 20
    student_epochs = 100
    args.num_epochs = teacher_epochs + ramping_epochs + student_epochs
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7

    raw_data = np.load(args.mocap_file)
    mocap_data = torch.from_numpy(raw_data["data"]).float().to(args.device)
    end_indices = raw_data["end_indices"]

    max = mocap_data.max(dim=0)[0]
    min = mocap_data.min(dim=0)[0]
    avg = mocap_data.mean(dim=0)
    std = mocap_data.std(dim=0)

    # Make sure we don't divide by 0
    std[std == 0] = 1.0

    normalization = {
        "mode": args.norm_mode,
        "max": max,
        "min": min,
        "avg": avg,
        "std": std,
    }

    if args.norm_mode == "minmax":
        mocap_data = 2 * (mocap_data - min) / (max - min) - 1

    elif args.norm_mode == "zscore":
        mocap_data = (mocap_data - avg) / std

    batch_size = mocap_data.size()[0]
    frame_size = mocap_data.size()[1]

    # bad indices are ones that has no required next frames
    # need to take account of num_steps_per_rollout and num_future_predictions
    bad_indices = np.sort(
        np.concatenate(
            [
                end_indices - i
                for i in range(
                    args.num_steps_per_rollout
                    + (args.num_condition_frames - 1)
                    + (args.num_future_predictions - 1)
                )
            ]
        )
    )
    all_indices = np.arange(batch_size)
    good_masks = np.isin(all_indices, bad_indices, assume_unique=True, invert=True)
    selectable_indices = all_indices[good_masks]

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

    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    history = torch.empty(shape).to(args.device)

    log_path = os.path.join(current_dir, "log_posevae_progress")
    logger = StatsLogger(args, csv_path=log_path)

    for ep in range(1, args.num_epochs + 1):
        sampler = BatchSampler(
            SubsetRandomSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        ep_recon_loss = 0
        ep_kl_loss = 0
        ep_perplexity = 0

        update_linear_schedule(
            vae_optimizer, ep - 1, args.num_epochs, args.initial_lr, args.final_lr
        )

        num_mini_batch = 1
        for num_mini_batch, indices in enumerate(sampler):
            t_indices = torch.LongTensor(indices)

            # condition is from newest...oldest, i.e. (t-1, t-2, ... t-n)
            condition_range = (
                t_indices.repeat((args.num_condition_frames, 1)).t()
                + torch.arange(args.num_condition_frames - 1, -1, -1).long()
            )

            t_indices += args.num_condition_frames
            history[:, : args.num_condition_frames].copy_(mocap_data[condition_range])

            for offset in range(args.num_steps_per_rollout):
                # dims: (num_parallel, num_window, feature_size)
                use_student = torch.rand(1) < sample_schedule[ep - 1]

                prediction_range = (
                    t_indices.repeat((args.num_future_predictions, 1)).t()
                    + torch.arange(offset, offset + args.num_future_predictions).long()
                )
                ground_truth = mocap_data[prediction_range]
                condition = history[:, : args.num_condition_frames]

                if isinstance(pose_vae, PoseVQVAE):
                    (vae_output, perplexity), (recon_loss, kl_loss) = feed_vae(
                        pose_vae, ground_truth, condition, future_weights
                    )
                    ep_perplexity += float(perplexity) / args.num_steps_per_rollout
                else:
                    # PoseVAE, PoseMixtureVAE, PoseMixtureSpecialistVAE
                    (vae_output, _, _), (recon_loss, kl_loss) = feed_vae(
                        pose_vae, ground_truth, condition, future_weights
                    )

                history = history.roll(1, dims=1)
                next_frame = vae_output[:, 0] if use_student else ground_truth[:, 0]
                history[:, 0].copy_(next_frame.detach())

                vae_optimizer.zero_grad()
                (recon_loss + args.kl_beta * kl_loss).backward()
                vae_optimizer.step()

                ep_recon_loss += float(recon_loss) / args.num_steps_per_rollout
                ep_kl_loss += float(kl_loss) / args.num_steps_per_rollout

        avg_ep_recon_loss = ep_recon_loss / num_mini_batch
        avg_ep_kl_loss = ep_kl_loss / num_mini_batch
        avg_ep_perplexity = ep_perplexity / num_mini_batch

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
