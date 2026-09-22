"""Convert legacy pickled .pt checkpoints to safetensors.

Run once per checkpoint, then delete the .pt file. This script is the only
place in the repo that unpickles a checkpoint; `torch.load` with
`weights_only=False` executes arbitrary code from the file, so only run it on
checkpoints you trust.

    python tools/convert_checkpoints.py vae_motion/models --verify
"""
import argparse
import glob
import os
import sys

import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vae_motion.models import (  # noqa: E402
    NormalizationMixin,
    PoseMixtureSpecialistVAE,
    PoseVAEPolicy,
    PoseVQVAE,
    load_model,
    save_model,
)

NORMALIZATION_BUFFERS = ["data_" + key for key in NormalizationMixin.STAT_KEYS]


def promote_normalization_buffers(model):
    """Turn the legacy plain attributes into real registered buffers.

    Unpickling restores __dict__ directly and never calls __init__, so
    data_max/min/avg/std arrive as plain attributes and are absent from
    state_dict(). Saving without this step would silently drop all four and
    leave normalize()/denormalize() producing wrong output with no error.
    """
    if not isinstance(model, NormalizationMixin):
        return
    normalization = {"mode": model.__dict__.get("mode")}
    for key in NormalizationMixin.STAT_KEYS:
        # pop first: register_buffer refuses a name that already exists.
        value = model.__dict__.pop("data_" + key, None)
        if value is not None:
            normalization[key] = value
    model._init_normalization(normalization)


def derive_extra_config(model):
    """Recover config that is not reachable through `model.config`."""
    extra = {}
    env_info = getattr(model, "env_info", None)
    if isinstance(env_info, dict) and "frame_skip" in env_info:
        extra["frame_skip"] = env_info["frame_skip"]
    elif isinstance(model, PoseVAEPolicy):
        extra["frame_skip"] = 1
    return extra


def deterministic_probe(model, config):
    """A seeded, sampling-free forward pass for equivalence checking."""
    model.eval()
    torch.manual_seed(1234)
    with torch.no_grad():
        if isinstance(model, PoseVAEPolicy):
            obs = torch.randn(4, config["observation_dim"])
            return torch.cat((model.actor(obs), model.critic(obs)), dim=1)

        frame_size = config["frame_size"]
        condition = torch.randn(4, frame_size * config["num_condition_frames"])
        if isinstance(model, PoseVQVAE):
            logits = torch.randn(4, config["num_embeddings"])
            return model.sample(logits, condition, deterministic=True)
        latent = torch.randn(4, config["latent_size"])
        if isinstance(model, PoseMixtureSpecialistVAE):
            return model.sample(latent, condition, deterministic=True)
        return model.sample(latent, condition)


def normalization_probe(model):
    """Output of normalize/denormalize, which the forward probe never touches."""
    if not isinstance(model, NormalizationMixin) or model.mode in (None, "none"):
        return None
    torch.manual_seed(4321)
    x = torch.randn(4, model.frame_size)
    with torch.no_grad():
        return torch.cat((model.normalize(x), model.denormalize(x)), dim=1)


def convert(pt_path, verify):
    out_path = os.path.splitext(pt_path)[0] + ".safetensors"

    # weights_only=False is required: torch 2.6+ defaults it to True, which
    # refuses whole-object pickles outright.
    original = torch.load(pt_path, map_location="cpu", weights_only=False)
    promote_normalization_buffers(original)
    extra = derive_extra_config(original)
    save_model(original, out_path, extra_config=extra)

    if not verify:
        print("converted %s -> %s" % (pt_path, out_path))
        return True

    reloaded = load_model(out_path)
    config = dict(original.config)
    config.update(extra)

    before, after = original.state_dict(), reloaded.state_dict()
    if set(before) != set(after):
        print("FAIL %s: state_dict keys differ" % pt_path)
        return False
    for name in before:
        if not torch.equal(before[name], after[name]):
            print("FAIL %s: tensor %s differs" % (pt_path, name))
            return False

    # Guard the migration's central risk explicitly: the forward probe below
    # never calls normalize(), so a dropped statistic would slip past it.
    if isinstance(original, NormalizationMixin):
        missing = [n for n in NORMALIZATION_BUFFERS if n not in after]
        if missing:
            print("FAIL %s: normalization buffers absent: %s" % (pt_path, missing))
            return False
        probe_before = normalization_probe(original)
        probe_after = normalization_probe(reloaded)
        if probe_before is not None and not torch.equal(probe_before, probe_after):
            print("FAIL %s: normalize/denormalize output differs" % pt_path)
            return False

    if not torch.equal(
        deterministic_probe(original, config), deterministic_probe(reloaded, config)
    ):
        print("FAIL %s: forward output differs" % pt_path)
        return False

    print(
        "OK   %-28s -> %-38s (%d tensors, output identical)"
        % (os.path.basename(pt_path), os.path.basename(out_path), len(before))
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A .pt file, or a directory of them")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Assert the converted file reproduces the original exactly",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        targets = sorted(glob.glob(os.path.join(args.path, "*.pt")))
    else:
        targets = [args.path]
    if not targets:
        print("no .pt files found at %s" % args.path)
        return 1

    results = [convert(target, args.verify) for target in targets]
    ok = all(results)
    print("\n%d file(s), %s" % (len(targets), "all verified" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
