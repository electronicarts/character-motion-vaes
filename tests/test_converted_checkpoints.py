import glob
import os

import pytest
import torch

from vae_motion.models import PoseMixtureVAE, PoseVAEPolicy, load_model, read_config

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vae_motion", "models"
)

EXPECTED = {
    "posevae_c1_e6_l32": {
        "class": "PoseMixtureVAE",
        "frame_size": 267,
        "latent_size": 32,
        "num_condition_frames": 1,
        "num_future_predictions": 1,
        "num_experts": 6,
        "normalization_mode": "zscore",
    },
    "con_TargetEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 269,
        "action_dim": 32,
        "frame_skip": 1,
    },
    "con_JoystickEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 269,
        "action_dim": 32,
        "frame_skip": 1,
    },
    "con_TimedTargetEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 270,
        "action_dim": 32,
        "frame_skip": 1,
    },
    "con_PathFollowEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 275,
        "action_dim": 32,
        "frame_skip": 1,
    },
    "con_HumanMazeEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 283,
        "action_dim": 2,
        "frame_skip": 1,
    },
    "con_HumanPacmanEnv-v0": {
        "class": "PoseVAEPolicy",
        "observation_dim": 283,
        "action_dim": 2,
        "frame_skip": 1,
    },
}


def test_all_expected_checkpoints_are_present():
    found = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(MODELS_DIR, "*.safetensors"))
    }
    assert found == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_checkpoint_config_matches_expected(name):
    config = read_config(os.path.join(MODELS_DIR, name + ".safetensors"))
    assert config == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_checkpoint_loads_and_runs_forward(name):
    config = EXPECTED[name]
    model = load_model(os.path.join(MODELS_DIR, name + ".safetensors"))
    model.eval()
    torch.manual_seed(0)

    with torch.no_grad():
        if config["class"] == "PoseVAEPolicy":
            assert isinstance(model, PoseVAEPolicy)
            out = model.actor(torch.randn(2, config["observation_dim"]))
            assert out.shape == (2, config["action_dim"])
        else:
            assert isinstance(model, PoseMixtureVAE)
            latent = torch.randn(2, config["latent_size"])
            condition = torch.randn(
                2, config["frame_size"] * config["num_condition_frames"]
            )
            out = model.sample(latent, condition)
            assert out.shape == (
                2,
                config["frame_size"] * config["num_future_predictions"],
            )
    assert torch.isfinite(out).all()


def test_vae_normalization_statistics_were_preserved():
    # The migration's biggest risk: these were absent from state_dict before
    # the buffer fix, so a dropped statistic would silently corrupt every
    # rollout without raising anything.
    model = load_model(os.path.join(MODELS_DIR, "posevae_c1_e6_l32.safetensors"))
    state = model.state_dict()
    for name in ("data_max", "data_min", "data_avg", "data_std"):
        assert name in state, name
        stat = getattr(model, name)
        assert stat is not None, name
        assert stat.shape == (267,), name
        assert torch.isfinite(stat).all(), name
    assert (model.data_std > 0).all()
    assert (model.data_max >= model.data_min).all()


def test_vae_normalization_round_trips_on_real_statistics():
    model = load_model(os.path.join(MODELS_DIR, "posevae_c1_e6_l32.safetensors"))
    torch.manual_seed(0)
    x = torch.randn(4, 267)
    assert torch.allclose(model.denormalize(model.normalize(x)), x, atol=1e-4)
