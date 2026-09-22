import pytest
import torch

from vae_motion.models import (
    AutoEncoder,
    PoseMixtureSpecialistVAE,
    PoseMixtureVAE,
    PoseVAE,
    PoseVQVAE,
)

FRAME = 8
LATENT = 4
COND = 1
FUTURE = 1
EXPERTS = 3
EMBEDDINGS = 5

STAT_NAMES = ["data_max", "data_min", "data_avg", "data_std"]


def normalization(mode="zscore"):
    return {
        "mode": mode,
        "max": torch.full((FRAME,), 3.0),
        "min": torch.full((FRAME,), -3.0),
        "avg": torch.zeros(FRAME),
        "std": torch.full((FRAME,), 2.0),
    }


def build(cls, mode="zscore"):
    torch.manual_seed(0)
    norm = normalization(mode)
    if cls is AutoEncoder:
        return cls(FRAME, LATENT, norm)
    if cls is PoseVAE:
        return cls(FRAME, LATENT, COND, FUTURE, norm)
    if cls is PoseVQVAE:
        return cls(FRAME, LATENT, EMBEDDINGS, COND, FUTURE, norm)
    return cls(FRAME, LATENT, COND, FUTURE, norm, EXPERTS)


ALL_CLASSES = [
    AutoEncoder,
    PoseVAE,
    PoseMixtureVAE,
    PoseMixtureSpecialistVAE,
    PoseVQVAE,
]


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_normalization_stats_appear_in_state_dict(cls):
    # Regression test: these were plain attributes, so they were absent from
    # state_dict and a state_dict-based save would silently drop them.
    model = build(cls)
    keys = model.state_dict().keys()
    for name in STAT_NAMES:
        assert name in keys, "{} missing from {} state_dict".format(name, cls.__name__)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_normalization_stats_are_buffers_not_parameters(cls):
    model = build(cls)
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    for name in STAT_NAMES:
        assert name in buffers
        assert name not in parameters


@pytest.mark.parametrize("cls", ALL_CLASSES)
@pytest.mark.parametrize("mode", ["minmax", "zscore", "none"])
def test_denormalize_inverts_normalize(cls, mode):
    model = build(cls, mode)
    model.eval()
    x = torch.randn(2, FRAME)
    assert torch.allclose(model.denormalize(model.normalize(x)), x, atol=1e-5)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_unknown_normalization_mode_raises(cls):
    model = build(cls, "bogus")
    with pytest.raises(ValueError, match="Unknown normalization mode"):
        model.normalize(torch.randn(2, FRAME))


def test_zscore_matches_explicit_formula():
    model = build(PoseVAE, "zscore")
    x = torch.randn(2, FRAME)
    expected = (x - model.data_avg) / model.data_std
    assert torch.allclose(model.normalize(x), expected)


def test_minmax_matches_explicit_formula():
    model = build(PoseVAE, "minmax")
    x = torch.randn(2, FRAME)
    expected = 2 * (x - model.data_min) / (model.data_max - model.data_min) - 1
    assert torch.allclose(model.normalize(x), expected)


def test_moving_model_to_device_moves_normalization_stats():
    # Buffers follow .to(); plain attributes did not.
    model = build(PoseVAE).to("cpu")
    for name in STAT_NAMES:
        assert getattr(model, name).device.type == "cpu"


def test_normalization_from_tensors_rebuilds_constructor_argument():
    model = build(PoseVAE)
    tensors = model.state_dict()
    config = {"normalization_mode": "zscore"}
    rebuilt = PoseVAE.normalization_from_tensors(config, tensors)
    assert rebuilt["mode"] == "zscore"
    assert torch.equal(rebuilt["max"], model.data_max)
    assert torch.equal(rebuilt["std"], model.data_std)
