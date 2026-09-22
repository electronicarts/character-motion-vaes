import pytest
import torch

from common.model_io import CheckpointFormatError
from vae_motion.models import (
    AutoEncoder,
    PoseMixtureSpecialistVAE,
    PoseMixtureVAE,
    PoseVAE,
    PoseVQVAE,
    load_model,
    read_config,
    save_model,
)

FRAME = 8
LATENT = 4
COND = 1
FUTURE = 1
EXPERTS = 3
EMBEDDINGS = 5


def normalization():
    return {
        "mode": "zscore",
        "max": torch.full((FRAME,), 3.0),
        "min": torch.full((FRAME,), -3.0),
        "avg": torch.zeros(FRAME),
        "std": torch.full((FRAME,), 2.0),
    }


def build(cls):
    torch.manual_seed(0)
    norm = normalization()
    if cls is AutoEncoder:
        return cls(FRAME, LATENT, norm)
    if cls is PoseVAE:
        return cls(FRAME, LATENT, COND, FUTURE, norm)
    if cls is PoseVQVAE:
        return cls(FRAME, LATENT, EMBEDDINGS, COND, FUTURE, norm)
    return cls(FRAME, LATENT, COND, FUTURE, norm, EXPERTS)


def deterministic_output(model):
    """A forward pass with no sampling, for output-equality comparisons."""
    model.eval()
    torch.manual_seed(1234)
    z = torch.randn(2, LATENT)
    c = torch.randn(2, FRAME * COND)
    with torch.no_grad():
        if isinstance(model, AutoEncoder):
            return model(torch.randn(2, FRAME))
        if isinstance(model, PoseVQVAE):
            logits = torch.randn(2, EMBEDDINGS)
            return model.sample(logits, c, deterministic=True)
        if isinstance(model, PoseMixtureSpecialistVAE):
            return model.sample(z, c, deterministic=True)
        return model.sample(z, c)


VAE_CLASSES = [
    AutoEncoder,
    PoseVAE,
    PoseMixtureVAE,
    PoseMixtureSpecialistVAE,
    PoseVQVAE,
]


@pytest.mark.parametrize("cls", VAE_CLASSES)
def test_state_dict_survives_round_trip_exactly(cls, tmp_path):
    model = build(cls)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)

    reloaded = load_model(path)

    assert type(reloaded) is cls
    original = model.state_dict()
    restored = reloaded.state_dict()
    assert set(original) == set(restored)
    for name in original:
        assert torch.equal(original[name], restored[name]), name


@pytest.mark.parametrize("cls", VAE_CLASSES)
def test_forward_output_survives_round_trip_exactly(cls, tmp_path):
    model = build(cls)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)
    reloaded = load_model(path)

    assert torch.equal(deterministic_output(model), deterministic_output(reloaded))


@pytest.mark.parametrize("cls", VAE_CLASSES)
def test_config_records_class_name_and_mode(cls, tmp_path):
    model = build(cls)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)

    config = read_config(path)
    assert config["class"] == cls.__name__
    assert config["normalization_mode"] == "zscore"
    assert config["frame_size"] == FRAME
    assert config["latent_size"] == LATENT


@pytest.mark.parametrize("cls", VAE_CLASSES)
def test_normalization_statistics_survive_round_trip(cls, tmp_path):
    # The migration's central risk: these must reach the file as tensors.
    model = build(cls)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)
    reloaded = load_model(path)

    for name in ("data_max", "data_min", "data_avg", "data_std"):
        assert name in reloaded.state_dict(), name
        assert torch.equal(getattr(reloaded, name), getattr(model, name)), name

    x = torch.randn(2, FRAME)
    assert torch.equal(model.normalize(x), reloaded.normalize(x))
    assert torch.equal(model.denormalize(x), reloaded.denormalize(x))


def test_mixture_config_records_num_experts(tmp_path):
    path = str(tmp_path / "m.safetensors")
    save_model(build(PoseMixtureVAE), path)
    assert read_config(path)["num_experts"] == EXPERTS


def test_specialist_config_records_num_experts(tmp_path):
    path = str(tmp_path / "m.safetensors")
    save_model(build(PoseMixtureSpecialistVAE), path)
    assert read_config(path)["num_experts"] == EXPERTS


def test_vqvae_config_records_num_embeddings(tmp_path):
    path = str(tmp_path / "m.safetensors")
    save_model(build(PoseVQVAE), path)
    assert read_config(path)["num_embeddings"] == EMBEDDINGS


def test_extra_config_is_merged(tmp_path):
    path = str(tmp_path / "m.safetensors")
    save_model(build(PoseVAE), path, extra_config={"frame_skip": 4})
    assert read_config(path)["frame_skip"] == 4


def test_mixed_decoder_forward_uses_loaded_weights(tmp_path):
    # MixedDecoder keeps its parameters in a plain Python list of tuples AND
    # registers them as w0..w2/b0..b2. load_state_dict copies in place, so the
    # list's references stay valid -- load-bearing but non-obvious.
    model = build(PoseMixtureVAE)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)

    reloaded = load_model(path)
    with torch.no_grad():
        for weight, _bias, _act in reloaded.decoder.decoder_layers:
            weight.zero_()

    # Zeroing via the list must be visible through the registered parameters,
    # proving both views alias the same storage.
    assert torch.equal(reloaded.decoder.w0, torch.zeros_like(reloaded.decoder.w0))
    assert not torch.equal(deterministic_output(model), deterministic_output(reloaded))


def test_unknown_class_in_config_is_rejected(tmp_path):
    from common.model_io import save_tensors_with_config

    path = str(tmp_path / "m.safetensors")
    save_tensors_with_config({"a": torch.zeros(2)}, {"class": "NoSuchModel"}, path)

    with pytest.raises(CheckpointFormatError, match="NoSuchModel"):
        load_model(path)


def test_load_rejects_tensor_set_that_does_not_match_architecture(tmp_path):
    from common.model_io import load_tensors_with_config, save_tensors_with_config

    model = build(PoseVAE)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)

    # Drop a required tensor; strict loading must complain rather than
    # silently produce a partly-initialised model.
    tensors, config = load_tensors_with_config(path)
    del tensors["fc1.weight"]
    save_tensors_with_config(tensors, config, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)
