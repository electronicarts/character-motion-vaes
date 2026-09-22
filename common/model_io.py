"""Reading and writing model checkpoints in safetensors format.

Deliberately knows nothing about any model class: `common/` must not import
`vae_motion/` or `environments/`. The model-aware layer lives in
`vae_motion/models.py`.

A checkpoint stores the model's state_dict as tensors, plus its architecture
hyperparameters as a JSON string in the safetensors header metadata (which is
a str -> str map, so the config has to be encoded).
"""
import json

from safetensors import safe_open
from safetensors.torch import save_file

CONFIG_KEY = "config"
VERSION_KEY = "cmv_format_version"
FORMAT_VERSION = "1"


class CheckpointFormatError(Exception):
    """Raised when a file is not a valid character-motion-vaes checkpoint."""


def save_tensors_with_config(tensors, config, path):
    """Write `tensors` and `config` to `path` as a safetensors file."""
    # safetensors rejects shared storage and non-contiguous layouts, so
    # detach/clone/contiguous every tensor rather than trusting the caller.
    prepared = {
        name: tensor.detach().cpu().clone().contiguous()
        for name, tensor in tensors.items()
    }
    metadata = {
        "format": "pt",
        VERSION_KEY: FORMAT_VERSION,
        CONFIG_KEY: json.dumps(config, sort_keys=True),
    }
    save_file(prepared, path, metadata=metadata)


def load_tensors_with_config(path, device="cpu"):
    """Return `(tensors, config)` from a checkpoint written by this module."""
    with safe_open(path, framework="pt", device=str(device)) as f:
        config = _parse_metadata(f.metadata(), path)
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return tensors, config


def read_config_from_file(path):
    """Return just the config, without reading any tensor data."""
    with safe_open(path, framework="pt", device="cpu") as f:
        return _parse_metadata(f.metadata(), path)


def _parse_metadata(metadata, path):
    metadata = metadata or {}
    if CONFIG_KEY not in metadata:
        raise CheckpointFormatError(
            "{} has no {!r} metadata, so it is not a character-motion-vaes "
            "checkpoint".format(path, CONFIG_KEY)
        )
    version = metadata.get(VERSION_KEY)
    if version != FORMAT_VERSION:
        raise CheckpointFormatError(
            "{} has format version {!r}, but this code reads version {!r}".format(
                path, version, FORMAT_VERSION
            )
        )
    return json.loads(metadata[CONFIG_KEY])
