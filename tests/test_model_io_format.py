import json

import pytest
import torch
from safetensors.torch import save_file

from common.model_io import (
    CheckpointFormatError,
    FORMAT_VERSION,
    load_tensors_with_config,
    read_config_from_file,
    save_tensors_with_config,
)


def test_round_trip_preserves_tensors_and_config(tmp_path):
    path = str(tmp_path / "m.safetensors")
    tensors = {"a": torch.randn(3, 4), "b": torch.arange(5).float()}
    config = {"class": "Thing", "frame_size": 267, "normalization_mode": "zscore"}

    save_tensors_with_config(tensors, config, path)
    loaded, loaded_config = load_tensors_with_config(path)

    assert loaded_config == config
    assert set(loaded) == set(tensors)
    for name in tensors:
        assert torch.equal(loaded[name], tensors[name])


def test_read_config_does_not_require_reading_tensors(tmp_path):
    path = str(tmp_path / "m.safetensors")
    save_tensors_with_config({"a": torch.zeros(2)}, {"class": "Thing", "frame_skip": 3}, path)
    assert read_config_from_file(path) == {"class": "Thing", "frame_skip": 3}


def test_non_contiguous_and_shared_storage_tensors_are_accepted(tmp_path):
    # safetensors rejects shared storage and non-contiguous layouts, so the
    # writer must clone and make contiguous before handing tensors over.
    base = torch.randn(4, 4)
    path = str(tmp_path / "m.safetensors")
    tensors = {"view": base[:, :2], "transposed": base.t(), "whole": base}

    save_tensors_with_config(tensors, {"class": "Thing"}, path)
    loaded, _ = load_tensors_with_config(path)

    for name in tensors:
        assert torch.equal(loaded[name], tensors[name])


def test_file_without_config_metadata_is_rejected(tmp_path):
    path = str(tmp_path / "plain.safetensors")
    save_file({"a": torch.zeros(2)}, path)  # no metadata at all

    with pytest.raises(CheckpointFormatError, match="not a character-motion-vaes"):
        load_tensors_with_config(path)


def test_wrong_format_version_is_rejected(tmp_path):
    path = str(tmp_path / "future.safetensors")
    save_file(
        {"a": torch.zeros(2)},
        path,
        metadata={"config": json.dumps({"class": "Thing"}), "cmv_format_version": "99"},
    )

    with pytest.raises(CheckpointFormatError, match="format version"):
        load_tensors_with_config(path)


def test_writer_stamps_the_current_format_version(tmp_path):
    from safetensors import safe_open

    path = str(tmp_path / "m.safetensors")
    save_tensors_with_config({"a": torch.zeros(2)}, {"class": "Thing"}, path)
    with safe_open(path, framework="pt", device="cpu") as f:
        assert f.metadata()["cmv_format_version"] == FORMAT_VERSION
