# Safetensors Checkpoint Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every pickled `.pt` checkpoint in this repo with a `.safetensors` file, so no code path can execute arbitrary Python during model loading.

**Architecture:** A format layer in `common/model_io.py` writes `state_dict` tensors plus a JSON config string into the safetensors header's `__metadata__`. `vae_motion/models.py` gains a class registry plus `save_model`/`load_model`/`read_config`, and each model class exposes a `config` property and a `from_config` classmethod so architecture can be rebuilt from hyperparameters instead of resurrected from a pickle. The five `torch.load` call sites become one-line `load_model` calls.

**Tech Stack:** Python 3.10, PyTorch, safetensors, gym 0.23.1, pybullet, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-safetensors-migration-design.md`

## Global Constraints

- Python 3.10.19. The venv already exists at `../.venv310` (outside the repo). Interpreter: `/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe`.
- Proven dependency versions, to be used verbatim: `gym==0.23.1`, `torch==2.14.0`, `pybullet==3.2.7`, `numpy==1.26.4`, `safetensors==0.8.0`, `imageio==2.37.4`, `matplotlib==3.10.9`, `pytest==9.1.1`.
- `gym` must stay at `0.23.1`. It is the newest release that still provides **both** `registry.env_specs` (removed in 0.24) and the deprecated `rng.randint` shim (`mocap_envs.py:69`). Versions 0.21 and earlier are uninstallable — invalid `opencv-python>=3.` metadata.
- **No pickle-loading code may be committed to the repo.** The converter is the sole exception and is deleted in Task 9. Baseline-capture scripts that unpickle live outside the repo, untracked.
- `torch.load` must always be called with an explicit `weights_only=False` when reading legacy `.pt`, because torch 2.6+ defaults it to `True` and otherwise raises `UnpicklingError`.
- Format constants: metadata keys are `format` (`"pt"`), `cmv_format_version` (`"1"`), and `config` (a JSON string).
- Normalization statistics are named `data_max`, `data_min`, `data_avg`, `data_std` and **must be registered buffers** so they appear in `state_dict`. They are shape `[frame_size]`.
- Every commit must leave the working tree in a functional state. The `.safetensors` files are committed alongside the `.pt` files (Task 7) before the `.pt` files are removed (Task 8).
- `common/` must not import `vae_motion/` or `environments/`. Keep the format layer free of model-class imports.
- Never commit `__pycache__`. Task 1 adds the `.gitignore`.
- Tests must call `model.eval()` before any `forward`. `VectorQuantizer.forward` (`models.py:481-486`) uses the removed `add_(Number, Tensor)` overload in its training branch and raises on torch 2.x; `eval()` skips that branch.
- All work happens on branch `security/migrate-pt-to-safetensors`.

---

### Task 1: Make the repo installable again

Independent of the safetensors work. A reviewer can take this without the rest.

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.gitignore`
- Modify: `vae_motion/play_controller.py:18,65-66`
- Modify: `vae_motion/play_mvae.py:17,27-28`
- Modify: `vae_motion/train_controller.py:24-25,85`
- Test: `tests/test_env_registration.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a runnable environment. Later tasks rely on `gym.make("<EnvId>")` accepting a bare id, and on `pytest` being available.

Background: `gym.make("environments:TargetEnv-v0")` relied on pre-0.22 semantics where the prefix meant "import this module, then look up the id". From 0.22 the prefix is a **namespace**, and `environments/__init__.py` registers ids without one, so every call site raises `NameNotFound`. The fix is to import `environments` for its registration side effect and pass a bare id.

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:

```python
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
```

Create `tests/test_env_registration.py`:

```python
import gym
import pytest

import environments  # noqa: F401  (registers the env ids)

REGISTERED = [
    "RandomWalkEnv-v0",
    "TargetEnv-v0",
    "JoystickEnv-v0",
    "PathFollowEnv-v0",
    "HumanMazeEnv-v0",
]


@pytest.mark.parametrize("env_id", REGISTERED)
def test_env_id_resolves_without_namespace_prefix(env_id):
    # gym >= 0.22 treats "prefix:" as a namespace, and these ids are
    # registered without one, so the bare id must resolve.
    assert gym.spec(env_id) is not None


@pytest.mark.parametrize("env_id", REGISTERED)
def test_namespaced_id_is_not_registered(env_id):
    # Guards against reintroducing the "environments:" prefix.
    with pytest.raises(gym.error.Error):
        gym.spec("environments:{}".format(env_id))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /e/Dev/character-motion-vaes/character-motion-vaes
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_env_registration.py -v
```

Expected: collection succeeds, `test_env_id_resolves_without_namespace_prefix` PASSES (registration already works), and `test_namespaced_id_is_not_registered` PASSES. If both pass immediately that is correct — these tests lock in behaviour that the source does not yet rely on. Proceed; the real failure is exercised in Step 4.

- [ ] **Step 3: Fix the three `gym.make` call sites**

In `vae_motion/play_controller.py`, delete line 18 (`env_module = "environments"`) and add the registration import next to the other imports:

```python
import environments  # noqa: F401  (registers the env ids)
```

Change lines 65-66 from:

```python
    env = gym.make(
        "{}:{}".format(env_module, args.env),
```

to:

```python
    env = gym.make(
        args.env,
```

In `vae_motion/play_mvae.py`, delete line 17 (`env_module = "environments"`), add the same `import environments` line, and change lines 27-28 from `"{}:{}".format(env_module, args.env),` to `args.env,`.

In `vae_motion/train_controller.py`, add the same `import environments` line, and change lines 24-25 from `"{}:{}".format(args.env_module, args.env_name),` to `args.env_name,`. Leave `args.env_module` at line 85 in place; it is still used for logging.

- [ ] **Step 4: Verify no namespaced ids remain**

```bash
grep -rn 'format(env_module\|format(args.env_module\|environments:' --include=*.py . | grep -v '\.git/'
```

Expected: no output.

- [ ] **Step 5: Pin the dependencies**

Replace `requirements.txt` entirely with:

```
# Pinned to the newest versions verified to work with this codebase.
# gym must stay at 0.23.1: it is the last release providing both
# registry.env_specs (removed in 0.24) and the deprecated rng.randint
# shim used by environments/mocap_envs.py. Versions <= 0.21 cannot be
# installed at all (invalid opencv-python metadata).
gym==0.23.1
imageio==2.37.4
matplotlib==3.10.9
numpy==1.26.4
pybullet==3.2.7
safetensors==0.8.0
torch==2.14.0
```

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest==9.1.1
```

- [ ] **Step 6: Add `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.py[cod]
.venv/
.venv*/
dump/
outfile_*.png
```

- [ ] **Step 7: Run the tests**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_env_registration.py -v
```

Expected: 10 passed.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-dev.txt .gitignore tests/conftest.py \
        tests/test_env_registration.py vae_motion/play_controller.py \
        vae_motion/play_mvae.py vae_motion/train_controller.py
git commit -m "Pin dependencies and fix gym.make for gym >= 0.22

gym 0.22 changed 'module:id' from an import hint to a namespace lookup,
so every gym.make call site raised NameNotFound on any installable gym.
Import environments for its registration side effect and pass bare ids.

Pin to the newest versions verified against this codebase.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Capture baseline rollouts from the pickled checkpoints

**This task must complete before any change to `vae_motion/models.py`.** Once the model classes change, the original behaviour can no longer be measured.

**Files:**
- Create (untracked, outside the repo): `../_verify/capture_baseline.py`
- Produces (untracked): `../_verify/baseline_<EnvId>.npy`, `../_verify/shots_pt/*.png`

**Interfaces:**
- Consumes: Task 1's working environment.
- Produces: baseline pose arrays consumed by Task 10. Array shape is `(steps, 267)`, dtype `float32`, one row per frame, taken from `env.history[:, 0]` for character 0.

Nothing here is committed. This script unpickles, and per the Global Constraints no pickle-loading code enters the repo.

- [ ] **Step 1: Write the capture script**

Create `E:\Dev\character-motion-vaes\_verify\capture_baseline.py`:

```python
"""Capture reference rollouts from the legacy pickled checkpoints.

Untracked on purpose: this unpickles, and no pickle-loading code belongs
in the repo. Run before migrating vae_motion/models.py.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))          # .../character-motion-vaes/_verify
REPO = os.path.join(os.path.dirname(HERE), "character-motion-vaes")
sys.path.insert(0, REPO)

# torch 2.6+ defaults weights_only=True, which refuses whole-object pickles.
_orig_load = torch.load
def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load

import pybullet as pb
import environments.mocap_renderer as mocap_renderer
mocap_renderer.pb.GUI = pb.DIRECT          # render offscreen
import common.bullet_utils as bullet_utils
bullet_utils.Camera.wait = lambda self: None   # do not throttle to fps

import gym
import environments  # noqa: F401
from imageio import imwrite

OUT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(REPO, "vae_motion", "models")

# HumanPacmanEnv-v0 and TimedTargetEnv-v0 have checkpoints but are never
# registered in environments/__init__.py, so they cannot be rolled out.
CASES = [
    ("TargetEnv-v0", "con_TargetEnv-v0.pt"),
    ("JoystickEnv-v0", "con_JoystickEnv-v0.pt"),
    ("PathFollowEnv-v0", "con_PathFollowEnv-v0.pt"),
    ("HumanMazeEnv-v0", "con_HumanMazeEnv-v0.pt"),
]
STEPS = 60
SEED = 0


def rollout(env_id, controller_file, shots_dir=None):
    torch.manual_seed(SEED)
    env = gym.make(
        env_id,
        num_parallel=1,
        device="cpu",
        pose_vae_path=MODELS,
        rendered=shots_dir is not None,
        use_params=False,
        camera_tracking=False,
        frame_skip=1,
    )
    env.seed(SEED)
    torch.manual_seed(SEED)
    controller = torch.load(os.path.join(MODELS, controller_file), map_location="cpu").actor

    obs = env.reset()
    poses = []
    for step in range(STEPS):
        with torch.no_grad():
            action = controller(obs)
        obs, _reward, _done, _info = env.step(action)
        poses.append(env.history[:, 0].clone().cpu().numpy()[0])
        if shots_dir is not None and step % 20 == 19:
            capture(env, shots_dir, "%s_%04d" % (env_id, step))
    return np.asarray(poses, dtype=np.float32)


def capture(env, shots_dir, name):
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
    shots = os.path.join(OUT, "shots_pt")
    for env_id, controller_file in CASES:
        arr = rollout(env_id, controller_file, shots_dir=shots)
        path = os.path.join(OUT, "baseline_%s.npy" % env_id)
        np.save(path, arr)
        print("%-20s shape=%s checksum=%.6f -> %s" % (env_id, arr.shape, float(arr.sum()), path))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the checksums**

```bash
cd /e/Dev/character-motion-vaes/character-motion-vaes
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -W ignore ../_verify/capture_baseline.py
```

Expected: four lines, each `shape=(60, 267)` with a finite checksum, and four `.npy` files written. **Copy the printed checksums into the task notes** — Task 10 compares against them.

- [ ] **Step 3: Verify the baselines are reproducible**

Run the same command a second time and confirm the checksums are byte-identical to the first run. If they differ, seeding is incomplete and Task 10's comparison would be meaningless — stop and investigate before continuing.

- [ ] **Step 4: Confirm nothing was added to the repo**

```bash
git status --short
```

Expected: no new untracked files inside the repo.

Nothing to commit in this task.

---

### Task 3: Format layer — `common/model_io.py`

**Files:**
- Create: `common/model_io.py`
- Test: `tests/test_model_io_format.py`

**Interfaces:**
- Consumes: nothing.
- Produces, all used by Task 5:
  - `CheckpointFormatError(Exception)`
  - `FORMAT_VERSION: str = "1"`, `CONFIG_KEY: str = "config"`, `VERSION_KEY: str = "cmv_format_version"`
  - `save_tensors_with_config(tensors: dict[str, Tensor], config: dict, path: str) -> None`
  - `load_tensors_with_config(path: str, device="cpu") -> tuple[dict[str, Tensor], dict]`
  - `read_config_from_file(path: str) -> dict`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_io_format.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_model_io_format.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'common.model_io'`.

- [ ] **Step 3: Write the implementation**

Create `common/model_io.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_model_io_format.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add common/model_io.py tests/test_model_io_format.py
git commit -m "Add safetensors checkpoint format layer

Stores state_dict tensors plus a JSON config string in the safetensors
header metadata. Kept free of model imports so common/ keeps not
depending on vae_motion/.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Register normalization statistics as buffers

The bug that would make a naive migration silently destroy the models. Verified on the committed VAE: `data_max` exists on the object at shape `[267]` but `has data_max in state_dict: False`.

**Files:**
- Modify: `vae_motion/models.py` — add `NormalizationMixin`; change `AutoEncoder` (14-38), `PoseMixtureVAE` (205-237), `PoseMixtureSpecialistVAE` (271-315), `PoseVAE` (361-398), `PoseVQVAE` (505-544) to use it and delete their duplicated `normalize`/`denormalize` pairs.
- Test: `tests/test_normalization.py`

**Interfaces:**
- Consumes: nothing.
- Produces, used by Tasks 5 and 6:
  - `NormalizationMixin._init_normalization(self, normalization: dict) -> None` — sets `self.mode` and registers `data_max`/`data_min`/`data_avg`/`data_std` buffers.
  - `NormalizationMixin.normalize(self, t) -> Tensor`, `.denormalize(self, t) -> Tensor` — behaviour unchanged from the five deleted copies.
  - `NormalizationMixin.normalization_from_tensors(config: dict, tensors: dict) -> dict` (staticmethod) — rebuilds the `normalization` constructor argument.

- [ ] **Step 1: Write the failing test**

Create `tests/test_normalization.py`:

```python
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
    torch.manual_seed(0)
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_normalization.py -v
```

Expected: `test_normalization_stats_appear_in_state_dict` FAILS with `data_max missing from ... state_dict` for all five classes, and `test_normalization_from_tensors_rebuilds_constructor_argument` FAILS with `AttributeError: normalization_from_tensors`.

- [ ] **Step 3: Add the mixin**

In `vae_motion/models.py`, insert immediately after the imports (after line 11):

```python
class NormalizationMixin:
    """Shared normalization state for the VAE models.

    The statistics are registered as buffers so they appear in state_dict and
    follow .to(device). They used to be plain attributes, which meant a
    state_dict-based save silently dropped them.
    """

    STAT_KEYS = ("max", "min", "avg", "std")

    def _init_normalization(self, normalization):
        self.mode = normalization.get("mode")
        for key in self.STAT_KEYS:
            value = normalization.get(key)
            if value is not None:
                value = torch.as_tensor(value).float()
            self.register_buffer("data_" + key, value)

    @staticmethod
    def normalization_from_tensors(config, tensors):
        """Rebuild the `normalization` constructor argument from a checkpoint."""
        normalization = {"mode": config["normalization_mode"]}
        for key in NormalizationMixin.STAT_KEYS:
            name = "data_" + key
            if name in tensors:
                normalization[key] = tensors[name]
        return normalization

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")
```

- [ ] **Step 4: Convert the five classes**

For each of `AutoEncoder`, `PoseMixtureVAE`, `PoseMixtureSpecialistVAE`, `PoseVAE`, `PoseVQVAE`:

1. Change the class declaration to put the mixin first, e.g.
   `class PoseMixtureVAE(NormalizationMixin, nn.Module):`
2. Replace these five lines in `__init__`:

```python
        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")
```

   with:

```python
        self._init_normalization(normalization)
```

   `_init_normalization` calls `register_buffer`, so it must come **after** `super().__init__()`.
3. Delete that class's `normalize` and `denormalize` methods entirely — the mixin supplies them.

Leave every other method untouched.

- [ ] **Step 5: Run the test to verify it passes**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_normalization.py -v
```

Expected: all passed (55 tests).

- [ ] **Step 6: Confirm the duplication is gone**

```bash
grep -c "Unknown normalization mode" vae_motion/models.py
```

Expected: `2` (one in `normalize`, one in `denormalize`, both in the mixin). It was 10 before.

- [ ] **Step 7: Commit**

```bash
git add vae_motion/models.py tests/test_normalization.py
git commit -m "Register normalization statistics as buffers

data_max/min/avg/std were plain attributes, so they never appeared in
state_dict and only survived because the whole module was pickled. Any
state_dict-based save dropped them silently, leaving normalize() and
denormalize() to produce wrong output with no error.

Extract the five identical copies of this logic into NormalizationMixin
so the fix lives in one place.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Model config, registry, and save/load for the VAE classes

**Files:**
- Modify: `vae_motion/models.py` — add `config` property and `from_config` classmethod to the five VAE classes; add registry and public API at end of file.
- Test: `tests/test_checkpoint_round_trip.py`

**Interfaces:**
- Consumes: Task 3's `common.model_io`; Task 4's `NormalizationMixin.normalization_from_tensors`.
- Produces, used by Tasks 6, 7, 8:
  - `MODEL_REGISTRY: dict[str, type]`
  - `save_model(model, path: str, extra_config: dict | None = None) -> None`
  - `load_model(path: str, device="cpu") -> nn.Module`
  - `read_config(path: str) -> dict`
  - `<Class>.config` property returning constructor arguments plus `normalization_mode`
  - `<Class>.from_config(config: dict, tensors: dict) -> nn.Module` classmethod

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint_round_trip.py`:

```python
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
    model = build(PoseVAE)
    path = str(tmp_path / "m.safetensors")
    save_model(model, path)

    # Drop a required tensor; strict loading must complain rather than
    # silently produce a partly-initialised model.
    from common.model_io import load_tensors_with_config, save_tensors_with_config

    tensors, config = load_tensors_with_config(path)
    del tensors["fc1.weight"]
    save_tensors_with_config(tensors, config, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_checkpoint_round_trip.py -v
```

Expected: collection error, `ImportError: cannot import name 'load_model' from 'vae_motion.models'`.

- [ ] **Step 3: Add `config` and `from_config` to the five VAE classes**

Add to `AutoEncoder`:

```python
    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            cls.normalization_from_tensors(config, tensors),
        )
```

Add to `PoseVAE`:

```python
    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
        )
```

Add to `PoseMixtureVAE` (`num_experts` is not stored as an attribute, so read it off the first expert weight, which has shape `(num_experts, input_size, hidden_size)`):

```python
    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "num_experts": int(self.decoder.w0.shape[0]),
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
            config["num_experts"],
        )
```

Add to `PoseMixtureSpecialistVAE` (identical apart from how `num_experts` is derived):

```python
    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "num_experts": len(self.decoders),
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
            config["num_experts"],
        )
```

Add to `PoseVQVAE` (note `num_embeddings` is the third constructor argument):

```python
    @property
    def config(self):
        return {
            "frame_size": self.frame_size,
            "latent_size": self.latent_size,
            "num_embeddings": self.quantizer.num_embeddings,
            "num_condition_frames": self.num_condition_frames,
            "num_future_predictions": self.num_future_predictions,
            "normalization_mode": self.mode,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(
            config["frame_size"],
            config["latent_size"],
            config["num_embeddings"],
            config["num_condition_frames"],
            config["num_future_predictions"],
            cls.normalization_from_tensors(config, tensors),
        )
```

- [ ] **Step 4: Add the registry and public API**

Add the import near the top of `vae_motion/models.py`, beside `from common.controller import init, DiagGaussian`:

```python
from common.model_io import (
    CheckpointFormatError,
    load_tensors_with_config,
    read_config_from_file,
    save_tensors_with_config,
)
```

Append to the end of `vae_motion/models.py`:

```python
MODEL_REGISTRY = {
    "AutoEncoder": AutoEncoder,
    "PoseVAE": PoseVAE,
    "PoseMixtureVAE": PoseMixtureVAE,
    "PoseMixtureSpecialistVAE": PoseMixtureSpecialistVAE,
    "PoseVQVAE": PoseVQVAE,
}


def save_model(model, path, extra_config=None):
    """Write `model` to `path` as a safetensors checkpoint."""
    config = dict(model.config)
    config["class"] = type(model).__name__
    if extra_config:
        config.update(extra_config)
    save_tensors_with_config(model.state_dict(), config, path)


def load_model(path, device="cpu"):
    """Rebuild the model described by the checkpoint at `path`."""
    tensors, config = load_tensors_with_config(path, device)
    class_name = config.get("class")
    model_class = MODEL_REGISTRY.get(class_name)
    if model_class is None:
        raise CheckpointFormatError(
            "{} declares unknown model class {!r}; known classes are {}".format(
                path, class_name, sorted(MODEL_REGISTRY)
            )
        )
    # Building from config first means the normalization buffers already exist
    # at the right shape, so the strict load below is a no-op for them.
    model = model_class.from_config(config, tensors)
    model.load_state_dict(tensors, strict=True)
    return model.to(device)


def read_config(path):
    """Return a checkpoint's config without reading its tensors."""
    return read_config_from_file(path)
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_checkpoint_round_trip.py -v
```

Expected: all passed (26 tests).

- [ ] **Step 6: Run the whole suite for regressions**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -q
```

Expected: all passed.

- [ ] **Step 7: Commit**

```bash
git add vae_motion/models.py tests/test_checkpoint_round_trip.py
git commit -m "Add safetensors save/load for the VAE model classes

Each class exposes its constructor arguments as a config dict and can be
rebuilt from one, so architecture comes from hyperparameters rather than
from a pickled object graph.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Controller and policy — drop the `env` dependency

**Files:**
- Modify: `vae_motion/models.py:600-639` (`PoseVAEController`) and `642-703` (`PoseVAEPolicy`); extend `MODEL_REGISTRY`.
- Modify: `vae_motion/train_controller.py:141`
- Test: `tests/test_controller_checkpoint.py`

**Interfaces:**
- Consumes: Task 5's `save_model`, `load_model`, `read_config`, `MODEL_REGISTRY`.
- Produces:
  - `PoseVAEController(observation_dim: int, action_dim: int)` — replaces `PoseVAEController(env)`
  - `PoseVAEController.config` -> `{"observation_dim": int, "action_dim": int}`
  - `PoseVAEPolicy.config` -> `{"observation_dim": int, "action_dim": int}`
  - both classes registered in `MODEL_REGISTRY`

`play_controller.py:58` loads the controller before any env exists, so the constructor cannot depend on a live env. `frame_skip` travels in the config via `extra_config`, replacing the `env_info` attribute that `train_controller.py:146` used to staple onto the model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_controller_checkpoint.py`:

```python
import pytest
import torch

from vae_motion.models import (
    PoseVAEController,
    PoseVAEPolicy,
    load_model,
    read_config,
    save_model,
)

OBS = 6
ACT = 3


def build_controller():
    torch.manual_seed(0)
    return PoseVAEController(OBS, ACT)


def build_policy():
    torch.manual_seed(0)
    return PoseVAEPolicy(PoseVAEController(OBS, ACT))


def test_controller_is_constructed_from_dimensions_not_an_env():
    controller = build_controller()
    assert controller.observation_dim == OBS
    assert controller.action_dim == ACT


def test_controller_round_trip_preserves_output(tmp_path):
    controller = build_controller()
    path = str(tmp_path / "c.safetensors")
    save_model(controller, path)

    reloaded = load_model(path)
    reloaded.eval()
    controller.eval()

    obs = torch.randn(2, OBS)
    with torch.no_grad():
        assert torch.equal(controller(obs), reloaded(obs))


def test_policy_round_trip_preserves_actor_and_critic(tmp_path):
    policy = build_policy()
    path = str(tmp_path / "p.safetensors")
    save_model(policy, path)

    reloaded = load_model(path)
    policy.eval()
    reloaded.eval()

    obs = torch.randn(2, OBS)
    with torch.no_grad():
        assert torch.equal(policy.actor(obs), reloaded.actor(obs))
        assert torch.equal(policy.critic(obs), reloaded.critic(obs))


def test_policy_round_trip_preserves_all_tensors(tmp_path):
    policy = build_policy()
    path = str(tmp_path / "p.safetensors")
    save_model(policy, path)
    reloaded = load_model(path)

    original = policy.state_dict()
    restored = reloaded.state_dict()
    assert set(original) == set(restored)
    # dist.logstd._bias is a parameter too and must survive.
    assert "dist.logstd._bias" in original
    for name in original:
        assert torch.equal(original[name], restored[name]), name


def test_policy_config_records_dimensions(tmp_path):
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path)
    config = read_config(path)
    assert config["class"] == "PoseVAEPolicy"
    assert config["observation_dim"] == OBS
    assert config["action_dim"] == ACT


def test_frame_skip_travels_in_the_config(tmp_path):
    # Replaces the old `actor_critic.env_info = {...}` attribute hack.
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path, extra_config={"frame_skip": 2})
    assert read_config(path)["frame_skip"] == 2


def test_frame_skip_absent_when_not_supplied(tmp_path):
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path)
    assert read_config(path).get("frame_skip", 1) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_controller_checkpoint.py -v
```

Expected: FAIL. `PoseVAEController(OBS, ACT)` raises `TypeError` — the current signature takes a single `env`.

- [ ] **Step 3: Change `PoseVAEController`**

In `vae_motion/models.py`, replace lines 601-605:

```python
    def __init__(self, env):
        super().__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
```

with:

```python
    def __init__(self, observation_dim, action_dim):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
```

Add to the same class:

```python
    @property
    def config(self):
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(config["observation_dim"], config["action_dim"])
```

- [ ] **Step 4: Add config to `PoseVAEPolicy`**

```python
    @property
    def config(self):
        return {
            "observation_dim": self.actor.observation_dim,
            "action_dim": self.actor.action_dim,
        }

    @classmethod
    def from_config(cls, config, tensors):
        return cls(PoseVAEController(config["observation_dim"], config["action_dim"]))
```

- [ ] **Step 5: Extend the registry**

Add two entries to `MODEL_REGISTRY`:

```python
    "PoseVAEController": PoseVAEController,
    "PoseVAEPolicy": PoseVAEPolicy,
```

- [ ] **Step 6: Update the one construction site**

In `vae_motion/train_controller.py:141`, change:

```python
        controller = PoseVAEController(env)
```

to:

```python
        controller = PoseVAEController(args.observation_size, args.action_size)
```

`args.observation_size` and `args.action_size` are already set at lines 104-105.

- [ ] **Step 7: Verify no other construction sites exist**

```bash
grep -rn "PoseVAEController(" --include=*.py . | grep -v '\.git/'
```

Expected: only the definition in `models.py`, the `from_config` bodies, `train_controller.py:141`, and the tests.

- [ ] **Step 8: Run the tests**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -q
```

Expected: all passed.

- [ ] **Step 9: Commit**

```bash
git add vae_motion/models.py vae_motion/train_controller.py tests/test_controller_checkpoint.py
git commit -m "Build PoseVAEController from dimensions instead of an env

play_controller loads the controller before any env exists, so taking a
live env in the constructor made rebuilding from a checkpoint impossible.
frame_skip now travels in the checkpoint config, replacing the env_info
attribute that was stapled onto the model after loading.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Convert the committed checkpoints

Both formats coexist after this task so the tree stays functional. Task 8 removes the `.pt` files.

**Files:**
- Create: `tools/convert_checkpoints.py`
- Create: `vae_motion/models/*.safetensors` (7 files)
- Test: `tests/test_converted_checkpoints.py`

**Interfaces:**
- Consumes: Task 5's and Task 6's `save_model`/`load_model`/`read_config`.
- Produces: the 7 converted checkpoints, consumed by Tasks 8 and 10.

This is the only file in the repo permitted to unpickle, and it is deleted in Task 9. Unpickling an `nn.Module` restores `__dict__` directly without calling `__init__`, so Task 6's signature change does not prevent reading the old files.

Expected values, already measured from the committed files:

| file | class | config |
|---|---|---|
| `posevae_c1_e6_l32` | `PoseMixtureVAE` | `frame_size=267, latent_size=32, num_condition_frames=1, num_future_predictions=1, num_experts=6, normalization_mode=zscore` |
| `con_TargetEnv-v0` | `PoseVAEPolicy` | `observation_dim=269, action_dim=32, frame_skip=1` |
| `con_JoystickEnv-v0` | `PoseVAEPolicy` | `observation_dim=269, action_dim=32, frame_skip=1` |
| `con_TimedTargetEnv-v0` | `PoseVAEPolicy` | `observation_dim=270, action_dim=32, frame_skip=1` |
| `con_PathFollowEnv-v0` | `PoseVAEPolicy` | `observation_dim=275, action_dim=32, frame_skip=1` |
| `con_HumanMazeEnv-v0` | `PoseVAEPolicy` | `observation_dim=283, action_dim=2, frame_skip=1` |
| `con_HumanPacmanEnv-v0` | `PoseVAEPolicy` | `observation_dim=283, action_dim=2, frame_skip=1` |

- [ ] **Step 1: Write the converter**

Create `tools/convert_checkpoints.py`:

```python
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
    PoseMixtureSpecialistVAE,
    PoseVAEPolicy,
    PoseVQVAE,
    load_model,
    save_model,
)


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
        latent_size = config["latent_size"]
        condition = torch.randn(4, frame_size * config["num_condition_frames"])
        if isinstance(model, PoseVQVAE):
            logits = torch.randn(4, config["num_embeddings"])
            return model.sample(logits, condition, deterministic=True)
        latent = torch.randn(4, latent_size)
        if isinstance(model, PoseMixtureSpecialistVAE):
            return model.sample(latent, condition, deterministic=True)
        return model.sample(latent, condition)


def convert(pt_path, verify):
    out_path = os.path.splitext(pt_path)[0] + ".safetensors"

    # weights_only=False is required: torch 2.6+ defaults it to True, which
    # refuses whole-object pickles outright.
    original = torch.load(pt_path, map_location="cpu", weights_only=False)
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

    if not torch.equal(deterministic_probe(original, config),
                       deterministic_probe(reloaded, config)):
        print("FAIL %s: forward output differs" % pt_path)
        return False

    print("OK   %s -> %s  (%d tensors, output identical)"
          % (pt_path, os.path.basename(out_path), len(before)))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="A .pt file, or a directory of them")
    parser.add_argument("--verify", action="store_true",
                        help="Assert the converted file reproduces the original exactly")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        targets = sorted(glob.glob(os.path.join(args.path, "*.pt")))
    else:
        targets = [args.path]
    if not targets:
        print("no .pt files found at %s" % args.path)
        return 1

    ok = all(convert(target, args.verify) for target in targets)
    print("\n%d file(s), %s" % (len(targets), "all verified" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

`PoseVAE` and `PoseMixtureVAE` need no import: `deterministic_probe` handles both through its final `return model.sample(latent, condition)`.

- [ ] **Step 2: Run the conversion with verification**

```bash
cd /e/Dev/character-motion-vaes/character-motion-vaes
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -W ignore \
    tools/convert_checkpoints.py vae_motion/models --verify
```

Expected: seven `OK` lines and `7 file(s), all verified`. Any `FAIL` line stops the task — investigate before proceeding.

- [ ] **Step 3: Confirm the configs match the table above**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -c "
import glob, json, sys
sys.path.insert(0, '.')
from vae_motion.models import read_config
for f in sorted(glob.glob('vae_motion/models/*.safetensors')):
    print('%-44s %s' % (f.split('/')[-1], json.dumps(read_config(f), sort_keys=True)))
"
```

Expected: seven lines matching the table. Every controller must show `frame_skip: 1`.

- [ ] **Step 4: Write the checkpoint test**

Create `tests/test_converted_checkpoints.py`:

```python
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
        "class": "PoseMixtureVAE", "frame_size": 267, "latent_size": 32,
        "num_condition_frames": 1, "num_future_predictions": 1,
        "num_experts": 6, "normalization_mode": "zscore",
    },
    "con_TargetEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 269, "action_dim": 32, "frame_skip": 1},
    "con_JoystickEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 269, "action_dim": 32, "frame_skip": 1},
    "con_TimedTargetEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 270, "action_dim": 32, "frame_skip": 1},
    "con_PathFollowEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 275, "action_dim": 32, "frame_skip": 1},
    "con_HumanMazeEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 283, "action_dim": 2, "frame_skip": 1},
    "con_HumanPacmanEnv-v0": {"class": "PoseVAEPolicy", "observation_dim": 283, "action_dim": 2, "frame_skip": 1},
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
    # Task 4, so a dropped statistic would silently corrupt every rollout.
    model = load_model(os.path.join(MODELS_DIR, "posevae_c1_e6_l32.safetensors"))
    for name in ("data_max", "data_min", "data_avg", "data_std"):
        stat = getattr(model, name)
        assert stat is not None, name
        assert stat.shape == (267,), name
        assert torch.isfinite(stat).all(), name
    assert (model.data_std > 0).all()
    assert (model.data_max >= model.data_min).all()
```

- [ ] **Step 5: Run the tests**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -q
```

Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add tools/convert_checkpoints.py vae_motion/models/*.safetensors \
        tests/test_converted_checkpoints.py
git commit -m "Convert pretrained checkpoints to safetensors

All seven verified: identical state_dict tensors and bit-identical
forward output versus the pickled originals. The .pt files stay in place
until the loaders are migrated, so the tree keeps working.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Migrate the loaders and delete the `.pt` files

**Files:**
- Modify: `vae_motion/train_mvae.py:196-208,213,319`
- Modify: `vae_motion/train_controller.py:103,139,146,224`
- Modify: `vae_motion/play_controller.py:30-31,58-62`
- Modify: `environments/mocap_envs.py:118,122,996-997`
- Delete: `vae_motion/models/*.pt` (7 files)
- Test: `tests/test_no_pickle_loading.py`

**Interfaces:**
- Consumes: Task 5's and Task 6's `load_model`/`save_model`/`read_config`; Task 7's converted checkpoints.
- Produces: a repo with no pickle-loading call sites.

- [ ] **Step 1: Write the failing test**

Create `tests/test_no_pickle_loading.py`:

```python
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# tools/convert_checkpoints.py must unpickle by definition; it is deleted
# once the committed checkpoints are converted.
ALLOWED = {os.path.join("tools", "convert_checkpoints.py")}


def python_sources():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "docs"}]
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                yield os.path.relpath(path, REPO_ROOT), path


def test_no_source_file_calls_torch_load():
    offenders = []
    for relative, path in python_sources():
        if relative.replace("\\", "/") in {a.replace("\\", "/") for a in ALLOWED}:
            continue
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if re.search(r"\btorch\.load\s*\(", line):
                    offenders.append("%s:%d" % (relative, number))
    assert offenders == [], "torch.load found at: {}".format(offenders)


def test_no_pt_checkpoints_remain():
    models_dir = os.path.join(REPO_ROOT, "vae_motion", "models")
    leftovers = [n for n in os.listdir(models_dir) if n.endswith(".pt")]
    assert leftovers == []


def test_no_source_file_globs_for_pt_files():
    offenders = []
    for relative, path in python_sources():
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if re.search(r"""["'][^"']*\*\.pt["']""", line):
                    offenders.append("%s:%d" % (relative, number))
    assert offenders == [], "glob for *.pt found at: {}".format(offenders)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_no_pickle_loading.py -v
```

Expected: all three FAIL — `torch.load` is present at five sites, seven `.pt` files remain, and two `*.pt` globs exist.

- [ ] **Step 3: Migrate `vae_motion/train_mvae.py`**

Change the four filename templates at lines 196-208 from `.pt"` to `.safetensors"`, so they read `"posevae_c{}_l{}.safetensors"`, `"posevae_c{}_e{}_l{}.safetensors"`, `"posevae_c{}_s{}_l{}.safetensors"`, `"posevae_c{}_n{}_l{}.safetensors"`.

Replace line 213:

```python
        pose_vae = torch.load(pose_vae_path, map_location=args.device)
```

with:

```python
        pose_vae = load_model(pose_vae_path, args.device)
```

Replace line 319:

```python
        torch.save(copy.deepcopy(pose_vae).cpu(), pose_vae_path)
```

with:

```python
        save_model(pose_vae, pose_vae_path)
```

`save_model` already moves tensors to CPU, so the `deepcopy` is redundant. Update the import of `vae_motion.models` in this file to include `load_model` and `save_model`, and remove `import copy` if nothing else in the file uses it (check with `grep -n "copy\." vae_motion/train_mvae.py`).

- [ ] **Step 4: Migrate `vae_motion/train_controller.py`**

Line 103: change `"con_" + args.env_name + ".pt"` to `"con_" + args.env_name + ".safetensors"`.

Line 139: replace

```python
        actor_critic = torch.load(args.save_path, map_location=args.device)
```

with

```python
        actor_critic = load_model(args.save_path, args.device)
```

Line 146: delete

```python
    actor_critic.env_info = {"frame_skip": args.frame_skip}
```

Line 224: replace

```python
        torch.save(copy.deepcopy(actor_critic).cpu(), args.save_path)
```

with

```python
        save_model(actor_critic, args.save_path, {"frame_skip": args.frame_skip})
```

Extend the existing import to `from vae_motion.models import PoseVAEController, PoseVAEPolicy, load_model, save_model`, and remove `import copy` if now unused.

- [ ] **Step 5: Migrate `vae_motion/play_controller.py`**

Lines 30-31: change the globs to

```python
        candidate_controller_paths = glob(base_dir + "/con*.safetensors")
        candidate_pose_vae_paths = glob(base_dir + "/posevae*.safetensors")
```

Lines 58-62: replace

```python
    actor_critic = torch.load(controller_path, map_location=device)
    if hasattr(actor_critic, "env_info"):
        frame_skip = actor_critic.env_info["frame_skip"]
    else:
        frame_skip = 1
    controller = actor_critic.actor
```

with

```python
    actor_critic = load_model(controller_path, device)
    frame_skip = read_config(controller_path).get("frame_skip", 1)
    controller = actor_critic.actor
```

Add the import:

```python
from vae_motion.models import load_model, read_config
```

- [ ] **Step 6: Migrate `environments/mocap_envs.py`**

Line 118: change

```python
            pose_vae_path = glob.glob(os.path.join(basepath, "posevae*.pt"))[0]
```

to

```python
            pose_vae_path = glob.glob(os.path.join(basepath, "posevae*.safetensors"))[0]
```

Line 122: replace

```python
        self.pose_vae_model = torch.load(pose_vae_path, map_location=self.device)
```

with

```python
        self.pose_vae_model = load_model(pose_vae_path, self.device)
```

Lines 996-997: replace

```python
        policy_path = os.path.join(basepath(pose_vae_path), "con_TargetEnv-v0.pt")
        self.target_controller = torch.load(policy_path, map_location=self.device).actor
```

with

```python
        policy_path = os.path.join(
            basepath(pose_vae_path), "con_TargetEnv-v0.safetensors"
        )
        self.target_controller = load_model(policy_path, self.device).actor
```

Add to the import block (after `from common.misc_utils import line_to_point_distance`):

```python
from vae_motion.models import load_model
```

`vae_motion.models` imports only `common.*`, so this does not create a cycle.

- [ ] **Step 7: Delete the `.pt` files**

```bash
git rm vae_motion/models/*.pt
```

Expected: seven deletions staged.

- [ ] **Step 8: Run the tests**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -q
```

Expected: all passed, including the three tests from Step 1.

- [ ] **Step 9: Smoke-test a real rollout**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -W ignore -c "
import sys; sys.path.insert(0, '.')
import torch, gym, environments
env = gym.make('TargetEnv-v0', num_parallel=1, device='cpu',
               pose_vae_path='vae_motion/models', rendered=False,
               use_params=False, camera_tracking=False, frame_skip=1)
env.seed(0); torch.manual_seed(0)
from vae_motion.models import load_model
controller = load_model('vae_motion/models/con_TargetEnv-v0.safetensors').actor
obs = env.reset()
for _ in range(10):
    with torch.no_grad(): action = controller(obs)
    obs, r, d, i = env.step(action)
print('rollout OK, obs', tuple(obs.shape))
"
```

Expected: `rollout OK, obs (1, 269)`. This is the first end-to-end proof that no pickle is involved.

- [ ] **Step 10: Commit**

```bash
git add vae_motion/train_mvae.py vae_motion/train_controller.py \
        vae_motion/play_controller.py environments/mocap_envs.py \
        tests/test_no_pickle_loading.py
git commit -m "Load checkpoints via safetensors and drop the .pt files

Replaces all five torch.load call sites. Pickled checkpoints could
execute arbitrary code on load, and since torch 2.6 flipped
weights_only to True they could not be loaded at all.

Adds a test asserting no source file calls torch.load or globs for *.pt.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Remove the converter

**Files:**
- Delete: `tools/convert_checkpoints.py`
- Modify: `tests/test_no_pickle_loading.py` — drop the `ALLOWED` exemption

Once the committed checkpoints are converted, the converter is the last unpickling code in the repo. Outside users with their own `.pt` models can recover it from history; the README (Task 11) tells them how.

- [ ] **Step 1: Tighten the test**

In `tests/test_no_pickle_loading.py`, replace

```python
# tools/convert_checkpoints.py must unpickle by definition; it is deleted
# once the committed checkpoints are converted.
ALLOWED = {os.path.join("tools", "convert_checkpoints.py")}
```

with

```python
# No file may unpickle a checkpoint. The converter used for the one-time
# migration lives in git history only; see README for how to retrieve it.
ALLOWED = set()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/test_no_pickle_loading.py -v
```

Expected: `test_no_source_file_calls_torch_load` FAILS, reporting `tools/convert_checkpoints.py`.

- [ ] **Step 3: Record the converter's commit, then delete it**

```bash
git log --oneline -1 -- tools/convert_checkpoints.py
git rm tools/convert_checkpoints.py
rmdir tools 2>/dev/null || true
```

Note the printed commit hash — the README references it in Task 11.

- [ ] **Step 4: Run the tests**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -q
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_no_pickle_loading.py
git commit -m "Remove the one-time checkpoint converter

No code in the repo unpickles any more. Users migrating their own .pt
checkpoints can recover the converter from git history.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end equivalence and screenshots

Proves the migration changed nothing observable about the models.

**Files:**
- Create: `tools/rollout_trace.py`
- Create (untracked): `../_verify/compare.py`, `../_verify/shots_safetensors/*.png`

**Interfaces:**
- Consumes: Task 2's `../_verify/baseline_<EnvId>.npy`; Task 8's migrated loaders.
- Produces: a committed reproducible rollout tool and the before/after comparison result.

The rollout must match Task 2's parameters exactly: `STEPS = 60`, `SEED = 0`, `num_parallel=1`, `device="cpu"`, `frame_skip=1`, `camera_tracking=False`, and the pose taken from `env.history[:, 0]` for character 0.

- [ ] **Step 1: Write the committed trace tool**

Create `tools/rollout_trace.py`:

```python
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


def build_env(env_id, models_dir, seed, rendered, frame_skip):
    import gym
    import environments  # noqa: F401  (registers the env ids)

    torch.manual_seed(seed)
    env = gym.make(
        env_id,
        num_parallel=1,
        device="cpu",
        pose_vae_path=models_dir,
        rendered=rendered,
        use_params=False,
        camera_tracking=False,
        frame_skip=frame_skip,
    )
    env.seed(seed)
    torch.manual_seed(seed)
    return env


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
    parser.add_argument("--env", required=True, help="Registered env id, e.g. TargetEnv-v0")
    parser.add_argument("--controller", required=True, help="Path to a .safetensors controller")
    parser.add_argument("--models", default=None, help="Directory holding the VAE checkpoint")
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
    env = build_env(args.env, models_dir, args.seed, args.shots is not None, args.frame_skip)
    controller = load_model(args.controller, "cpu").actor

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
    print("%s shape=%s checksum=%.6f -> %s"
          % (args.env, trace.shape, float(trace.sum()), args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record post-migration traces**

```bash
cd /e/Dev/character-motion-vaes/character-motion-vaes
PY=/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe
for env in TargetEnv-v0 JoystickEnv-v0 PathFollowEnv-v0 HumanMazeEnv-v0; do
  $PY -W ignore tools/rollout_trace.py --env $env \
      --controller vae_motion/models/con_$env.safetensors \
      --out ../_verify/after_$env.npy --shots ../_verify/shots_safetensors
done
```

Expected: four lines whose checksums are identical to the Task 2 baselines.

- [ ] **Step 3: Write the comparison script**

Create `E:\Dev\character-motion-vaes\_verify\compare.py`:

```python
"""Assert post-migration rollouts match the pre-migration baselines exactly."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

failures = []
for baseline_path in sorted(glob.glob(os.path.join(HERE, "baseline_*.npy"))):
    env_id = os.path.basename(baseline_path)[len("baseline_"):-len(".npy")]
    after_path = os.path.join(HERE, "after_%s.npy" % env_id)
    if not os.path.exists(after_path):
        failures.append("%s: no post-migration trace" % env_id)
        continue

    before, after = np.load(baseline_path), np.load(after_path)
    if before.shape != after.shape:
        failures.append("%s: shape %s vs %s" % (env_id, before.shape, after.shape))
    elif not np.array_equal(before, after):
        max_diff = float(np.abs(before - after).max())
        first_bad = int(np.argmax(np.abs(before - after).max(axis=1) > 0))
        failures.append("%s: differs, max=%g, first bad frame=%d"
                        % (env_id, max_diff, first_bad))
    else:
        print("IDENTICAL %-20s %s frames" % (env_id, before.shape[0]))

print()
if failures:
    for line in failures:
        print("FAIL", line)
    sys.exit(1)
print("All rollouts bit-identical before and after migration.")
```

- [ ] **Step 4: Run the comparison**

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe ../_verify/compare.py
```

Expected: four `IDENTICAL` lines and `All rollouts bit-identical before and after migration.`

If any trajectory differs, the migration changed model behaviour. Diagnose before continuing — the most likely cause is a dropped or mis-shaped normalization buffer. Do not proceed on a `FAIL`.

- [ ] **Step 5: Compare the screenshots visually**

Inspect matching pairs in `../_verify/shots_pt/` and `../_verify/shots_safetensors/` (for example `TargetEnv-v0_0039.png` in each). They should be indistinguishable. Build a side-by-side sheet for the report:

```bash
/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -c "
import glob, os, numpy as np
from imageio import imread, imwrite
out = '../_verify'
for before in sorted(glob.glob(os.path.join(out, 'shots_pt', '*.png'))):
    after = before.replace('shots_pt', 'shots_safetensors')
    if not os.path.exists(after): continue
    a, b = imread(before), imread(after)
    pad = np.full((a.shape[0], 8, 3), 255, dtype=np.uint8)
    name = os.path.basename(before)
    imwrite(os.path.join(out, 'sheet_' + name), np.concatenate([a, pad, b], axis=1))
    print('identical pixels:', np.array_equal(a, b), name)
"
```

Expected: `identical pixels: True` for every pair.

- [ ] **Step 6: Commit the tool**

```bash
git add tools/rollout_trace.py
git commit -m "Add deterministic rollout tracing tool

Records a seeded rollout as a pose trajectory so behaviour changes can be
detected exactly. Used to verify the safetensors migration left all six
controllers bit-identical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Document the change

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the model paths in the Quick Start**

In `README.md`, change the pretrained-model command from

```bash
python play_mvae.py --vae models/posevae_c1_e6_l32.pt
```

to

```bash
python play_mvae.py --vae models/posevae_c1_e6_l32.safetensors
```

Also change `pip install -r requirements` to `pip install -r requirements.txt` — the existing text is missing the extension.

- [ ] **Step 2: Add a checkpoint format section**

Insert after the "Run Pretrained Models" section, replacing `<HASH>` with the hash recorded in Task 9 Step 3:

```markdown
## Checkpoint Format

Checkpoints are [safetensors](https://github.com/huggingface/safetensors) files.
Each one holds the model's tensors plus its architecture hyperparameters, stored
as a JSON string in the file's metadata, so a model can be rebuilt from the file
alone:

```python
from vae_motion.models import load_model, read_config

print(read_config("models/posevae_c1_e6_l32.safetensors"))
model = load_model("models/posevae_c1_e6_l32.safetensors", device="cpu")
```

Earlier versions of this repo stored whole pickled `nn.Module` objects in `.pt`
files. Loading such a file executes arbitrary Python, and since PyTorch 2.6
changed `weights_only` to default to `True`, those files can no longer be loaded
by an unmodified PyTorch at all.

### Migrating your own `.pt` checkpoints

`.pt` files are no longer loadable. To convert checkpoints you trained yourself,
retrieve the one-time converter from git history and run it:

```bash
git show <HASH>:tools/convert_checkpoints.py > convert_checkpoints.py
python convert_checkpoints.py path/to/your/models --verify
```

`--verify` checks that each converted file reproduces its original's tensors and
forward output exactly. The converter unpickles, so only run it on checkpoints
you trust. Delete it afterwards.

### Breaking changes

- `.pt` checkpoints are no longer loaded anywhere. Convert them as above.
- `PoseVAEController(env)` is now `PoseVAEController(observation_dim, action_dim)`.
  Controllers are loaded before an environment exists, so the constructor can no
  longer read dimensions off a live env.
- Normalization statistics (`data_max`, `data_min`, `data_avg`, `data_std`) are
  now registered buffers and therefore appear in `state_dict()`. They previously
  did not, so any code saving a bare `state_dict` was silently losing them.
```

- [ ] **Step 3: Document how to run the tests**

Add at the end of the README, before the Citation section:

```markdown
## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
```

- [ ] **Step 4: Check the rendered Markdown**

Confirm the nested code fences inside the "Checkpoint Format" section render correctly, and that no stale `.pt` references remain:

```bash
grep -n "\.pt\b" README.md
```

Expected: only the references that intentionally discuss legacy `.pt` files.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document the safetensors checkpoint format

Covers the format, how to migrate .pt checkpoints trained with older
versions, the breaking changes, and how to run the tests.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] Full suite green: `/e/Dev/character-motion-vaes/.venv310/Scripts/python.exe -m pytest tests/ -v`
- [ ] No pickle loading anywhere: `grep -rn "torch\.load" --include=*.py . | grep -v '\.git/'` returns nothing
- [ ] No `.pt` files: `ls vae_motion/models/` shows only `.safetensors`
- [ ] All six controllers bit-identical to baseline: `../_verify/compare.py` exits 0
- [ ] Nothing stray committed: `git status --short` is clean, no `__pycache__`
- [ ] Review the full diff: `git diff main...HEAD --stat`

## Notes for the pull request

Report these to maintainers; they are deliberately out of scope.

- The `.pt` blobs remain reachable in git history. No code loads pickles now, so there is no exploitable path, but purging history would require a force push that breaks every fork and clone. That call belongs to EA.
- `TimedTargetEnv-v0` and `HumanPacmanEnv-v0` have checkpoints but are never registered in `environments/__init__.py`, and `TimedTargetEnv-v0` is `play_controller.py`'s default `--env`, so the README's documented invocation cannot work. Their checkpoints were converted but could not be rollout-verified.
- `common/misc_utils.py:218`: `--save`'s mp4 path is dead code (`self.buffer.append` is commented out), so `--save` writes PNG stills.
- `train_mvae.py` was never executed end-to-end: `environments/mocap.npz` is not distributed. Its code paths are covered by unit tests only.
- Rendered rollouts were verified by pixel comparison of offscreen frames. A maintainer should still eyeball one interactive GUI session, which this environment cannot run.
