# Migrating model checkpoints from pickled `.pt` to safetensors

Date: 2026-09-23
Status: approved, pending implementation plan

## Problem

Every checkpoint in this repo is a pickled `nn.Module`, not a `state_dict`:

- `vae_motion/train_mvae.py:319` — `torch.save(copy.deepcopy(pose_vae).cpu(), pose_vae_path)`
- `vae_motion/train_controller.py:224` — the same for `PoseVAEPolicy`

They are read back at five sites: `train_mvae.py:213`, `train_controller.py:139`,
`play_controller.py:58`, `environments/mocap_envs.py:122`, and `mocap_envs.py:997`.
Seven such files are committed under `vae_motion/models/`.

Loading a pickle executes arbitrary Python by design, which is what the security scan
flagged. `weights_only=True` cannot mitigate it here: that mode refuses whole-object
pickles outright, so it is not a drop-in hardening option.

### The repo is already broken on current PyTorch

PyTorch 2.6 changed the `weights_only` default from `False` to `True`. Verified against
torch 2.14.0: every `torch.load` call in this repo now raises `UnpicklingError`, and the
committed pretrained models cannot be loaded at all. This migration therefore repairs a
functional break in addition to closing the security finding.

### The trap that makes a naive migration dangerous

`models.py:221-225` (and the four sibling classes) assign the normalization statistics as
**plain attributes**, not buffers:

```python
self.data_max = normalization.get("max")
```

They are consequently absent from `state_dict()`. Verified on the committed VAE:

```
mode: 'zscore'
data_max Tensor torch.Size([267])    # present on the object
state_dict keys: 20
has data_max in state_dict: False    # absent from state_dict
```

These tensors survive today only because the entire object is pickled. A `state_dict`-based
migration that does not address this would silently drop all four, making
`normalize`/`denormalize` (`models.py:239-257`) produce wrong output with no error raised.
Fixing this with `register_buffer` is mandatory, not cosmetic.

## Decisions taken

| Decision | Choice |
|---|---|
| Compatibility | Hard cut. Convert the 7 committed checkpoints, delete the `.pt` files, no pickle-loading path remains. |
| Config storage | Inside the safetensors header `__metadata__`. One self-contained file per model. |
| Conversion | Performed locally in a venv, converted files committed. |
| Git history | Left alone. No history rewrite, no force push. |
| Dependency rot | Fixed in a separate commit (pins + `gym.make` call sites). |

## On-disk format

```
posevae_c1_e6_l32.safetensors
  __metadata__:
    format:             "pt"
    cmv_format_version: "1"
    config:             {"class": "PoseMixtureVAE", "frame_size": 267,
                         "latent_size": 32, "num_condition_frames": 1,
                         "num_future_predictions": 1, "num_experts": 6,
                         "normalization_mode": "zscore"}
  tensors:
    data_max, data_min, data_avg, data_std          # newly part of state_dict
    encoder.fc1.weight, ... decoder.w0, decoder.gate.0.weight, ...
```

`__metadata__` is a `str -> str` map, so `config` holds a JSON string. Tensor-valued
normalization statistics stay real tensors; only `mode` (a string) and the integer
hyperparameters live in the config. `cmv_format_version` costs nothing now and buys a clear
error if the schema ever changes.

## Components

### `common/model_io.py` (new)

Format primitives only, with no knowledge of any model class. This placement is deliberate:
`common/` currently imports neither `vae_motion/` nor `environments/`, so keeping the format
layer free of model imports avoids a cycle.

- `save_tensors_with_config(tensors, config, path)` — calls `.detach().cpu().clone().contiguous()`
  on every tensor, because safetensors rejects shared storage and non-contiguous layouts,
  then writes via `save_file` with the metadata map.
- `load_tensors_with_config(path, device)` — `safe_open`, validate `cmv_format_version`,
  parse the config JSON, return `(tensors, config)`. A safetensors file without a `config`
  key raises a clear "not a character-motion-vaes checkpoint" error rather than `KeyError`.

### `vae_motion/models.py` (modified)

**`NormalizationMixin` (new).** Five classes (`AutoEncoder`, `PoseVAE`, `PoseMixtureVAE`,
`PoseMixtureSpecialistVAE`, `PoseVQVAE`) each carry a byte-identical copy of the same five
attribute assignments and the same `normalize`/`denormalize` pair — roughly 100 duplicated
lines. Extract a mixin providing `_init_normalization(normalization)`, `normalize`, and
`denormalize`, registering `data_max/min/avg/std` via `register_buffer`.

This is the only refactor beyond the strict minimum, and it is in direct service of the
goal: the silent-data-loss bug gets fixed once rather than in five places a reviewer would
have to verify independently.

**Loading normalization without guessing shapes.** `load_model` reads the tensor dict first,
pulls the four statistic tensors out of it, and passes them into the constructor as the
`normalization` dict. The buffers therefore already exist at the correct shape before
`load_state_dict(strict=True)` runs, which then re-copies them as a no-op. No placeholder
shapes, no `strict=False` escape hatch; genuine key or shape mismatches still fail loudly.

**`PoseVAEController.__init__(observation_dim, action_dim)`** replaces the `env` parameter
(`models.py:601`). `play_controller.py:58` loads the controller before any env exists, so
depending on a live env is unworkable. The single construction site
(`train_controller.py:141`) already has `args.observation_size` and `args.action_size` in
scope from lines 104-105.

**`env_info` is removed.** `save_model(model, path, extra_config={"frame_skip": n})` folds
`frame_skip` into the config, deleting the attribute-stuffing at `train_controller.py:146`
and the `hasattr` probe at `play_controller.py:59`.

**Public API added to `models.py`:**

- `MODEL_REGISTRY` — class name to class, covering the five VAE classes plus
  `PoseVAEController` and `PoseVAEPolicy`.
- `save_model(model, path, extra_config=None)`
- `load_model(path, device)`
- `read_config(path)` — header-only read, used solely to recover `frame_skip`.
- a `config` property on each registered class returning its own constructor arguments.

### `tools/convert_checkpoints.py` (new)

Unpickling an `nn.Module` does not call `__init__` (state is restored into `__dict__`
directly), so the changed `PoseVAEController` signature does not prevent reading the old
files. The converter must pass `weights_only=False` explicitly, since torch 2.6+ defaults it
to `True`.

Per file: load, derive config from the live object's attributes (`num_experts` from
`decoder.w0.shape[0]`, `num_embeddings` from `quantizer.num_embeddings`, controller dims from
the nested controller, `frame_skip` from `env_info`), write the `.safetensors`.

`--verify` is the part that earns reviewer trust. For each file it reloads the converted
model and asserts:

1. `state_dict` keys match exactly and every tensor compares equal under `torch.equal`.
2. A fixed-seed forward pass through the original pickle and the reloaded model produces
   exactly equal output, in `eval()` mode on the deterministic code paths.

## Call-site changes

| Location | Change |
|---|---|
| `train_mvae.py:196-208` | `.pt` to `.safetensors` in the four filename templates |
| `train_mvae.py:213` | `load_model(pose_vae_path, args.device)` |
| `train_mvae.py:319` | `save_model(pose_vae, pose_vae_path)`; drops `copy.deepcopy().cpu()` |
| `train_controller.py:103` | `.pt` to `.safetensors` |
| `train_controller.py:139` | `load_model(...)` |
| `train_controller.py:141` | new constructor signature |
| `train_controller.py:146` | delete the `env_info` assignment |
| `train_controller.py:224` | `save_model(actor_critic, path, {"frame_skip": args.frame_skip})` |
| `play_controller.py:30-31` | glob `con*.safetensors` and `posevae*.safetensors` |
| `play_controller.py:58-62` | `load_model` plus `read_config(...).get("frame_skip", 1)` |
| `mocap_envs.py:118` | glob `posevae*.safetensors` |
| `mocap_envs.py:122` | `load_model(...)` |
| `mocap_envs.py:996-997` | `con_TargetEnv-v0.safetensors`, `load_model(...).actor` |

`mocap_envs.py` gains `from vae_motion.models import load_model`; that direction is acyclic.
The now-unused `copy` imports are removed.

## Verification

### Environment

Reproducible via `uv`: Python 3.10.19, `gym==0.23.1`, `torch`, `pybullet`, `numpy<2`,
`safetensors`, `pytest`. gym 0.21 and earlier are uninstallable (invalid
`opencv-python>=3.` metadata); 0.23.1 is the newest release that still provides both
`registry.env_specs` and the deprecated `rng.randint` shim this code relies on.

### Unit tests (`tests/`, new — the repo has none today)

- `test_model_io.py`, parametrized over all seven registered classes at small random dims:
  save, load, then assert both `state_dict` equality and forward-output equality.
- Normalization round-trip, including that `data_max` and siblings actually survive — the
  regression test for the buffer bug — and that `denormalize(normalize(x)) == x`.
- A `MixedDecoder`-specific test. `models.py:152-175` keeps its parameters in a plain Python
  list of tuples *and* registers them as `w0..w2`/`b0..b2`. `load_state_dict` copies in place
  so the list's references stay valid, but that is load-bearing and non-obvious, so it gets an
  explicit test asserting `forward` reflects loaded weights.
- Error cases: safetensors file lacking `config`, unknown class name, version mismatch.
- `test_converted_checkpoints.py`: load each committed `.safetensors` and run a forward pass.

### End-to-end equivalence

Rollouts are bit-exactly reproducible under `torch.manual_seed` plus `env.seed` (verified by
running the same seed twice and comparing arrays). Therefore:

1. **Before any code change**, capture baselines from the current pickles: seeded rollouts
   producing per-frame pose CSVs for all six controllers, plus offscreen screenshots.
2. After migration, re-run byte-identical seeded rollouts against the `.safetensors` models
   and assert the CSVs are *exactly* equal. Any drift is a bug.
3. Re-render screenshots for visual before/after comparison.

Offscreen rendering works without a display by forcing `pybullet.DIRECT` and calling
`getCameraImage` with `ER_TINY_RENDERER`; the repo's own `dump_rgb_array`
(`common/bullet_utils.py:405`) requests 3840x2160 hardware OpenGL, which is unsuitable
headless, so the harness uses its own lower-resolution capture.

## Dependency repair (separate commit)

`requirements.txt` pins nothing, which is why the repo no longer installs. Pinning alone is
insufficient: `gym.make("environments:TargetEnv-v0")` relied on pre-0.22 semantics where the
prefix meant "import this module, then look up the id". From 0.22 the prefix is a *namespace*,
and `environments/__init__.py` registers without one, so all three call sites fail on every
installable gym.

Scope of this commit: pin the proven version set, add `safetensors`, fix the three `gym.make`
call sites, and add a `.gitignore` for `__pycache__` (previously committed by accident and
deleted in commits `1bdafd9`, `441df16`, `e1a7e96`).

## Documentation

`README.md` gains a checkpoint-format note, converter usage, and a breaking-change callout
covering both the `PoseVAEController` signature and `.pt` files no longer being loadable.

## Out of scope

Recorded so they are visible to maintainers rather than silently absorbed:

- Rewriting git history to purge the `.pt` blobs. They remain reachable in history, but no
  code path loads pickles any more. Rewriting a public repo's history breaks every fork and
  clone, and is a decision for EA, not this change.
- Any `.pt` fallback loading path.
- `TimedTargetEnv-v0` and `HumanPacmanEnv-v0` have committed checkpoints but are never
  registered in `environments/__init__.py`, and `TimedTargetEnv-v0` is `play_controller.py`'s
  default `--env`, so the README's documented invocation cannot work. Reported in the PR.
- `misc_utils.py:218`: the `--save` mp4 path is dead code (`self.buffer.append` is commented
  out), so `--save` emits PNG stills. Misleading but harmless.
- Retraining. `environments/mocap.npz` is not distributed, so `train_mvae.py` cannot run here.
  Training code paths are updated but exercised only by unit tests.

## Rollback

The `.pt` files are deleted in this branch but remain in history:
`git checkout <sha> -- vae_motion/models/`.
