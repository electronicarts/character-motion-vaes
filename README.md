# Character Controllers using Motion VAEs

This repo is the codebase for the SIGGRAPH 2020 paper with the title above. 
Please find the paper and demo at our project website https://www.cs.ubc.ca/~hyuling/projects/mvae/.

## Quick Start

This library should run on Linux, Mac, or Windows.

### Install Requirements

```bash
# TODO: Create and activate virtual env

cd MotionVAEs
pip install -r requirements.txt
NOTE: installing pybullet requires Visual C++ 14 or higher. You can get it from here: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Run Pretrained Models

Run pretrained models using the play scripts.
The results are rendered in [PyBullet](https://github.com/bulletphysics/bullet3).
Use mouse to control camera.
Hit `r` reset task and `g` for additional controls.

```bash
cd vae_motion

# Random Walk
python play_mvae.py --vae models/posevae_c1_e6_l32.safetensors

# Control Tasks: {Target, Joystick, PathFollow, HumanMaze}Env-v0
python play_controller.py --dir models --env TargetEnv-v0
```

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
git show 5359a9b:tools/convert_checkpoints.py > convert_checkpoints.py
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

## Train from Scratch
Train models from scratch using train scripts.

The `train_mvae.py` script assumes the mocap data to be at `environments/mocap.npz`.
The original training data is not included in this repo; but can be easily extracted from other public datasets.
Please refer to our paper for more detail on the input format.
All training parameters can be set inside `main()` in the code.

Use `train_controller.py` to train controllers on top of trained MVAE models.
The trained model path, control task, and learning hyperparameters can be set inside `main()` in the code.
The task names follow the same convention as above, e.g. `TargetEnv-v0`, `JoystickEnv-v0`, and so on.


## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Citation

Please cite the following paper if you find our work useful.

```bibtex
@article{ling2020character,
  author    = {Ling, Hung Yu and Zinno, Fabio and Cheng, George and van de Panne, Michiel},
  title     = {Character Controllers Using Motion VAEs},
  year      = {2020},
  publisher = {Association for Computing Machinery},
  volume    = {39},
  number    = {4},
  journal   = {ACM Trans. Graph.}
}
```