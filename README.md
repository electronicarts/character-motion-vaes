# Character Controllers using Motion VAEs

This repo is the codebase for the SIGGRAPH 2020 paper with the title above. 
Please find the paper and demo at our project website https://www.cs.ubc.ca/~hyuling/projects/mvae/.

## Quick Start

This library should run on Linux, Mac, or Windows.

### Install Requirements

```bash
# TODO: Create and activate virtual env

cd MotionVAEs
pip install -r requirements
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
python play_mvae.py --vae models/posevae_c1_e6_l32.pt

# Control Tasks: {Target, Joystick, PathFollow, HumanMaze}Env-v0
python play_controller.py --dir models --env TargetEnv-v0
```

## Train from Scratch
Train models from scratch using train scripts.

The `train_mvae.py` script assumes the mocap data to be at `environments/mocap.npz`.
The original training data is not included in this repo; but can be easily extracted from other public datasets.
Please refer to our paper for more detail on the input format.
All training parameters can be set inside `main()` in the code.

Use `train_controller.py` to train controllers on top of trained MVAE models.
The trained model path, control task, and learning hyperparameters can be set inside `main()` in the code.
The task names follow the same convention as above, e.g. `TargetEnv-v0`, `JoystickEnv-v0`, and so on.


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