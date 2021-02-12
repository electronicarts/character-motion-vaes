import datetime
import glob
import inspect
import math
import os
import time
import types

import numpy as np
from imageio import imwrite

if not hasattr(time, "perf_counter_ns"):
    setattr(time, "perf_counter_ns", getattr(time, "perf_counter"))


POSE_CSV_HEADER = (
    "RootDeltaPos.X,RootDeltaPos.Z,RootDeltaFacing,"
    "LeftToeBasePos.X,LeftToeBasePos.Y,LeftToeBasePos.Z,"
    "RightToeBasePos.X,RightToeBasePos.Y,RightToeBasePos.Z,"
    "Spine2Pos.X,Spine2Pos.Y,Spine2Pos.Z,"
    "Spine3Pos.X,Spine3Pos.Y,Spine3Pos.Z,"
    "NeckPos.X,NeckPos.Y,NeckPos.Z,"
    "LeftShoulderPos.X,LeftShoulderPos.Y,LeftShoulderPos.Z,"
    "RightShoulderPos.X,RightShoulderPos.Y,RightShoulderPos.Z,"
    "LeftArmPos.X,LeftArmPos.Y,LeftArmPos.Z,"
    "RightArmPos.X,RightArmPos.Y,RightArmPos.Z,"
    "RightForeArmPos.X,RightForeArmPos.Y,RightForeArmPos.Z,"
    "LeftForeArmPos.X,LeftForeArmPos.Y,LeftForeArmPos.Z,"
    "HipsPos.X,HipsPos.Y,HipsPos.Z,"
    "LeftFootPos.X,LeftFootPos.Y,LeftFootPos.Z,"
    "RightFootPos.X,RightFootPos.Y,RightFootPos.Z,"
    "LeftUpLegPos.X,LeftUpLegPos.Y,LeftUpLegPos.Z,"
    "RightUpLegPos.X,RightUpLegPos.Y,RightUpLegPos.Z,"
    "LeftLegPos.X,LeftLegPos.Y,LeftLegPos.Z,"
    "RightLegPos.X,RightLegPos.Y,RightLegPos.Z,"
    "SpinePos.X,SpinePos.Y,SpinePos.Z,"
    "Spine1Pos.X,Spine1Pos.Y,Spine1Pos.Z,"
    "LeftHandPos.X,LeftHandPos.Y,LeftHandPos.Z,"
    "RightHandPos.X,RightHandPos.Y,RightHandPos.Z,"
    "LeftToeBaseYDir.X,LeftToeBaseYDir.Y,LeftToeBaseYDir.Z,"
    "RightToeBaseYDir.X,RightToeBaseYDir.Y,RightToeBaseYDir.Z,"
    "Spine2YDir.X,Spine2YDir.Y,Spine2YDir.Z,"
    "Spine3YDir.X,Spine3YDir.Y,Spine3YDir.Z,"
    "NeckYDir.X,NeckYDir.Y,NeckYDir.Z,"
    "LeftShoulderYDir.X,LeftShoulderYDir.Y,LeftShoulderYDir.Z,"
    "RightShoulderYDir.X,RightShoulderYDir.Y,RightShoulderYDir.Z,"
    "LeftArmYDir.X,LeftArmYDir.Y,LeftArmYDir.Z,"
    "RightArmYDir.X,RightArmYDir.Y,RightArmYDir.Z,"
    "RightForeArmYDir.X,RightForeArmYDir.Y,RightForeArmYDir.Z,"
    "LeftForeArmYDir.X,LeftForeArmYDir.Y,LeftForeArmYDir.Z,"
    "HipsYDir.X,HipsYDir.Y,HipsYDir.Z,"
    "LeftFootYDir.X,LeftFootYDir.Y,LeftFootYDir.Z,"
    "RightFootYDir.X,RightFootYDir.Y,RightFootYDir.Z,"
    "LeftUpLegYDir.X,LeftUpLegYDir.Y,LeftUpLegYDir.Z,"
    "RightUpLegYDir.X,RightUpLegYDir.Y,RightUpLegYDir.Z,"
    "LeftLegYDir.X,LeftLegYDir.Y,LeftLegYDir.Z,"
    "RightLegYDir.X,RightLegYDir.Y,RightLegYDir.Z,"
    "SpineYDir.X,SpineYDir.Y,SpineYDir.Z,"
    "Spine1YDir.X,Spine1YDir.Y,Spine1YDir.Z,"
    "LeftHandYDir.X,LeftHandYDir.Y,LeftHandYDir.Z,"
    "RightHandYDir.X,RightHandYDir.Y,RightHandYDir.Z,"
    "LeftToeBaseZDir.X,LeftToeBaseZDir.Y,LeftToeBaseZDir.Z,"
    "RightToeBaseZDir.X,RightToeBaseZDir.Y,RightToeBaseZDir.Z,"
    "Spine2ZDir.X,Spine2ZDir.Y,Spine2ZDir.Z,"
    "Spine3ZDir.X,Spine3ZDir.Y,Spine3ZDir.Z,"
    "NeckZDir.X,NeckZDir.Y,NeckZDir.Z,"
    "LeftShoulderZDir.X,LeftShoulderZDir.Y,LeftShoulderZDir.Z,"
    "RightShoulderZDir.X,RightShoulderZDir.Y,RightShoulderZDir.Z,"
    "LeftArmZDir.X,LeftArmZDir.Y,LeftArmZDir.Z,"
    "RightArmZDir.X,RightArmZDir.Y,RightArmZDir.Z,"
    "RightForeArmZDir.X,RightForeArmZDir.Y,RightForeArmZDir.Z,"
    "LeftForeArmZDir.X,LeftForeArmZDir.Y,LeftForeArmZDir.Z,"
    "HipsZDir.X,HipsZDir.Y,HipsZDir.Z,"
    "LeftFootZDir.X,LeftFootZDir.Y,LeftFootZDir.Z,"
    "RightFootZDir.X,RightFootZDir.Y,RightFootZDir.Z,"
    "LeftUpLegZDir.X,LeftUpLegZDir.Y,LeftUpLegZDir.Z,"
    "RightUpLegZDir.X,RightUpLegZDir.Y,RightUpLegZDir.Z,"
    "LeftLegZDir.X,LeftLegZDir.Y,LeftLegZDir.Z,"
    "RightLegZDir.X,RightLegZDir.Y,RightLegZDir.Z,"
    "SpineZDir.X,SpineZDir.Y,SpineZDir.Z,"
    "Spine1ZDir.X,Spine1ZDir.Y,Spine1ZDir.Z,"
    "LeftHandZDir.X,LeftHandZDir.Y,LeftHandZDir.Z,"
    "RightHandZDir.X,RightHandZDir.Y,RightHandZDir.Z"
)


def rad_to_deg(rad):
    return rad * 180 / np.pi


def deg_to_rad(deg):
    return deg / 180 * np.pi


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (as_minutes(s), as_minutes(rs))


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr, final_lr=0):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr - final_lr) * epoch / float(total_num_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_exponential_schedule(optimizer, epoch, rate, initial_lr, final_lr=0):
    assert initial_lr >= final_lr, "Initial lr must be greater than final lr."
    lr = initial_lr * (rate ** epoch)
    lr = max(lr, final_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class FPSController:
    def __init__(self, target_fps):
        self.timestamp = time.perf_counter_ns()
        self.target_fps = target_fps

    def wait(self):
        fps = 1e9 / (time.perf_counter_ns() - self.timestamp)
        time.sleep(max(1.0 / self.target_fps - 1.0 / fps, 0))
        fps = 1e9 / (time.perf_counter_ns() - self.timestamp)
        self.timestamp = time.perf_counter_ns()
        return fps


class EpisodeRunner(object):
    def __init__(self, env, save=False, dir=None, max_steps=None, csv=None):
        self.env = env
        self.save = save
        self.done = False
        self.csv = csv

        self.max_steps = float("inf") if max_steps is None else max_steps
        self.step = 0

        base_dir = os.path.dirname(os.path.realpath(inspect.stack()[-1][1]))
        self.dump_dir = os.path.join(base_dir, "dump") if dir is None else dir
        if self.csv is not None:
            self.csv = os.path.join(base_dir, self.csv)

        self.override_reset()
        self.override_render()

        if self.csv is not None:
            self.pose_buffer = []
            self.additional_render_data_buffer = {}

        if self.save:
            self.camera = env.viewer.camera
            self.buffer = []
            self.max_steps = env.max_timestep if max_steps is None else max_steps

            now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.filename = os.path.join(self.dump_dir, "{}.mp4".format(now_string))
            print("\nRecording... Close to terminate recording.")

            try:
                os.makedirs(self.dump_dir)
            except OSError:
                files = glob.glob(os.path.join(self.dump_dir, "*.png"))
                for f in files:
                    os.remove(f)

        self.pbar = None
        if self.max_steps != env.max_timestep and self.max_steps != float("inf"):
            try:
                from tqdm import tqdm

                self.pbar = tqdm(total=self.max_steps)
            except ImportError:
                pass

    def override_reset(self):
        old_reset_func = self.env.reset
        runner = self

        def new_reset(self, indices=None):
            return old_reset_func(indices)

        self.env.reset = types.MethodType(new_reset, self.env)

    def override_render(self):

        old_render_func = self.env.render

        runner = self

        def new_render(self):
            old_render_func()
            runner.store_current_frame()
            runner.save_csv_render_data()
            runner.step += 1

            if runner.pbar is not None:
                runner.pbar.update(1)

            if runner.step >= runner.max_steps:
                runner.done = True
                if runner.pbar is not None:
                    runner.pbar.close()

        self.env.render = types.MethodType(new_render, self.env)

    def store_current_frame(self):
        if self.save:
            image = self.camera.dump_rgb_array()
            # self.buffer.append(image)
            imwrite("outfile_{:04d}.png".format(self.step), image)

    def save_csv_render_data(self):
        if self.csv is not None:
            np_obs = self.env.history[:, 0].clone().cpu().numpy()
            # Ignore velocities
            pose = np.concatenate((np_obs[0, 0:69], np_obs[0, 135:267]))
            self.pose_buffer.append(pose)

            # Target needs to be in global coordinate
            render_data_dict = self.env.dump_additional_render_data()
            for file, render_data in render_data_dict.items():
                header = render_data["header"]
                data = render_data["data"].clone().cpu().numpy()
                once = render_data.get("once", False)

                if file not in self.additional_render_data_buffer:
                    list_data = data if once else []
                    self.additional_render_data_buffer[file] = {
                        "header": header,
                        "data": list_data,
                    }

                if not once:
                    self.additional_render_data_buffer[file]["data"].append(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.save and len(self.buffer) >= self.max_steps:
            import moviepy.editor as mp

            # clip = mp.ImageSequenceClip(self.buffer, fps=self.env.data_fps)
            # clip.write_videofile(self.filename)

        if self.csv is not None:
            np.savetxt(
                os.path.join(self.csv, "pose.csv"),
                np.asarray(self.pose_buffer),
                delimiter=",",
                header=POSE_CSV_HEADER,
                comments="",
            )

            for file, data_dict in self.additional_render_data_buffer.items():
                np.savetxt(
                    os.path.join(self.csv, file),
                    np.asarray(data_dict["data"]),
                    delimiter=",",
                    header=data_dict["header"],
                    comments="",
                )


def str2bool(v):
    """
    Argument Parse helper function for boolean values
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def line_to_point_distance(p1, p2, c):
    l = p2 - p1

    d = (
        l.select(-1, 1) * c.select(-1, 0)
        - l.select(-1, 0) * c.select(-1, 1)
        + p2.select(-1, 0) * p1.select(-1, 1)
        - p2.select(-1, 1) * p1.select(-1, 0)
    ).abs() / l.norm(dim=-1)

    vector2 = c - p2
    vector3 = c - p1
    cosine1 = (vector2 * -l).sum(dim=-1)
    cosine2 = (vector3 * l).sum(dim=-1)

    mask = (cosine1 > 0) * (cosine2 > 0)
    # masks = masks.view(self.num_parallel, self.num_eyes, 2, self.num_pellets)

    # distances = (
    #     vector3.view(self.num_parallel, 1, 2, self.num_pellets, p1.size(-1))
    #     .norm(dim=-1)
    #     .expand_as(masks)
    # )

    return d, mask
