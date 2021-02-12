import os
import sys
from imageio import imwrite

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import numpy as np
import matplotlib.cm as mpl_color
import pybullet as pb
import torch
import torch.nn.functional as F

from common.bullet_objects import VSphere, VCylinder, VCapsule, FlagPole, Arrow
from common.bullet_utils import BulletClient, Camera, SinglePlayerStadiumScene


FOOT2METER = 0.3048
DEG2RAD = np.pi / 180
FADED_ALPHA = 1.0


def extract_joints_xyz(v, x_ind, y_ind, z_ind, dim=0):
    x = v.index_select(dim=dim, index=x_ind)
    y = v.index_select(dim=dim, index=z_ind)
    z = v.index_select(dim=dim, index=y_ind)
    return x, y, z


class PBLMocapViewer:
    def __init__(
        self,
        env,
        num_characters=1,
        x_ind=None,
        y_ind=None,
        z_ind=None,
        target_fps=30,
        use_params=True,
        camera_tracking=True,
    ):
        indices = torch.arange(0, 69).long()
        self.x_indices = indices[slice(3, 69, 3)] if x_ind is None else x_ind
        self.y_indices = indices[slice(4, 69, 3)] if y_ind is None else y_ind
        self.z_indices = indices[slice(5, 69, 3)] if z_ind is None else z_ind
        self.joint_indices = (self.x_indices, self.y_indices, self.z_indices)

        self.env = env
        self.num_characters = num_characters
        self.use_params = use_params

        self.device = env.device
        self.character_index = 0
        self.controller_autonomy = 1.0
        self.debug = False
        self.gui = False

        self.camera_tracking = camera_tracking
        # use 1.5 for close up, 3 for normal, 6 with GUI
        self.camera_distance = 6 if self.camera_tracking else 12
        self.camera_smooth = np.array([1, 1, 1])

        connection_mode = pb.GUI if env.is_rendered else pb.DIRECT
        self._p = BulletClient(connection_mode=connection_mode)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_MOUSE_PICKING, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

        # Disable rendering during creation
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        self.camera = Camera(
            self._p, fps=target_fps, dist=self.camera_distance, pitch=-10, yaw=45
        )
        scene = SinglePlayerStadiumScene(
            self._p, gravity=9.8, timestep=1 / target_fps, frame_skip=1
        )
        scene.initialize()

        cmap = mpl_color.get_cmap("coolwarm")
        self.colours = cmap(np.linspace(0, 1, self.num_characters))

        if num_characters == 1:
            self.colours[0] = (0.376, 0.490, 0.545, 1)

        # here order is important for some reason ?
        # self.targets = MultiTargets(self._p, num_characters, self.colours)
        self.characters = MultiMocapCharacters(self._p, num_characters, self.colours)

        # Re-enable rendering
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

        self.state_id = self._p.saveState()

        if self.use_params:
            self._setup_debug_parameters()

    def reset(self):
        # self._p.restoreState(self.state_id)
        self.env.reset()

    def add_path_markers(self, path):
        if not hasattr(self, "path"):
            num_points = min(100, len(path))
            colours = np.tile([1, 0, 0, 0.5], [num_points, 1])
            self.path = MultiTargets(self._p, num_points, colours)

        indices = torch.linspace(0, len(path) - 1, num_points).long()
        positions = F.pad(path[indices] * FOOT2METER, pad=[0, 1], value=0)
        for index, position in enumerate(positions.cpu().numpy()):
            self.path.set_position(position, index)

    def update_target_markers(self, targets):
        from environments.mocap_envs import JoystickEnv

        render_arrow = isinstance(self.env, JoystickEnv)
        if not hasattr(self, "targets"):
            marker = Arrow if render_arrow else FlagPole
            self.targets = MultiTargets(
                self._p, self.num_characters, self.colours, marker
            )

        if render_arrow:
            target_xyzs = F.pad(self.env.root_xz, pad=[0, 1]) * FOOT2METER
            target_orns = self.env.target_direction_buf

            for index, (pos, angle) in enumerate(zip(target_xyzs, target_orns)):
                orn = self._p.getQuaternionFromEuler([0, 0, float(angle)])
                self.targets.set_position(pos, index, orn)
        else:
            target_xyzs = (
                (F.pad(targets, pad=[0, 1], value=0) * FOOT2METER).cpu().numpy()
            )
            for index in range(self.num_characters):
                self.targets.set_position(target_xyzs[index], index)

    def duplicate_character(self):
        characters = self.characters
        colours = self.colours
        num_characters = self.num_characters
        bc = self._p

        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        if self.characters.has_links:
            for index, colour in zip(range(num_characters), colours):
                faded_colour = colour.copy()
                faded_colour[-1] = FADED_ALPHA
                characters.heads[index].set_color(faded_colour)
                characters.links[index] = []

        self.characters = MultiMocapCharacters(bc, num_characters, colours)

        if hasattr(self, "targets") and self.targets.marker == Arrow:
            self.targets = MultiTargets(
                self._p, self.num_characters, self.colours, Arrow
            )

        bc.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)

    def render(self, frames, facings, root_xzs, time_remain, action):

        x, y, z = extract_joints_xyz(frames, *self.joint_indices, dim=1)
        mat = self.env.get_rotation_matrix(facings).to(self.device)
        rotated_xy = torch.matmul(mat, torch.stack((x, y), dim=1))
        poses = torch.cat((rotated_xy, z.unsqueeze(dim=1)), dim=1).permute(0, 2, 1)
        root_xyzs = F.pad(root_xzs, pad=[0, 1])

        joint_xyzs = ((poses + root_xyzs.unsqueeze(dim=1)) * FOOT2METER).cpu().numpy()
        self.root_xyzs = (
            (F.pad(root_xzs, pad=[0, 1], value=3) * FOOT2METER).cpu().numpy()
        )
        self.joint_xyzs = joint_xyzs

        for index in range(self.num_characters):
            self.characters.set_joint_positions(joint_xyzs[index], index)

            if self.debug and index == self.character_index:
                target_dist = (
                    -float(self.env.linear_potential[index])
                    if hasattr(self.env, "linear_potential")
                    else 0
                )
                print(
                    "FPS: {:4.1f} | Time Left: {:4.1f} | Distance: {:4.1f} ".format(
                        self.camera._fps, float(time_remain), target_dist
                    )
                )
                if action is not None:
                    a = action[index]
                    print(
                        "max: {:4.2f} | mean: {:4.2f} | median: {:4.2f} | min: {:4.2f}".format(
                            float(a.max()),
                            float(a.mean()),
                            float(a.median()),
                            float(a.min()),
                        )
                    )

        self._handle_mouse_press()
        self._handle_key_press()
        if self.use_params:
            self._handle_parameter_update()
        if self.camera_tracking:
            self.camera.track(self.root_xyzs[self.character_index], self.camera_smooth)
        else:
            self.camera.wait()

    def close(self):
        self._p.disconnect()
        sys.exit(0)

    def _setup_debug_parameters(self):
        max_frame = self.env.mocap_data.shape[0] - self.env.num_condition_frames
        self.parameters = [
            {
                # -1 for random start frame
                "default": -1,
                "args": ("Start Frame", -1, max_frame, -1),
                "dest": (self.env, "debug_frame_index"),
                "func": lambda x: int(x),
                "post": lambda: self.env.reset(),
            },
            {
                "default": self.env.data_fps,
                "args": ("Target FPS", 1, 240, self.env.data_fps),
                "dest": (self.camera, "_target_period"),
                "func": lambda x: 1 / (x + 1),
            },
            {
                "default": 1,
                "args": ("Controller Autonomy", 0, 1, 1),
                "dest": (self, "controller_autonomy"),
                "func": lambda x: x,
            },
            {
                "default": 1,
                "args": ("Camera Track Character", 0, 1, int(self.camera_tracking)),
                "dest": (self, "camera_tracking"),
                "func": lambda x: x > 0.5,
            },
        ]

        if self.num_characters > 1:
            self.parameters.append(
                {
                    "default": 1,
                    "args": ("Selected Character", 1, self.num_characters + 0.999, 1),
                    "dest": (self, "character_index"),
                    "func": lambda x: int(x - 1.001),
                }
            )

        max_frame_skip = self.env.pose_vae_model.num_future_predictions
        if max_frame_skip > 1:
            self.parameters.append(
                {
                    "default": 1,
                    "args": (
                        "Frame Skip",
                        1,
                        max_frame_skip + 0.999,
                        self.env.frame_skip,
                    ),
                    "dest": (self.env, "frame_skip"),
                    "func": lambda x: int(x),
                }
            )

        if hasattr(self.env, "target_direction"):
            self.parameters.append(
                {
                    "default": 0,
                    "args": ("Target Direction", 0, 359, 0),
                    "dest": (self.env, "target_direction"),
                    "func": lambda x: x / 180 * np.pi,
                    "post": lambda: self.env.reset_target(),
                }
            )

        if hasattr(self.env, "target_speed"):
            self.parameters.append(
                {
                    "default": 0,
                    "args": ("Target Speed", 0.0, 0.8, 0.5),
                    "dest": (self.env, "target_speed"),
                    "func": lambda x: x,
                }
            )

        # setup Pybullet parameters
        for param in self.parameters:
            param["id"] = self._p.addUserDebugParameter(*param["args"])

    def _handle_parameter_update(self):
        for param in self.parameters:
            func = param["func"]
            value = func(self._p.readUserDebugParameter(param["id"]))
            cur_value = getattr(*param["dest"], param["default"])
            if cur_value != value:
                setattr(*param["dest"], value)
                if "post" in param:
                    post_func = param["post"]
                    post_func()

    def _handle_mouse_press(self):
        events = self._p.getMouseEvents()
        for ev in events:
            if ev[0] == 2 and ev[3] == 0 and ev[4] == self._p.KEY_WAS_RELEASED:
                # (is mouse click) and (is left click)

                width, height, _, proj, _, _, _, _, yaw, pitch, dist, target = (
                    self._p.getDebugVisualizerCamera()
                )

                pitch *= DEG2RAD
                yaw *= DEG2RAD

                R = np.reshape(
                    self._p.getMatrixFromQuaternion(
                        self._p.getQuaternionFromEuler([pitch, 0, yaw])
                    ),
                    (3, 3),
                )

                # Can't use the ones returned by pybullet API, because they are wrong
                camera_up = np.matmul(R, [0, 0, 1])
                camera_forward = np.matmul(R, [0, 1, 0])
                camera_right = np.matmul(R, [1, 0, 0])

                x = ev[1] / width
                y = ev[2] / height

                # calculate from field of view, which should be constant 90 degrees
                # can also get from projection matrix
                # d = 1 / np.tan(np.pi / 2 / 2)
                d = proj[5]

                A = target - camera_forward * dist
                aspect = height / width

                B = (
                    A
                    + camera_forward * d
                    + (x - 0.5) * 2 * camera_right / aspect
                    - (y - 0.5) * 2 * camera_up
                )

                C = (
                    np.array(
                        [
                            (B[2] * A[0] - A[2] * B[0]) / (B[2] - A[2]),
                            (B[2] * A[1] - A[2] * B[1]) / (B[2] - A[2]),
                            0,
                        ]
                    )
                    / FOOT2METER
                )

                if hasattr(self.env, "reset_target"):
                    self.env.reset_target(location=C)

    def _handle_key_press(self, keys=None):
        if keys is None:
            keys = self._p.getKeyboardEvents()
        RELEASED = self._p.KEY_WAS_RELEASED

        # keys is a dict, so need to check key exists
        if keys.get(ord("d")) == RELEASED:
            self.debug = not self.debug
        elif keys.get(ord("g")) == RELEASED:
            self.gui = not self.gui
            self._p.configureDebugVisualizer(pb.COV_ENABLE_GUI, int(self.gui))
        elif keys.get(ord("n")) == RELEASED:
            # doesn't work with pybullet's UserParameter
            self.character_index = (self.character_index + 1) % self.num_characters
            self.camera.lookat(self.root_xyzs[self.character_index])
        elif keys.get(ord("m")) == RELEASED:
            self.camera_tracking = not self.camera_tracking
        elif keys.get(ord("q")) == RELEASED:
            self.close()
        elif keys.get(ord("r")) == RELEASED:
            self.reset()
        elif keys.get(ord("t")) == RELEASED:
            self.env.reset_target()
        elif keys.get(ord("i")) == RELEASED:
            image = self.camera.dump_rgb_array()
            imwrite("image_c.png", image)
        elif keys.get(ord("a")) == RELEASED:
            image = self.camera.dump_orthographic_rgb_array()
            imwrite("image_o.png", image)
        elif keys.get(ord("v")) == RELEASED:
            import datetime

            now_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = "{}.mp4".format(now_string)

            self._p.startStateLogging(self._p.STATE_LOGGING_VIDEO_MP4, filename)
        elif keys.get(ord(" ")) == RELEASED:
            while True:
                keys = self._p.getKeyboardEvents()
                if keys.get(ord(" ")) == RELEASED:
                    break
                elif keys.get(ord("a")) == RELEASED or keys.get(ord("i")) == RELEASED:
                    self._handle_key_press(keys)


class MultiMocapCharacters:
    def __init__(self, bc, num_characters, colours=None, links=True):
        self._p = bc
        self.num_joints = 22
        total_parts = num_characters * self.num_joints

        # create all spheres at once using batchPositions
        # self.start_index = self._p.getNumBodies()
        # useMaximalCoordinates=True is faster for things that don't `move`
        joints = VSphere(bc, radius=0.07, max=True, replica=total_parts)
        self.ids = joints.ids
        self.has_links = links

        if links:
            self.linked_joints = np.array(
                [
                    [12, 0],  # right foot
                    [16, 12],  # right shin
                    [14, 16],  # right leg
                    [15, 17],  # left foot
                    [17, 13],  # left shin
                    [13, 1],  # left leg
                    [5, 7],  # right shoulder
                    [7, 10],  # right upper arm
                    [10, 20],  # right lower arm
                    [6, 8],  # left shoulder
                    [8, 9],  # left upper arm
                    [9, 21],  # left lower arm
                    [3, 18],  # torso
                    [14, 15],  # hip
                ]
            )

            self.links = {
                i: [
                    VCapsule(self._p, radius=0.06, height=0.1, rgba=colours[i])
                    for _ in range(self.linked_joints.shape[0])
                ]
                for i in range(num_characters)
            }
            self.z_axes = np.zeros((self.linked_joints.shape[0], 3))
            self.z_axes[:, 2] = 1

            self.heads = [VSphere(bc, radius=0.12) for _ in range(num_characters)]

        if colours is not None:
            self.colours = colours
            for index, colour in zip(range(num_characters), colours):
                self.set_colour(colour, index)
                if links:
                    self.heads[index].set_color(colour)

    def set_colour(self, colour, index):
        # start = self.start_index + index * self.num_joints
        start = index * self.num_joints
        joint_ids = self.ids[start : start + self.num_joints]
        for id in joint_ids:
            self._p.changeVisualShape(id, -1, rgbaColor=colour)

    def set_joint_positions(self, xyzs, index):
        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        start = index * self.num_joints
        joint_ids = self.ids[start : start + self.num_joints]
        for xyz, id in zip(xyzs, joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))

        if self.has_links:
            rgba = self.colours[index].copy()
            rgba[-1] = FADED_ALPHA

            deltas = xyzs[self.linked_joints[:, 1]] - xyzs[self.linked_joints[:, 0]]
            heights = np.linalg.norm(deltas, axis=-1)
            positions = xyzs[self.linked_joints].mean(axis=1)

            a = np.cross(deltas, self.z_axes)
            b = np.linalg.norm(deltas, axis=-1) + (deltas * self.z_axes).sum(-1)
            orientations = np.concatenate((a, b[:, None]), axis=-1)
            orientations[:, [0, 1]] *= -1

            for lid, (delta, height, pos, orn, link) in enumerate(
                zip(deltas, heights, positions, orientations, self.links[index])
            ):
                # 0.05 feet is about 1.5 cm
                if abs(link.height - height) > 0.05:
                    self._p.removeBody(link.id[0])
                    link = VCapsule(self._p, radius=0.06, height=height, rgba=rgba)
                    self.links[index][lid] = link

                link.set_position(pos, orn)

            self.heads[index].set_position(0.5 * (xyzs[4] - xyzs[3]) + xyzs[4])

        self._p.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)


class MultiTargets:
    def __init__(self, bc, num_characters, colours=None, obj_class=FlagPole):
        self._p = bc
        self.marker = obj_class

        # self.start_index = self._p.getNumBodies()
        flags = obj_class(self._p, replica=num_characters)
        self.ids = flags.ids

        if colours is not None:
            for index, colour in zip(range(num_characters), colours):
                self.set_colour(colour, index)

    def set_colour(self, colour, index):
        self._p.changeVisualShape(self.ids[index], -1, rgbaColor=colour)

    def set_position(self, xyz, index, orn=(1, 0, 0, 1)):
        self._p.resetBasePositionAndOrientation(self.ids[index], posObj=xyz, ornObj=orn)


class MocapCharacter:
    def __init__(self, bc, rgba=None):

        self._p = bc
        num_joints = 22

        # useMaximalCoordinates=True is faster for things that don't `move`
        body = VSphere(bc, radius=0.07, rgba=rgba, max=True, replica=num_joints)
        self.joint_ids = body.ids

    def set_joint_positions(self, xyzs):
        for xyz, id in zip(xyzs, self.joint_ids):
            self._p.resetBasePositionAndOrientation(id, posObj=xyz, ornObj=(0, 0, 0, 1))
