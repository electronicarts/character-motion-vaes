import glob
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import gym
import gym.spaces
import gym.utils
import gym.utils.seeding
import numpy as np
import torch
import torch.nn.functional as F

from common.misc_utils import line_to_point_distance
from environments.mocap_renderer import extract_joints_xyz



FOOT2METER = 0.3048
METER2FOOT = 1 / 0.3048


class EnvBase(gym.Env):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        self.np_random = None
        self.seed()

        self.is_rendered = rendered
        self.num_parallel = num_parallel
        self.frame_skip = frame_skip
        self.device = device

        self.load_data(pose_vae_path)

        self.action_scale = 4.0
        self.data_fps = 30
        self.frame_dim = self.mocap_data.shape[1]
        self.num_condition_frames = self.pose_vae_model.num_condition_frames

        # Action is the latent dim
        self.action_dim = (
            self.pose_vae_model.latent_size
            if not hasattr(self.pose_vae_model, "quantizer")
            else self.pose_vae_model.quantizer.num_embeddings
        )

        self.max_timestep = int(1200 / self.frame_skip)

        # history size is used to calculate floating as well
        self.history_size = 5
        assert (
            self.history_size >= self.num_condition_frames
        ), "History size has to be greater than condition size."
        self.history = torch.zeros(
            (self.num_parallel, self.history_size, self.frame_dim)
        ).to(self.device)

        indices = self.np_random.randint(0, self.mocap_data.shape[0], self.num_parallel)
        indices = torch.from_numpy(indices).long()

        self.start_indices = indices
        self.root_facing = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.root_xz = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.reward = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.potential = torch.zeros((self.num_parallel, 2)).to(self.device)
        self.done = torch.zeros((self.num_parallel, 1)).bool().to(self.device)

        # used for reward-based early termination
        self.parallel_ind_buf = (
            torch.arange(0, self.num_parallel).long().to(self.device)
        )

        # 4 and 7 are height for right and left toes respectively
        # y-axis in the data, but z-axis in the env
        self.foot_xy_ind = torch.LongTensor([[3, 5], [6, 8]])
        self.foot_z_ind = torch.LongTensor([4, 7])
        self.contact_threshold = 0.03 * METER2FOOT
        self.foot_pos_history = torch.zeros((self.num_parallel, 2, 6)).to(self.device)

        indices = torch.arange(0, 69).long().to(self.device)
        x_indices = indices[slice(3, 69, 3)]
        y_indices = indices[slice(4, 69, 3)]
        z_indices = indices[slice(5, 69, 3)]
        self.joint_indices = (x_indices, y_indices, z_indices)

        if self.is_rendered:
            from .mocap_renderer import PBLMocapViewer

            self.viewer = PBLMocapViewer(
                self,
                num_characters=num_parallel,
                target_fps=self.data_fps,
                use_params=use_params,
                camera_tracking=camera_tracking,
            )

        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def load_data(self, pose_vae_path):
        mocap_file = os.path.join(current_dir, "pose0.npy")
        data = torch.from_numpy(np.load(mocap_file))
        self.mocap_data = data.float().to(self.device)

        if os.path.isdir(pose_vae_path):
            basepath = os.path.normpath(pose_vae_path)
            pose_vae_path = glob.glob(os.path.join(basepath, "posevae*.pt"))[0]
        else:
            basepath = os.path.dirname(pose_vae_path)

        self.pose_vae_model = torch.load(pose_vae_path, map_location=self.device)
        self.pose_vae_model.eval()

        assert (
            self.pose_vae_model.num_future_predictions >= self.frame_skip
        ), "VAE cannot skip this many frames"

        print("=========")
        print("Loaded: ", mocap_file)
        print("Loaded: ", pose_vae_path)
        print("=========")

    def integrate_root_translation(self, pose):
        mat = self.get_rotation_matrix(self.root_facing)
        displacement = (mat * pose[:, 0:2].unsqueeze(1)).sum(dim=2)
        self.root_facing.add_(pose[:, [2]]).remainder_(2 * np.pi)
        self.root_xz.add_(displacement)

        self.history = self.history.roll(1, dims=1)
        self.history[:, 0].copy_(pose)

        foot_z = pose[:, self.foot_z_ind].unsqueeze(-1)
        foot_xy = pose[:, self.foot_xy_ind]
        foot_pos = torch.cat((self.root_xz.unsqueeze(1) + foot_xy, foot_z), dim=-1)
        self.foot_pos_history = self.foot_pos_history.roll(1, dims=1)
        self.foot_pos_history[:, 0].copy_(foot_pos.flatten(1, 2))

    def get_rotation_matrix(self, yaw, dim=2):
        yaw = -yaw
        zeros = torch.zeros_like(yaw)
        ones = torch.ones_like(yaw)
        if dim == 3:
            col1 = torch.cat((yaw.cos(), yaw.sin(), zeros), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos(), zeros), dim=-1)
            col3 = torch.cat((zeros, zeros, ones), dim=-1)
            matrix = torch.stack((col1, col2, col3), dim=-1)
        else:
            col1 = torch.cat((yaw.cos(), yaw.sin()), dim=-1)
            col2 = torch.cat((-yaw.sin(), yaw.cos()), dim=-1)
            matrix = torch.stack((col1, col2), dim=-1)
        return matrix

    def get_vae_condition(self, normalize=False, flatten=True):
        condition = self.history[:, : self.num_condition_frames]
        if normalize:
            condition = self.pose_vae_model.normalize(condition)
        if flatten:
            condition = condition.flatten(start_dim=1, end_dim=2)
        return condition

    def get_vae_next_frame(self, action):
        self.action = action
        condition = self.get_vae_condition(normalize=True, flatten=False)

        with torch.no_grad():
            condition = condition.flatten(start_dim=1, end_dim=2)

            vae_output = self.pose_vae_model.sample(
                action, condition, deterministic=True
            )
            vae_output = vae_output.view(
                -1,
                self.pose_vae_model.num_future_predictions,
                self.pose_vae_model.frame_size,
            )

        next_frame = self.pose_vae_model.denormalize(vae_output)
        return next_frame

    def reset_initial_frames(self, frame_index=None):
        # Make sure condition_range doesn't blow up
        self.start_indices.random_(
            0, self.mocap_data.shape[0] - self.num_condition_frames + 1
        )

        if self.is_rendered:
            # controlled from GUI
            param_name = "debug_frame_index"
            if hasattr(self, param_name) and getattr(self, param_name) != -1:
                self.start_indices.fill_(getattr(self, param_name))

        # controlled from CLI
        if frame_index is not None:
            if self.is_rendered:
                setattr(self, param_name, frame_index)
                self.start_indices.fill_(getattr(self, param_name))
            else:
                self.start_indices.fill_(frame_index)

        # Newer has smaller index (ex. index 0 is newer than 1)
        condition_range = (
            self.start_indices.repeat((self.num_condition_frames, 1)).t()
            + torch.arange(self.num_condition_frames - 1, -1, -1).long()
        )

        self.history[:, : self.num_condition_frames].copy_(
            self.mocap_data[condition_range]
        )

    def calc_foot_slide(self):
        foot_z = self.foot_pos_history[:, :, [2, 5]]
        # in_contact = foot_z < self.contact_threshold
        # contact_coef = in_contact.all(dim=1).float()

        # foot_xy = self.foot_pos_history[:, :, [[0, 1], [3, 4]]]
        # displacement = (
        #     (foot_xy.unsqueeze(1) - foot_xy.unsqueeze(2))
        #     .norm(dim=-1)
        #     .max(dim=1)[0]
        #     .max(dim=1)[0]
        # )

        # print(self.foot_pos_history[:, 0, [2, 5]], contact_coef * displacement)
        # foot_slide = contact_coef * displacement

        displacement = self.foot_pos_history[:, 0] - self.foot_pos_history[:, 1]
        displacement = displacement[:, [[0, 1], [3, 4]]].norm(dim=-1)

        foot_slide = displacement.mul(
            2 - 2 ** (foot_z.max(dim=1)[0] / self.contact_threshold).clamp_(0, 1)
        )

        return foot_slide

    def calc_energy_penalty(self, next_frame):
        action_energy = (
            next_frame[:, [0, 1]].pow(2).sum(1)
            + next_frame[:, 2].pow(2)
            + next_frame[:, 69:135].pow(2).mean(1)
        )
        return -0.8 * action_energy.unsqueeze(dim=1)

    def calc_action_penalty(self):
        prob_energy = self.action.abs().mean(-1, keepdim=True)
        return -0.01 * prob_energy

    def step(self, action: torch.Tensor):
        action = action * self.action_scale
        next_frame = self.get_vae_next_frame(action)
        for i in range(self.frame_skip):
            state = self.calc_env_state(next_frame[:, i])
        return state

    def calc_env_state(self, next_frame):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.is_rendered:
            self.viewer.close()

    def render(self, mode="human"):
        self.viewer.render(
            self.history[:, 0],  # 0 is the newest
            self.root_facing,
            self.root_xz,
            0.0,  # No time in this env
            self.action,
        )


class RandomWalkEnv(EnvBase):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        self.max_timestep = 1000
        self.base_action = torch.zeros((self.num_parallel, self.action_dim)).to(
            self.device
        )

        self.observation_dim = (
            self.frame_dim * self.num_condition_frames + self.action_dim
        )
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def get_observation_components(self):
        self.base_action.normal_(0, 1)
        condition = self.get_vae_condition(normalize=False)
        return condition, self.base_action

    def reset(self, indices=None):
        self.timestep = 0
        self.substep = 0
        self.root_facing.fill_(0)
        self.root_xz.fill_(0)
        self.done.fill_(False)
        # Need to clear this if we want to use calc_foot_slide()
        self.foot_pos_history.fill_(1)

        self.reset_initial_frames()
        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def get_vae_next_frame(self, action):
        action = (self.base_action + action) / 2
        return super().get_vae_next_frame(action)

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        self.reward.fill_(1)  # Alive bonus
        # energy_penalty = self.calc_energy_penalty(next_frame)
        # self.reward.add_(energy_penalty)
        foot_slide = self.calc_foot_slide()
        self.reward.add_(foot_slide.sum(dim=-1, keepdim=True) * -10.0)

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

    def dump_additional_render_data(self):
        from common.misc_utils import POSE_CSV_HEADER

        current_frame = self.history[:, 0]
        pose_data = torch.cat(
            (current_frame[:, 0:69], current_frame[:, 135:267]), dim=-1
        )

        data_dict = {
            "pose{}.csv".format(index): {"header": POSE_CSV_HEADER}
            for index in range(pose_data.shape[0])
        }
        for index, pose in enumerate(pose_data):
            key = "pose{}.csv".format(index)
            data_dict[key]["data"] = pose.clone()

        return data_dict


class TargetEnv(EnvBase):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        self.arena_length = (-60.0, 60.0)
        self.arena_width = (-40.0, 40.0)

        # 2D delta to task in root space
        target_dim = 2
        self.target = torch.zeros((self.num_parallel, target_dim)).to(self.device)

        self.observation_dim = (self.frame_dim * self.num_condition_frames) + target_dim
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def calc_potential(self):
        target_delta, target_angle = self.get_target_delta_and_angle()
        self.linear_potential = -target_delta.norm(dim=1).unsqueeze(1)
        self.angular_potential = target_angle.cos()

    def get_target_delta_and_angle(self):
        target_delta = self.target - self.root_xz
        target_angle = (
            torch.atan2(target_delta[:, 1], target_delta[:, 0]).unsqueeze(1)
            + self.root_facing
        )
        return target_delta, target_angle

    def get_observation_components(self):
        target_delta, _ = self.get_target_delta_and_angle()
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        delta = (mat * target_delta.unsqueeze(1)).sum(dim=2)
        condition = self.get_vae_condition(normalize=False)
        return condition, delta

    def reset(self, indices=None):
        if indices is None:
            self.root_facing.fill_(0)
            self.root_xz.fill_(0)
            self.reward.fill_(0)
            self.timestep = 0
            self.substep = 0
            self.done.fill_(False)
            # value bigger than contact_threshold
            self.foot_pos_history.fill_(1)

            self.reset_target()
            self.reset_initial_frames()
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.root_xz.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.reset_target(indices)

            # value bigger than contact_threshold
            self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)

        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def reset_target(self, indices=None, location=None):
        if location is None:
            if indices is None:
                self.target[:, 0].uniform_(*self.arena_length)
                self.target[:, 1].uniform_(*self.arena_width)
            else:
                # if indices is a pytorch tensor, this returns a new storage
                new_lengths = self.target[indices, 0].uniform_(*self.arena_length)
                self.target[:, 0].index_copy_(dim=0, index=indices, source=new_lengths)
                new_widths = self.target[indices, 1].uniform_(*self.arena_width)
                self.target[:, 1].index_copy_(dim=0, index=indices, source=new_widths)
        else:
            # Reaches this branch only with mouse click in render mode
            self.target[:, 0] = location[0]
            self.target[:, 1] = location[1]

        # l = np.random.uniform(*self.arena_length)
        # w = np.random.uniform(*self.arena_width)
        # self.target[:, 0].fill_(0)
        # self.target[:, 1].fill_(100)

        # set target to be in front
        # facing_delta = self.root_facing.clone().uniform_(-np.pi / 2, np.pi / 2)
        # angle = self.root_facing + facing_delta
        # distance = self.root_facing.clone().uniform_(20, 60)
        # self.target[:, 0].copy_((distance * angle.cos()).squeeze(1))
        # self.target[:, 1].copy_((distance * angle.sin()).squeeze(1))

        # Getting image
        # facing_delta = self.root_facing.clone().fill_(-np.pi / 6)
        # angle = self.root_facing + facing_delta
        # distance = self.root_facing.clone().fill_(40)
        # self.target[:, 0].copy_((distance * angle.cos()).squeeze(1))
        # self.target[:, 1].copy_((distance * angle.sin()).squeeze(1))

        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

        # Should do this every time target is reset
        self.calc_potential()

    def calc_progress_reward(self):
        old_linear_potential = self.linear_potential
        old_angular_potential = self.angular_potential

        self.calc_potential()

        linear_progress = self.linear_potential - old_linear_potential
        angular_progress = self.angular_potential - old_angular_potential
        progress = linear_progress

        return progress

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        # Check if target is reached
        # Has to be done after new potentials are calculated
        target_dist = -self.linear_potential
        target_is_close = target_dist < 2.0

        if is_external_step:
            self.reward.copy_(progress)
        else:
            self.reward.add_(progress)

        self.reward.add_(target_is_close.float() * 20.0)

        energy_penalty = self.calc_energy_penalty(next_frame)
        self.reward.add_(energy_penalty)

        # action_penalty = self.calc_action_penalty()
        # self.reward.add_(action_penalty)

        if target_is_close.any():
            reset_indices = self.parallel_ind_buf.masked_select(
                target_is_close.squeeze(1)
            )
            self.reset_target(indices=reset_indices)

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def dump_additional_render_data(self):
        return {"extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]}}

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

        # if self.is_rendered and self.timestep % 10 == 0:
        #     self.viewer.duplicate_character()


class JoystickEnv(TargetEnv):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):

        # Need to do this before calling super()
        # otherwise parameter will not be set up
        self.target_direction = 0
        self.target_speed = 0

        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        condition_size = self.frame_dim * self.num_condition_frames
        # 2 because we are doing cos() and sin()
        self.observation_dim = condition_size + 2

        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # tensor buffers for direction and speed
        self.target_direction_buf = torch.zeros((self.num_parallel, 1)).to(self.device)
        self.target_speed_buf = torch.zeros((self.num_parallel, 1)).to(self.device)

    def reset_target(self, indices=None, location=None):
        facing_switch_every = self.max_timestep if self.is_rendered else 120
        speed_switch_every = self.max_timestep if self.is_rendered else 240

        if self.timestep % facing_switch_every == 0:

            if self.is_rendered:
                # in rendered mode, we use these because they are linked to gui
                self.target_direction = self.np_random.uniform(0, 2 * np.pi)
            else:
                # in training mode, we have tensors
                self.target_direction_buf.uniform_(0, 2 * np.pi)

        if self.timestep % speed_switch_every == 0:
            if self.is_rendered:
                # in rendered mode, we use these because they are linked to gui
                self.target_speed = self.np_random.choice(np.linspace(0, 0.8, 9))
            else:
                # in training mode, we have tensors
                choices = torch.linspace(0, 0.8, 9).to(self.device)
                sample = torch.randint(0, choices.size(0), (self.num_parallel, 1))
                self.target_speed_buf.copy_(choices[sample])

        self.target.copy_(self.root_xz)
        if self.is_rendered:
            self.target[:, 0].add_(10 * np.cos(self.target_direction))
            self.target[:, 1].add_(10 * np.sin(self.target_direction))
            # Overwrite buffer because they are part of the observation returned to controller
            self.target_speed_buf.fill_(self.target_speed)
            # Need to overwrite this for dumping rendering data
            self.target_direction_buf.fill_(self.target_direction)
        else:
            self.target[:, 0].add_(10 * self.target_direction_buf.cos().squeeze())
            self.target[:, 1].add_(10 * self.target_speed_buf.sin().squeeze())

        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

        self.calc_potential()

    def calc_progress_reward(self):
        _, target_angle = self.get_target_delta_and_angle()
        direction_reward = target_angle.cos().add(-1)
        speed = self.next_frame[:, [0, 1]].norm(dim=1, keepdim=True)
        speed_reward = (self.target_speed_buf - speed).abs().mul(-1)
        return (direction_reward + speed_reward).exp()

    def calc_energy_penalty(self, next_frame):
        return 0

    def calc_action_penalty(self):
        prob_energy = self.action.abs().mean(-1, keepdim=True)
        return 0

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        if is_external_step:
            self.reward.copy_(progress)
        else:
            self.reward.add_(progress)

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def integrate_root_translation(self, pose):
        # set new target every step to make sure angle doesn't change
        super().integrate_root_translation(pose)
        self.reset_target()

    def get_observation_components(self):
        condition = self.get_vae_condition(normalize=False)
        _, target_angle = self.get_target_delta_and_angle()
        forward_speed = self.target_speed_buf * target_angle.cos()
        sideway_speed = self.target_speed_buf * target_angle.sin()
        return condition, forward_speed, sideway_speed

    def dump_additional_render_data(self):
        return {
            "extra.csv": {
                "header": "TargetSpeed, TargetFacing",
                "data": torch.cat(
                    (self.target_speed_buf[0], self.target_direction_buf[0]), dim=-1
                ),
            }
        }

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

        # if self.is_rendered and self.timestep % 15 == 0:
        #     self.viewer.duplicate_character()


class PathFollowEnv(TargetEnv):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        # controller receives 4 upcoming targets
        self.lookahead = 4
        # time gap between each lookahead frame
        # 15 frames is 0.5 seconds in real-time
        self.lookahead_skip = 4
        self.lookahead_gap = 15

        condition_size = self.frame_dim * self.num_condition_frames
        self.observation_dim = condition_size + 2 * self.lookahead

        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Define path equal to episode length
        # Should be careful about the magnitude of the path
        t = torch.linspace(0, 2 * np.pi, self.max_timestep).to(device)

        # Heart
        # scale = 4
        # x = t.sin().pow(3).mul(16 * scale)
        # y = (
        #     (1 * t).cos().mul(13 * scale)
        #     - (2 * t).cos().mul(5 * scale)
        #     - (3 * t).cos().mul(2 * scale)
        #     - (4 * t).cos().mul(1 * scale)
        # )
        # # Test Easy
        # y = (
        #     (1 * t).cos().mul(4 * scale)
        #     - (2 * t).cos().mul(3 * scale)
        #     - (3 * t).cos().mul(12 * scale)
        #     - (4 * t).cos().mul(1 * scale)
        # )
        # # Test Hard
        # y = (
        #     (1 * t).cos().mul(4 * scale)
        #     - (2 * t).cos().mul(4 * scale)
        #     - (3 * t).cos().mul(4 * scale)
        #     - (4 * t).cos().mul(16 * scale)
        # )

        # Figure 8
        scale = 50
        speed = 2
        x = scale * (speed * t).sin()
        y = scale * (speed * t).sin() * (speed * t).cos()

        # Double Figure 8
        # scale = 50
        # speed = 2
        # x = scale * t.pow(1 / 4) * (speed * t).sin()
        # y = scale * t.pow(1 / 4) * (speed * t).sin() * (speed * t).cos()

        # Figure 8 Tear Drop
        # scale = 50
        # speed = 2
        # x = scale * (speed * t).sin().pow(2)
        # y = scale * (speed * t).sin().pow(2) * (speed * t).cos()

        # Figure 8 with Stop
        # scale = 60
        # speed = 2
        # x = scale * (speed * t).sin().pow(3)
        # y = scale * (speed * t).sin().pow(3) * (speed * t).cos()

        self.path = torch.stack((x, y), dim=1)

        if self.is_rendered:
            self.viewer.add_path_markers(self.path)

    def reset(self, indices=None):
        self.path_offsets = torch.randint(
            0, self.path.size(0), (self.num_parallel, 1)
        ).long()
        return super().reset(indices)

    def reset_initial_frames(self, frame_index=None):
        super().reset_initial_frames(frame_index)

        # set initial root position to random place on path
        self.root_xz.copy_(self.path[self.path_offsets.squeeze(1)])
        next_two = (
            torch.arange(0, 2) * self.lookahead_gap
            + self.path_offsets
            + self.lookahead_skip
        ) % self.path.size(0)
        delta = self.path[next_two[:, 1]] - self.path[next_two[:, 0]]
        facing = -torch.atan2(delta[:, 1], delta[:, 0]).unsqueeze(1)
        self.root_facing.copy_(facing)

        if self.is_rendered:
            # don't forget to convert feet to meters
            centre = self.path.mean(dim=0) * 0.3048
            xyz = F.pad(centre, pad=[0, 1]).cpu().numpy()
            self.viewer.camera.lookat(xyz)

    def reset_target(self, indices=None, location=None):
        # don't add skip to accurate calculate is target is close
        index = (
            self.timestep + self.path_offsets.squeeze(1) + self.lookahead_skip
        ) % self.path.size(0)
        self.target.copy_(self.path[index])
        self.calc_potential()

        if self.is_rendered:
            self.viewer.update_target_markers(self.target)

    def get_delta_to_k_targets(self):
        # + lookahead_skip so it's not too close to character
        next_k = (
            torch.arange(0, self.lookahead) * self.lookahead_gap
            + self.timestep
            + self.path_offsets
            + self.lookahead_skip
        ) % self.path.size(0)
        # (np x lookahead x 2) - (np x 1 x 2)
        target_delta = self.path[next_k] - self.root_xz.unsqueeze(1)
        # Should be negative because going from global to local
        mat = self.get_rotation_matrix(-self.root_facing)
        # (np x 1 x 2 x 2) x (np x lookahead x 1 x 2)
        delta = (mat.unsqueeze(1) * target_delta.unsqueeze(2)).sum(dim=-1)
        return delta

    def get_observation_components(self):
        deltas = self.get_delta_to_k_targets()
        condition = self.get_vae_condition(normalize=False)
        return condition, deltas.flatten(start_dim=1, end_dim=2)

    def dump_additional_render_data(self):
        return {
            "extra.csv": {"header": "Target.X, Target.Z", "data": self.target[0]},
            "root0.csv": {
                "header": "Root.X, Root.Z, RootFacing",
                "data": torch.cat((self.root_xz, self.root_facing), dim=-1)[0],
            },
        }

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        progress = self.calc_progress_reward()

        if is_external_step:
            self.reward.copy_(progress)
        else:
            self.reward.add_(progress)

        # Check if target is reached
        # Has to be done after new potentials are calculated
        target_dist = -self.linear_potential
        target_is_close = target_dist < 2.0
        self.reward.add_(target_is_close.float() * 2.0)

        energy_penalty = self.calc_energy_penalty(next_frame)
        self.reward.add_(energy_penalty * 0.75)

        action_penalty = self.calc_action_penalty()
        self.reward.add_(action_penalty * 0.5)

        # Need to reset target to next point in path
        # can only do this after progress is calculated
        self.reset_target()

        obs_components = self.get_observation_components()
        self.done.fill_(self.timestep >= self.max_timestep)

        # Everytime this function is called, should call render
        # otherwise the fps will be wrong
        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

        # if self.is_rendered and self.timestep % 15 == 0:
        #     self.viewer.duplicate_character()


class HumanMazeEnv(EnvBase):
    def __init__(
        self,
        num_parallel,
        device,
        pose_vae_path,
        rendered=False,
        use_params=False,
        camera_tracking=True,
        frame_skip=1,
    ):
        super().__init__(
            num_parallel,
            device,
            pose_vae_path,
            rendered,
            use_params,
            camera_tracking,
            frame_skip,
        )

        basepath = os.path.normpath if os.path.isdir(pose_vae_path) else os.path.dirname
        policy_path = os.path.join(basepath(pose_vae_path), "con_TargetEnv-v0.pt")
        self.target_controller = torch.load(policy_path, map_location=self.device).actor

        self.action_dim = 2
        high = np.inf * np.ones([self.action_dim])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.max_timestep = 1500
        self.max_reward = 2048.0
        self.arena_bounds = (-100.0, 100.0)
        self.ep_reward = torch.zeros_like(self.reward)

        # coverage to encourage exploration
        map_shape = (self.num_parallel, 1024)
        self.coverage_map = torch.zeros(map_shape).bool().to(self.device)
        self.scale = np.sqrt(self.coverage_map.size(-1)) / (
            self.arena_bounds[1] - self.arena_bounds[0]
        )

        # Simple vision system
        limit = 60 / 180 * np.pi
        self.vision_distance = 100
        self.num_eyes = 16
        self.vision = torch.empty((self.num_parallel, self.num_eyes, 1)).to(self.device)
        self.vision.fill_(self.vision_distance)
        self.fov = torch.linspace(-limit, limit, self.num_eyes).to(self.device)

        # Overwrite, always start with same pose
        self.start_indices.fill_(0)

        base_obs_dim = (
            # condition + vision + coverage_map + normalized_root
            (self.frame_dim * self.num_condition_frames)
            + self.num_eyes
        )

        self.observation_dim = base_obs_dim
        high = np.inf * np.ones([self.observation_dim])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.create_simulation_world()

    def create_simulation_world(self):

        w2 = (self.arena_bounds[1] - self.arena_bounds[0]) / 2
        w4 = w2 / 2
        w8 = w4 / 2

        self.walls_start = torch.tensor(
            [
                [-w2, +w2],  # top left
                [+w2, +w2],  # top right
                [+w2, -w2],  # bottom right
                [-w2, -w2],  # bottom left
                [-w2, -w4 - w8],
                [-w4 - w8, -w4],
                [-w4 - w8, -w4],
                [-w4, -w8],
                [w4 + w8, -w8],
                [w4 + w8, -w8],
                [0, -w8],
                [w4 + w8, w4 + w8],
                [+w4, 0],
                [+w8, 0],
                [0, 0],
                [-w8, 0],
            ]
        )
        self.walls_end = torch.tensor(
            [
                [+w2, +w2],  # top right
                [+w2, -w2],  # bottom right
                [-w2, -w2],  # bottom left
                [-w2, +w2],  # top left
                [w4 + w8, -w4 - w8],
                [w2, -w4],
                [-w4 - w8, w2 - w8],
                [-w4, w2 - w8],
                [w4 + w8, w2 - w8],
                [w8, -w8],
                [-w4, -w8],
                [-w4, w4 + w8],
                [+w4, w4],
                [+w8, w4],
                [0, w4],
                [-w8, w4],
            ]
        )
        self.wall_thickness = 0.5

        if self.is_rendered:
            from common.bullet_objects import Rectangle, VSphere

            # Disable rendering during creation
            self.viewer._p.configureDebugVisualizer(
                self.viewer._p.COV_ENABLE_RENDERING, 0
            )

            half_height = 3.0

            for start, end in zip(self.walls_start, self.walls_end):
                delta = end - start
                half_length = delta.norm() / 2

                wall = Rectangle(
                    self.viewer._p,
                    half_length * FOOT2METER,
                    self.wall_thickness * FOOT2METER,
                    half_height * FOOT2METER,
                    max=True,
                    replica=1,
                )

                centre = F.pad((start + end) / 2, pad=[0, 1], value=half_height)
                centre *= FOOT2METER
                direction = float(torch.atan2(delta[1], delta[0]))
                quat = self.viewer._p.getQuaternionFromEuler([0, 0, direction])
                wall.set_positions([centre.numpy()], [quat])

            # re-enable rendering
            self.viewer._p.configureDebugVisualizer(
                self.viewer._p.COV_ENABLE_RENDERING, 1
            )

            # Need to do this else reset() doesn't work
            self.viewer.state_id = self.viewer._p.saveState()

        # Save to GPU for faster calculation later
        self.walls_start = self.walls_start.to(self.device)
        self.walls_end = self.walls_end.to(self.device)
        self.walls_mid = (self.walls_start + self.walls_end) / 2
        self.walls_hl = (self.walls_start - self.walls_end).norm(dim=-1) / 2
        delta = self.walls_start - self.walls_end
        self.walls_direction = torch.atan2(delta[:, 1], delta[:, 0])

    def reset_initial_frames(self, indices=None):

        # Newer has smaller index (ex. index 0 is newer than 1)
        condition_range = (
            self.start_indices.repeat((self.num_condition_frames, 1)).t()
            + torch.arange(self.num_condition_frames - 1, -1, -1).long()
        )

        condition = self.mocap_data[condition_range]

        if indices is None:
            self.history[:, : self.num_condition_frames].copy_(condition)
        else:
            self.history[:, : self.num_condition_frames].index_copy_(
                dim=0, index=indices, source=condition[indices]
            )

    def reset_root_state(self, indices=None, deterministic=False):
        reset_bound = (0.9 * self.arena_bounds[0], 0.9 * self.arena_bounds[1])
        if indices is None:
            if deterministic:
                self.root_facing.fill_(0)
                self.root_xz.fill_(0)
            else:
                self.root_facing.uniform_(0, 2 * np.pi)
                self.root_xz.uniform_(*reset_bound)
        else:
            if deterministic:
                self.root_facing.index_fill_(dim=0, index=indices, value=0)
                self.root_xz.index_fill_(dim=0, index=indices, value=0)
            else:
                new_facing = self.root_facing[indices].uniform_(0, 2 * np.pi)
                new_xz = self.root_xz[indices].uniform_(*reset_bound)
                self.root_facing.index_copy_(dim=0, index=indices, source=new_facing)
                self.root_xz.index_copy_(dim=0, index=indices, source=new_xz)

        if not deterministic:
            x, y, _ = extract_joints_xyz(self.history[:, 0], *self.joint_indices, dim=1)
            while True:
                joints_pos = self.root_xz.unsqueeze(1) + torch.stack((x, y), dim=-1)
                collision = (
                    self.calc_collision_with_walls(joints_pos)
                    .any(dim=-1, keepdim=True)
                    .squeeze(-1)
                )
                if collision.any():
                    new_xz = self.root_xz[collision].uniform_(*self.arena_bounds)
                    self.root_xz[collision] = new_xz
                else:
                    break


    def reset(self, indices=None):
        if indices is None:
            self.timestep = 0
            self.substep = 0
            self.root_facing.fill_(0)
            self.reward.fill_(0)
            self.ep_reward.fill_(0)
            self.done.fill_(False)
            self.coverage_map.fill_(False)

            # value bigger than contact_threshold
            self.foot_pos_history.fill_(1)
        else:
            self.root_facing.index_fill_(dim=0, index=indices, value=0)
            self.reward.index_fill_(dim=0, index=indices, value=0)
            self.ep_reward.index_fill_(dim=0, index=indices, value=0)
            self.done.index_fill_(dim=0, index=indices, value=False)
            self.coverage_map.index_fill_(dim=0, index=indices, value=False)

            # value bigger than contact_threshold
            self.foot_pos_history.index_fill_(dim=0, index=indices, value=1)

        self.reset_initial_frames(indices)
        self.reset_root_state(indices)

        self.calc_vision_state()
        obs_components = self.get_observation_components()
        return torch.cat(obs_components, dim=1)

    def step(self, action: torch.Tensor):
        # This one is arena size
        hl_action = action * 40.0
        condition = self.get_vae_condition(normalize=False)
        with torch.no_grad():
            ll_action = self.target_controller(torch.cat((condition, hl_action), dim=1))
            ll_action *= self.action_scale
        next_frame = self.get_vae_next_frame(ll_action)
        state = self.calc_env_state(next_frame[:, 0])
        return state

    def get_observation_components(self):
        condition = self.get_vae_condition(normalize=False)
        vision = self.vision.flatten(start_dim=1, end_dim=2)
        base_obs_component = (condition, vision)
        return base_obs_component

    def dump_additional_render_data(self):
        return {
            "root0.csv": {
                "header": "Root.X, Root.Z, RootFacing",
                "data": torch.cat((self.root_xz, self.root_facing), dim=-1)[0],
            },
            "walls.csv": {
                "header": "Wall.X, Wall.Z, Wall.HalfLength, Wall.Angle",
                "data": torch.cat(
                    (
                        self.walls_mid,
                        self.walls_hl.unsqueeze(-1),
                        self.walls_direction.unsqueeze(-1),
                    ),
                    dim=-1,
                ),
                "once": True,
            },
        }

    def calc_collision_with_walls(self, joints_pos, radius=0.01):
        # (num_character, num_walls, num_joins, 2)
        p1 = self.walls_start[None, :, None, :]
        p2 = self.walls_end[None, :, None, :]
        c = joints_pos.unsqueeze(1)
        # size of each joint is defined in mocap_renderer
        # need to account for joint size and wall thickness
        d, mask = line_to_point_distance(p1, p2, c)
        mask = mask * (d < (self.wall_thickness + radius * METER2FOOT))
        return mask.any(dim=-1).any(dim=-1, keepdim=True)

    def calc_distance_to_walls(self):
        angles = (self.fov - self.root_facing).unsqueeze(-1)
        directions = torch.cat([angles.cos(), angles.sin()], dim=2)

        # (num_character, num_eyes, num_walls, 2)
        p0 = self.walls_start[None, None, :, :]
        p1 = self.walls_end[None, None, :, :]
        p2 = self.root_xz[:, None, None, :]
        p3 = p2 + self.vision_distance * directions.unsqueeze(2)

        p3p2 = p3 - p2
        p1p0 = p1 - p0

        a = p3p2.select(-1, 0) * p1p0.select(-1, 0)
        b = p3p2.select(-1, 0) * p1p0.select(-1, 1)
        c = p3p2.select(-1, 1) * p1p0.select(-1, 0)

        d = a * (p0.select(-1, 1) - p2.select(-1, 1))
        e = b * p0.select(-1, 0)
        f = c * p2.select(-1, 0)

        x = (d - e + f) / (c - b)
        m = p3p2.select(-1, 1) / p3p2.select(-1, 0)
        y = m * x + (p2.select(-1, 1) - m * p2.select(-1, 0))

        solution = torch.stack((x, y), dim=-1)

        vector1 = -self.vision_distance * directions.unsqueeze(2)
        vector2 = solution - p3
        vector3 = solution - p2
        cosine1 = (vector2 * vector1).sum(dim=-1)
        cosine2 = (vector3 * -vector1).sum(dim=-1)

        distance = vector3.norm(dim=-1)
        mask = (
            ((solution - self.walls_mid).norm(dim=-1) < self.walls_hl)
            * (vector3.norm(dim=-1) < self.vision_distance)
            * (cosine1 > 0)
            * (cosine2 > 0)
        )

        self.vision[:, :, 0].copy_(
            distance.masked_fill(~mask, self.vision_distance).min(dim=-1)[0]
        )

    def calc_coverage_reward(self):
        # Exploration reward
        normalized_xz = (self.root_xz - self.arena_bounds[0]) * self.scale
        map_coordinate = (
            (
                normalized_xz[:, 0] * (np.sqrt(self.coverage_map.size(-1)) - 1)
                + normalized_xz[:, 1]
            )
            .long()
            .clamp(min=0, max=self.coverage_map.size(1) - 1)
        )

        old_coverage_count = self.coverage_map.sum(dim=-1)
        self.coverage_map[self.parallel_ind_buf, map_coordinate] = True
        coverage_bonus = self.coverage_map.sum(dim=-1) - old_coverage_count
        self.reward.add_(coverage_bonus.float().unsqueeze(-1) * 0.5)

    def calc_vision_state(self):
        # Two functions calculate intersection differently
        self.calc_distance_to_walls()
        # self.draw_debug_lines()
        # Walls can block line of sight, others cannot
        # mask = self.vision[:, :, [0]] < self.vision
        # self.vision[mask] = self.vision_distance

    def calc_env_state(self, next_frame):
        self.next_frame = next_frame
        is_external_step = self.substep == 0

        if self.substep == self.frame_skip - 1:
            self.timestep += 1
        self.substep = (self.substep + 1) % self.frame_skip

        self.integrate_root_translation(next_frame)

        # (num_character, 1, 2) - (num_character, num_pellets, 2)
        x, y, _ = extract_joints_xyz(next_frame, *self.joint_indices, dim=1)
        joints_pos = self.root_xz.unsqueeze(1) + torch.stack((x, y), dim=-1)

        collision = self.calc_collision_with_walls(joints_pos).any(dim=-1, keepdim=True)
        self.calc_vision_state()

        self.reward.fill_(0)

        self.calc_coverage_reward()

        obs_components = self.get_observation_components()
        self.done.copy_(collision + (self.timestep >= self.max_timestep))

        self.render()

        return (
            torch.cat(obs_components, dim=1),
            self.reward,
            self.done,
            {"reset": self.timestep >= self.max_timestep},
        )

    def draw_debug_lines(self):
        if not self.is_rendered:
            return

        ray_from = self.root_xz.unsqueeze(1).expand(
            self.num_parallel, self.num_eyes, self.root_xz.size(-1)
        )

        angles = (self.fov - self.root_facing).unsqueeze(-1)
        directions = torch.cat([angles.cos(), angles.sin()], dim=2)
        deltas = self.vision.min(dim=-1)[0].unsqueeze(-1) * directions
        ray_to = ray_from + deltas

        ray_from = (
            (F.pad(ray_from, pad=[0, 1], value=3) * FOOT2METER)
            .flatten(0, 1)
            .cpu()
            .numpy()
        )
        ray_to = (
            (F.pad(ray_to, pad=[0, 1], value=3) * FOOT2METER)
            .flatten(0, 1)
            .cpu()
            .numpy()
        )

        if not hasattr(self, "ray_ids"):
            self.ray_ids = [
                self.viewer._p.addUserDebugLine((0, 0, 0), (1, 0, 0), (1, 0, 0))
                for i in range(self.num_parallel * self.num_eyes)
            ]

        for i, (start, end, dist) in enumerate(
            zip(ray_from, ray_to, self.vision.min(dim=-1)[0].flatten())
        ):
            rayHitColor = [1, 0, 0]
            rayMissColor = [0, 1, 0]
            colour = rayHitColor if dist < self.vision_distance else rayMissColor
            self.viewer._p.addUserDebugLine(
                start, end, colour, replaceItemUniqueId=self.ray_ids[i]
            )

    def render(self, mode="human"):
        if self.is_rendered:
            self.viewer.render(
                self.history[:, 0],  # 0 is the newest
                self.root_facing,
                self.root_xz,
                0.0,  # No time in this env
                self.action,
            )

        # if self.is_rendered and self.timestep % 15 == 0:
        #     self.viewer.duplicate_character()
