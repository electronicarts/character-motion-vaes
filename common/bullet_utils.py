from __future__ import absolute_import
from __future__ import division

import functools
import inspect
import os
import time

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import pybullet

FOOT2METER = 0.3048
METER2FOOT = 1 / 0.3048


class BulletClient(object):
    """A wrapper for pybullet to manage different clients."""

    def __init__(self, connection_mode=None):
        """Creates a Bullet client and connects to a simulation.

    Args:
      connection_mode:
        `None` connects to an existing simulation or, if fails, creates a
          new headless simulation,
        `pybullet.GUI` creates a new simulation with a GUI,
        `pybullet.DIRECT` creates a headless simulation,
        `pybullet.SHARED_MEMORY` connects to an existing simulation.
    """
        if connection_mode is None:
            self._client = pybullet.connect(pybullet.SHARED_MEMORY)
            if self._client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT
        self._client = pybullet.connect(
            connection_mode,
            options=(
                "--background_color_red=0.2 "
                "--background_color_green=0.2 "
                "--background_color_blue=0.2"
            ),
        )

    def __del__(self):
        """Clean up connection if not already done."""
        try:
            pybullet.disconnect(physicsClientId=self._client)
        except pybullet.error:
            pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute):
            if name not in [
                "invertTransform",
                "multiplyTransforms",
                "getMatrixFromQuaternion",
                "getEulerFromQuaternion",
                "computeViewMatrixFromYawPitchRoll",
                "computeProjectionMatrixFOV",
                "getQuaternionFromEuler",
            ]:  # A temporary hack for now.
                attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class Pose_Helper:  # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def state_fields_of_pose_of(
        self, body_id, link_id=-1
    ):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = self._p.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_position(self):
        return self.current_position()

    def get_pose(self):
        return self.state_fields_of_pose_of(
            self.bodies[self.bodyIndex], self.bodyPartIndex
        )

    def angular_speed(self):
        if self.bodyPartIndex == -1:
            _, (vr, vp, vy) = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            _, _, _, _, _, _, _, (vr, vp, vy) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
            )
        return np.array([vr, vp, vy])

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (
                vr,
                vp,
                vy,
            ) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
            )
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def get_orientation(self):
        return self.current_orientation()

    def reset_position(self, position):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex], position, self.get_orientation()
        )

    def reset_orientation(self, orientation):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex], self.get_position(), orientation
        )

    def reset_velocity(self, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]):
        self._p.resetBaseVelocity(
            self.bodies[self.bodyIndex], linearVelocity, angularVelocity
        )

    def reset_pose(self, position, orientation):
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.bodyIndex], position, orientation
        )

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return self._p.getContactPoints(
            bodyA=self.bodies[self.bodyIndex], linkIndexA=self.bodyPartIndex
        )


class Joint:
    def __init__(
        self, bullet_client, joint_name, bodies, bodyIndex, jointIndex, torque_limit=0
    ):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name
        self.torque_limit = torque_limit

        jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.lowerLimit = jointInfo[8]
        self.upperLimit = jointInfo[9]

        self.power_coeff = 0

    def set_torque_limit(self, torque_limit):
        self.torque_limit = torque_limit

    def set_state(self, x, vx):
        self._p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self):
        return self.get_state()

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), vel)

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(
            self.bodies[self.bodyIndex], self.jointIndex
        )
        return x, vx

    def get_position(self):
        x, _ = self.get_state()
        return x

    def get_orientation(self):
        _, r = self.get_state()
        return r

    def get_velocity(self):
        _, vx = self.get_state()
        return vx

    def set_position(self, position):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            pybullet.POSITION_CONTROL,
            targetPosition=position,
            force=self.torque_limit,
        )

    def set_velocity(self, velocity):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            pybullet.VELOCITY_CONTROL,
            targetVelocity=velocity,
        )

    def set_motor_torque(self, torque):
        self.set_torque(torque)

    def set_torque(self, torque):
        self._p.setJointMotorControl2(
            bodyIndex=self.bodies[self.bodyIndex],
            jointIndex=self.jointIndex,
            controlMode=pybullet.TORQUE_CONTROL,
            force=torque,
        )

    def reset_current_position(self, position, velocity):
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self._p.resetJointState(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            targetValue=position,
            targetVelocity=velocity,
        )
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0,
        )


class Scene:
    "A base class for single- and multiplayer scenes"

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.timestep = timestep
        self.frame_skip = frame_skip

        self.dt = self.timestep * self.frame_skip
        self.cpp_world = World(self._p, gravity, timestep, frame_skip)

        self.test_window_still_open = True
        self.human_render_detected = False

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer:
            return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def set_physics_parameters(self):
        "This function gets overridden by specific scene, to reset specific objects into their start positions"
        self.cpp_world.set_physics_parameters()

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self.cpp_world.step()


class World:
    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.set_physics_parameters()

    def set_physics_parameters(self):
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=self.numSolverIterations,
            numSubSteps=self.frame_skip,
        )

    def step(self):
        self._p.stepSimulation()


class StadiumScene(Scene):

    stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID

    def initialize(self):
        current_dir = os.path.dirname(__file__)
        filename = os.path.join(current_dir, "data", "misc", "plane_stadium.sdf")
        self.ground_plane_mjcf = self._p.loadSDF(filename, useMaximalCoordinates=True)
        # self.ground_plane_mjcf = (0,)

        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=0.8, restitution=0.5)
            self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])


class SinglePlayerStadiumScene(StadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False


class Camera:
    def __init__(self, bc, fps=60, dist=2.5, yaw=0, pitch=-5):

        self._p = bc
        self._dist = dist
        self._yaw = yaw
        self._pitch = pitch
        self._coef = np.array([1.0, 1.0, 0.1])

        self._fps = fps
        self._target_period = 1 / fps
        self._counter = time.perf_counter()

        try:
            self.width, self.height = self._p.getDebugVisualizerCamera()[0:2]
        except:
            self.width, self.height = 1024, 768

        self.lookat((0, 0, 1))

    def track(self, pos, smooth_coef=None):

        try:
            smooth_coef = self._coef if smooth_coef is None else smooth_coef
            assert (smooth_coef <= 1).all(), "Invalid camera smoothing parameters"

            yaw, pitch, dist, lookat_ = self._p.getDebugVisualizerCamera()[-4:]
            lookat = (1 - smooth_coef) * lookat_ + smooth_coef * pos

            self._p.resetDebugVisualizerCamera(dist, yaw, pitch, lookat)
            self.camera_target = lookat

            # Remember camera for reset
            self._yaw, self._pitch, self._dist = yaw, pitch, dist
            self.wait()
        except:
            pass

    def lookat(self, pos):
        self.camera_target = pos
        self._p.resetDebugVisualizerCamera(self._dist, self._yaw, self._pitch, pos)

    def dump_rgb_array(self):

        lookat = [0, 0, 1]
        distance = 10

        yaw, pitch, _, _ = self._p.getDebugVisualizerCamera()[-4:]
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            lookat, distance, yaw, pitch, 0, upAxisIndex=2
        )

        (_, _, rgb_array, _, _) = self._p.getCameraImage(
            width=1920 * 2,
            height=1080 * 2,
            viewMatrix=view_matrix,
            # lightDirection=[-1, -1, -1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            # flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def dump_orthographic_rgb_array(self, lookat=[0, 0, 0]):
        distance = 20
        # yaw = 0
        # pitch = -10
        # lookat = np.array([13, 13, 0])
        # lookat *= FOOT2METER

        yaw, pitch, _, lookat = self._p.getDebugVisualizerCamera()[-4:]
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            lookat, distance, yaw, pitch, 0, upAxisIndex=2
        )

        aspect = self.width / self.height
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=20, aspect=aspect, nearVal=0.01, farVal=1000
        )

        (_, _, rgb_array, _, _) = self._p.getCameraImage(
            width=self.width * 2,
            height=self.height * 2,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=pybullet.ER_NO_SEGMENTATION_MASK,
        )

        # rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def wait(self):
        delta = time.perf_counter() - self._counter
        time.sleep(max(self._target_period - delta, 0))
        now = time.perf_counter()
        self._fps = 0.99 * self._fps + 0.01 / (now - self._counter)
        self._counter = now
