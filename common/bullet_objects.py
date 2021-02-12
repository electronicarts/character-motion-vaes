import os

current_dir = os.path.dirname(os.path.realpath(__file__))

import numpy as np

DEG2RAD = np.pi / 180


class VSphere:
    def __init__(
        self,
        bc,
        radius=None,
        pos=None,
        rgba=None,
        max=False,
        collide=False,
        flags=0,
        replica=1,
    ):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else tuple(rgba)

        shape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            specularColor=(0.4, 0.4, 0),
        )

        if not collide:
            self.id = self._p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=shape,
                basePosition=pos,
                batchPositions=[pos for _ in range(replica)],
                useMaximalCoordinates=max,
                flags=flags,
            )
        else:
            cshape = self._p.createCollisionShape(self._p.GEOM_SPHERE, radius=radius)
            self.id = self._p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cshape,
                baseVisualShapeIndex=shape,
                basePosition=pos,
                batchPositions=[pos for _ in range(replica)],
                useMaximalCoordinates=max,
                flags=flags,
            )

        # Need this otherwise batchPositions does not work
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)

        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba

    def set_positions(self, pos, index=None):
        if index is None:
            for index, id in enumerate(self.ids):
                self._p.resetBasePositionAndOrientation(
                    id, posObj=pos[index], ornObj=(0, 0, 0, 1)
                )
        else:
            id = self.ids[index]
            self._p.resetBasePositionAndOrientation(id, posObj=pos, ornObj=(0, 0, 0, 1))

    def set_position(self, pos):
        self._p.resetBasePositionAndOrientation(
            self.id[0], posObj=pos, ornObj=(0, 0, 0, 1)
        )

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id[0], -1, rgbaColor=rgba)
            self._rgba = t_rgba


class VCylinder:
    def __init__(
        self,
        bc,
        radius=None,
        height=None,
        pos=None,
        rgba=None,
        max=False,
        flags=0,
        replica=1,
    ):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        radius = 0.3 if radius is None else radius
        height = 1.0 if height is None else height
        pos = (0, 0, height / 2) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else tuple(rgba)

        shape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=pos,
            batchPositions=[pos for _ in range(replica)],
            useMaximalCoordinates=max,
            flags=flags,
        )

        # Need this otherwise batchPositions does not work?
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)

        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba

    def set_positions(self, pos, orn):
        for index, id in enumerate(self.ids):
            self._p.resetBasePositionAndOrientation(
                id, posObj=pos[index], ornObj=orn[index]
            )

    def set_position(self, pos, orn):
        self._p.resetBasePositionAndOrientation(self.id[0], posObj=pos, ornObj=orn)

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id[0], -1, rgbaColor=rgba)
            self._rgba = t_rgba


class VCapsule:
    def __init__(
        self,
        bc,
        radius=None,
        height=None,
        pos=None,
        rgba=None,
        max=False,
        flags=0,
        replica=1,
    ):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        radius = 0.3 if radius is None else radius
        height = 1.0 if height is None else height
        pos = (0, 0, height / 2) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else tuple(rgba)

        shape = self._p.createVisualShape(
            self._p.GEOM_CAPSULE,
            radius=radius,
            length=height,
            rgbaColor=rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=pos,
            batchPositions=[pos for _ in range(replica)],
            useMaximalCoordinates=max,
            flags=flags,
        )

        # Need this otherwise batchPositions does not work?
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)

        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba
        self.height = height

    def set_positions(self, pos, orn):
        for index, id in enumerate(self.ids):
            self._p.resetBasePositionAndOrientation(
                id, posObj=pos[index], ornObj=orn[index]
            )

    def set_position(self, pos, orn):
        self._p.resetBasePositionAndOrientation(self.id[0], posObj=pos, ornObj=orn)

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id[0], -1, rgbaColor=rgba)
            self._rgba = t_rgba


class FlagPole:
    def __init__(self, bc, pos=None, rgba=None, replica=1):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        self._pos = np.asarray(pos) if pos is not None else np.array([0, 0, 0])
        self._rgba = tuple(rgba) if rgba is not None else (1, 0, 0, 1)
        self._quat = [1, 0, 0, 1]

        base_path = os.path.join(current_dir, "data", "misc")
        pole_path = os.path.join(base_path, "pole.obj")

        shape = self._p.createVisualShapeArray(
            shapeTypes=[self._p.GEOM_MESH, self._p.GEOM_BOX],
            halfExtents=[[0, 0, 0], [0.01, 0.3, 0.2]],
            fileNames=[pole_path, ""],
            rgbaColors=[self._rgba, self._rgba],
            visualFramePositions=[[0, 0, 0], [0, 1.12, 0]],
            meshScales=[[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],
        )

        id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            basePosition=self._pos,
            baseOrientation=self._quat,
            batchPositions=[self._pos for _ in range(replica)],
            useMaximalCoordinates=True,
        )

        self.id = id[0]

        text_path = os.path.join(base_path, "zink.jpg")
        texture = self._p.loadTexture(text_path)
        # don't need to change for every id for some reason
        self._p.changeVisualShape(self.id, -1, textureUniqueId=texture)

        # Need this otherwise batchPositions does not work
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)


class Arrow:
    def __init__(self, bc, pos=None, rgba=None, replica=1):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        self._pos = np.asarray(pos) if pos is not None else np.array([0, 0, 0])
        self._rgba = (
            tuple(rgba) if rgba is not None else (10 / 255, 60 / 255, 150 / 255, 1)
        )
        self._quat = [0, 0, 0, 1]

        base_path = os.path.join(current_dir, "data", "misc")
        pole_path = os.path.join(base_path, "arrow.obj")

        shape = self._p.createVisualShape(
            shapeType=self._p.GEOM_MESH, fileName=pole_path, rgbaColor=self._rgba
        )

        self.id = self._p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=shape,
            baseInertialFramePosition=(0, 1, 0),
            baseInertialFrameOrientation=(0, 0, -1, 1),
            batchPositions=[self._pos for _ in range(replica)],
            useMaximalCoordinates=True,
        )

        # Need this otherwise batchPositions does not work
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)


class Rectangle:
    def __init__(
        self, bc, hdx, hdy, hdz, mass=0.0, pos=None, rgba=None, max=False, replica=1
    ):
        self._p = bc

        # create all spheres at once using batchPositions
        old_num_bodies = self._p.getNumBodies()

        dims = np.array([hdx, hdy, hdz], dtype=np.float32)

        pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos
        rgba = (55 / 255, 55 / 255, 55 / 255, 1) if rgba is None else rgba

        self._pos = pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0])

        box_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=dims)
        box_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=dims,
            rgbaColor=rgba,
            specularColor=(0.4, 0.4, 0),
        )

        self.id = self._p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_vshape,
            basePosition=self._pos,
            batchPositions=[pos for _ in range(replica)],
            useMaximalCoordinates=max,
        )

        # Need this otherwise batchPositions does not work
        self._p.syncBodyInfo()
        new_num_bodies = self._p.getNumBodies()
        self.ids = range(old_num_bodies, new_num_bodies)

    def set_positions(self, pos, quat=None, index=None):
        if index is None:
            for index, id in enumerate(self.ids):
                orn = (0, 0, 0, 1) if quat is None else quat[index]
                self._p.resetBasePositionAndOrientation(
                    id, posObj=pos[index], ornObj=orn
                )
        else:
            id = self.ids[index]
            orn = (0, 0, 0, 1) if quat is None else quat
            self._p.resetBasePositionAndOrientation(id, posObj=pos, ornObj=orn)

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id, -1, rgbaColor=rgba)
            self._rgba = t_rgba
