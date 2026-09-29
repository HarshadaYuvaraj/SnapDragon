"""Optional PyBullet check: a Franka Panda executes the synthesized grasp (approach, close, lift).

Requires `pip install pybullet` (no official Windows-on-ARM64 wheel; run it on any x64/Linux/
macOS box using the saved grasps.json, or use the built-in kinematic GIF on the Snapdragon PC).
The robot base frame equals `StereoRig.T_base_cam`'s target frame (table plane z = 0).
"""
import time
import numpy as np


def _quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()          # x, y, z, w (PyBullet order)


def run(res, rig, gui=False, hold=1.0):
    import pybullet as p
    import pybullet_data
    T = rig.T_base_cam
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81); p.setTimeStep(1 / 240)
    p.loadURDF("plane.urdf")
    arm = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    home = [0, -0.4, 0, -2.1, 0, 1.8, 0.8]
    for j, q in enumerate(home):
        p.resetJointState(arm, j, q)

    # spawn one box per detected object (footprint from cluster extent in the table plane)
    ids = {}
    for c in res.clusters:
        pts = c.points @ T[:3, :3].T + T[:3, 3]
        lo, hi = pts.min(0), pts.max(0)
        half = [max((hi[0] - lo[0]) / 2, 0.01), max((hi[1] - lo[1]) / 2, 0.01), c.top_height / 2]
        ctr = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, c.top_height / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.3, 0.6, 0.9, 1])
        ids[c.id] = p.createMultiBody(0.15, col, vis, ctr)
        p.changeDynamics(ids[c.id], -1, lateralFriction=1.0)
    for _ in range(120):
        p.stepSimulation()

    g = res.best.transformed(T)
    a, z = g.R[:, 0], g.R[:, 2]
    R_ee = np.stack([np.cross(a, z), a, z], 1)         # Panda fingers close along the hand's y axis
    q_ee = _quat(R_ee)
    EE = 11

    def move(pos, steps=240, fingers=None):
        for _ in range(steps):
            q = p.calculateInverseKinematics(arm, EE, pos, q_ee, maxNumIterations=100)
            for j in range(7):
                p.setJointMotorControl2(arm, j, p.POSITION_CONTROL, q[j], force=200)
            if fingers is not None:
                for j in (9, 10):
                    p.setJointMotorControl2(arm, j, p.POSITION_CONTROL, fingers, force=60)
            p.stepSimulation()
            if gui:
                time.sleep(1 / 240)

    above = g.center + np.array([0, 0, 0.15])
    move(above, fingers=(g.width + 0.03) / 2)
    move(g.center, fingers=(g.width + 0.03) / 2)
    move(g.center, 120, fingers=max(g.width / 2 - 0.004, 0.0))       # close
    z0 = p.getBasePositionAndOrientation(ids[g.object_id])[0][2]
    move(g.center + np.array([0, 0, 0.15]), 300, fingers=max(g.width / 2 - 0.004, 0.0))   # lift
    z1 = p.getBasePositionAndOrientation(ids[g.object_id])[0][2]
    if gui and hold:
        time.sleep(hold)
    p.disconnect(cid)
    return dict(lifted_m=float(z1 - z0), success=bool(z1 - z0 > 0.05))
