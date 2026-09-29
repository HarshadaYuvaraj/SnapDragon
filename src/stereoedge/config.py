"""Camera rig, gripper and hand-eye configuration (all lengths in metres)."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StereoRig:
    width: int = 320
    height: int = 240
    fx: float = 280.0
    fy: float = 280.0
    cx: float = 160.0
    cy: float = 120.0
    baseline: float = 0.06
    num_disp: int = 48
    # Camera -> robot base transform (camera mounted 0.55 m above the table, looking down).
    T_base_cam: np.ndarray = field(default_factory=lambda: np.array([
        [1, 0, 0, 0.50],
        [0, -1, 0, 0.00],
        [0, 0, -1, 0.55],
        [0, 0, 0, 1.0]]))

    @property
    def shape(self):
        return (self.height, self.width)


@dataclass
class GripperSpec:
    """Parallel-jaw gripper (Franka-Hand-like dimensions)."""
    max_width: float = 0.080
    min_width: float = 0.012
    finger_width: float = 0.020      # extent along the finger's long side
    finger_thickness: float = 0.010
    finger_below: float = 0.012      # fingertip extends this far below the grasp centre
    finger_above: float = 0.100      # finger + palm volume above grasp centre
    table_margin: float = 0.004
    friction_angle_deg: float = 25.0
