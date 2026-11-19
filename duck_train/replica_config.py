"""Everything replica-specific, in one place.

Each value says where it came from. "Replica" means the community replica repo
AI-FanGe/Microduck-build-tutorial (cloned read-only at external/Microduck-build-tutorial,
commit 4967821). Paths below are relative to that repo.

The replica task (duck_train/replica_task.py) reads ONLY these values; everything else
comes unchanged from upstream's walking task (third_party/microduck_rl,
Mjlab-Velocity-Flat-MicroDuck).
"""

# ---------------------------------------------------------------------------
# Battery / servo supply voltage  ---  PLACEHOLDER, the battery is not chosen yet.
# ---------------------------------------------------------------------------
# The ONE setting: the pack's working voltage range [V], (min, max). Each training
# environment draws its supply voltage uniformly from this range (BAM vin_range).
#
# Placeholder values from the replica's own constants:
#   max 7.5 V = BAM_VIN      (microduck/src/constants.py:56, used by its sim,
#                             microduck/src/sim/mujoco_controller.py:118)
#   min 6.0 V = BAM_VIN_MIN  (microduck/src/constants.py:57)
# NOTE the replica is not self-consistent: its README says the servos run from a
# "ready-made 6V rechargeable battery" (README.md:52 and :607), which does not fit a
# 7.5 V nominal. Replace this with the real pack's range once it is chosen.
# Upstream trains with (6.5, 8.2) (src/mjlab_microduck/robot/microduck_constants.py:126).
PACK_VOLTAGE_RANGE_V: tuple[float, float] = (6.0, 7.5)  # PLACEHOLDER

# Floor on the supply voltage after load sag (BAM vin_min). Upstream uses 6.0 V
# (microduck_constants.py:128), which is also the replica's BAM_VIN_MIN
# (constants.py:57). It is never allowed above the pack minimum, so changing only
# PACK_VOLTAGE_RANGE_V to a lower-voltage pack still gives a valid configuration.
SAG_FLOOR_V: float = min(6.0, PACK_VOLTAGE_RANGE_V[0])

# ---------------------------------------------------------------------------
# Servo stiffness while walking
# ---------------------------------------------------------------------------
# Dynamixel "Position P Gain" register value written to every servo when the walk
# starts: KP_RL = 125 (microduck/src/constants.py:52, written at
# microduck/src/moves/walk.py:87). Upstream trains with kp_fw = 200
# (microduck_constants.py:124). In BAM's XL330 model kp is the firmware register value
# (duty = error_gain * kp * position_error), not a fitted parameter, so it can be set
# directly (bam/mjlab.py BamActuatorCfg.kp_fw).
SERVO_KP_REGISTER: float = 125.0

# ---------------------------------------------------------------------------
# IMU mounting (vertical BNO08x)
# ---------------------------------------------------------------------------
# Quaternion (w, x, y, z): "Rotation from trunk frame (body) to IMU sensor frame",
# IMU_MOUNT_QUAT (microduck/src/constants.py:94). The replica's own training uses
# the same value, BNO080_MOUNT_QUAT
# (mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:82).
# 90 degrees about the body y axis. TO BE CONFIRMED ON HARDWARE for our build.
IMU_MOUNT_QUAT: tuple[float, float, float, float] = (0.7071068, 0.0, 0.7071068, 0.0)

# ---------------------------------------------------------------------------
# Policy observation layout (51 entries)
# ---------------------------------------------------------------------------
# Order built by WalkMove.build_observation (microduck/src/moves/walk.py:126-156):
#   gyro (131), projected gravity (132), joint pos - default (135-136),
#   joint vel (139-140), last action (143), command vx, vy, vtheta (146-148).
# Phase inputs (150-154) are only appended when the ONNX input is wider than 51
# (walk.py:53-54); we train without them.
# Names are the mjlab observation-term names in the replica task.
POLICY_OBS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),       # gyro, sensor frame
    ("projected_gravity", 3),  # world UP vector in the sensor frame (see replica_task.py)
    ("joint_pos", 14),
    ("joint_vel", 14),
    ("actions", 14),
    ("command", 3),
)
POLICY_OBS_SIZE: int = sum(n for _, n in POLICY_OBS_LAYOUT)  # 51

# Joint order of the 14 servos in observations and actions,
# OBSERVATION_DOF_ORDER (microduck/src/constants.py:97-112). walk.py:42-46 refuses
# an ONNX whose joint_names metadata differs from this.
JOINT_ORDER: tuple[str, ...] = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)

# ---------------------------------------------------------------------------
# Head / neck joints without head commands
# ---------------------------------------------------------------------------
# With no head command, the replica's training keeps the 4 neck/head joints as
# ordinary policy outputs and pulls them to the home pose through the posture reward,
# with these Gaussian widths [rad]
# (mjlab_microduck/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py:237-255).
HEAD_POSE_STD_STANDING: float = 0.08
HEAD_POSE_STD_WALKING: float = 0.12

# Task name and log folder (logs/rsl_rl/<experiment>/ under the current directory).
TASK_ID: str = "Mjlab-Velocity-Flat-MicroDuck-Replica"
EXPERIMENT_NAME: str = "velocity_replica"
