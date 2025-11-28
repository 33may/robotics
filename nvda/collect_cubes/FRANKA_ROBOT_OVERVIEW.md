# Franka Panda Robot - Complete Overview

## Robot Specifications

The **Franka Emika Panda** is a 7-DOF collaborative robot arm with a 2-DOF parallel gripper.

### Physical Structure

**Arm Joints (7 DOF):**
1. `panda_joint1` - Shoulder pan
2. `panda_joint2` - Shoulder lift
3. `panda_joint3` - Shoulder roll
4. `panda_joint4` - Elbow
5. `panda_joint5` - Wrist 1
6. `panda_joint6` - Wrist 2
7. `panda_joint7` - Wrist 3

**Gripper Joints (2 DOF):**
- `panda_finger_joint1` - Left finger
- `panda_finger_joint2` - Right finger

**Total DOF:** 9 (7 arm + 2 gripper)

---

## Configuration in Isaac Lab

### Available Configurations

Isaac Lab provides three pre-configured Franka setups:

1. **`FRANKA_PANDA_CFG`** - Standard Franka with Panda hand
   - Default PD gains for general manipulation
   - Gravity enabled
   - File: `/isaac/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py`

2. **`FRANKA_PANDA_HIGH_PD_CFG`** - High stiffness variant
   - Stiffer PD control (stiffness: 400, damping: 80)
   - Gravity disabled
   - Recommended for task-space control with differential IK
   - Better tracking performance for precise manipulation

3. **`FRANKA_ROBOTIQ_GRIPPER_CFG`** - Franka with Robotiq 2F-85 gripper
   - Different gripper with more complex actuation
   - Higher effort limits on arm joints
   - Multiple gripper control groups (drive, finger, passive)

---

## Default Configuration (FRANKA_PANDA_CFG)

### Initial Joint Positions

```python
{
    "panda_joint1": 0.0,      # rad
    "panda_joint2": -0.569,   # rad (~-32.6°)
    "panda_joint3": 0.0,      # rad
    "panda_joint4": -2.810,   # rad (~-161°)
    "panda_joint5": 0.0,      # rad
    "panda_joint6": 3.037,    # rad (~174°)
    "panda_joint7": 0.741,    # rad (~42.5°)
    "panda_finger_joint.*": 0.04,  # m (4cm opening)
}
```

This default pose positions the arm in a canonical "ready" configuration with the end-effector facing forward.

### Actuator Groups

The Franka is divided into **3 actuator groups** for control:

#### 1. **Shoulder Actuators** (`panda_shoulder`)
Controls joints 1-4 (shoulder and elbow)

```python
joint_names_expr: ["panda_joint[1-4]"]
effort_limit_sim: 87.0  # Nm (solver limit)
stiffness: 80.0         # Position control gain
damping: 4.0            # Velocity control gain
```

#### 2. **Forearm Actuators** (`panda_forearm`)
Controls joints 5-7 (wrist)

```python
joint_names_expr: ["panda_joint[5-7]"]
effort_limit_sim: 12.0  # Nm (solver limit)
stiffness: 80.0         # Position control gain
damping: 4.0            # Velocity control gain
```

#### 3. **Hand Actuators** (`panda_hand`)
Controls gripper fingers

```python
joint_names_expr: ["panda_finger_joint.*"]
effort_limit_sim: 200.0  # N (solver limit)
stiffness: 2000.0        # Position control gain
damping: 100.0           # Velocity control gain
```

---

## Control Methods

### 1. Position Control (Default)

Using implicit actuators with PD control:

```python
# Set target joint positions
robot.set_joint_position_target(target_positions, joint_ids=None)
robot.write_data_to_sim()
```

The actuator automatically computes:
```
τ = Kp * (q_target - q_current) - Kd * q_dot_current
```

Where:
- `Kp` = stiffness
- `Kd` = damping
- Clamped by `effort_limit_sim`

### 2. Velocity Control

```python
# Set target joint velocities
robot.set_joint_velocity_target(target_velocities, joint_ids=None)
robot.write_data_to_sim()
```

Computed as:
```
τ = -Kp * q_current + Kd * (v_target - v_current)
```

### 3. Effort (Torque) Control

For direct torque control, set stiffness and damping to zero:

```python
# In configuration
actuators={
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit_sim=87.0,
        stiffness=0.0,  # Zero for effort control
        damping=0.0,
    ),
}

# In code
robot.set_joint_effort_target(target_efforts, joint_ids=None)
robot.write_data_to_sim()
```

### 4. Hybrid Control

You can control different joints with different modes by creating separate actuator groups:

```python
actuators={
    "arm_position": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-7]"],
        stiffness=80.0,
        damping=4.0,
    ),
    "gripper_effort": ImplicitActuatorCfg(
        joint_names_expr=["panda_finger_joint.*"],
        stiffness=0.0,
        damping=0.0,
    ),
}
```

---

## State Information

### Accessing Robot State

The robot state is stored in `robot.data` (instance of `ArticulationData`):

#### Joint States
```python
# Joint positions (rad or m)
robot.data.joint_pos  # shape: (num_envs, num_joints)

# Joint velocities (rad/s or m/s)
robot.data.joint_vel  # shape: (num_envs, num_joints)

# Joint accelerations (rad/s² or m/s²)
robot.data.joint_acc  # shape: (num_envs, num_joints)

# Applied joint efforts (Nm or N)
robot.data.applied_torque  # shape: (num_envs, num_joints)
```

#### Root State (Base)
```python
# Root position in world frame (m)
robot.data.root_pos_w  # shape: (num_envs, 3)

# Root orientation in world frame (quaternion: w, x, y, z)
robot.data.root_quat_w  # shape: (num_envs, 4)

# Root linear velocity in world frame (m/s)
robot.data.root_lin_vel_w  # shape: (num_envs, 3)

# Root angular velocity in world frame (rad/s)
robot.data.root_ang_vel_w  # shape: (num_envs, 3)
```

#### Body States
```python
# All body positions in world frame
robot.data.body_pos_w  # shape: (num_envs, num_bodies, 3)

# All body orientations in world frame
robot.data.body_quat_w  # shape: (num_envs, num_bodies, 4)

# All body linear velocities
robot.data.body_lin_vel_w  # shape: (num_envs, num_bodies, 3)

# All body angular velocities
robot.data.body_ang_vel_w  # shape: (num_envs, num_bodies, 3)
```

#### Jacobians and Dynamics
```python
# Jacobian matrix for specified body
robot.data.body_jacobian_w  # shape: (num_envs, num_bodies, 6, num_dof)

# Mass matrix
robot.data.mass_matrix  # shape: (num_envs, num_dof, num_dof)

# Gravity vector
robot.data.gravity_force  # shape: (num_envs, num_dof)

# Coriolis and centrifugal forces
robot.data.coriolis_force  # shape: (num_envs, num_dof)
```

### Finding Specific Joints/Bodies

```python
# Find joint indices by name
joint_ids, joint_names = robot.find_joints("panda_joint[1-4]")

# Find body indices by name
body_ids, body_names = robot.find_bodies("panda_hand")

# Examples
cart_dof_idx, _ = robot.find_joints(["panda_joint1"])
gripper_idx, _ = robot.find_joints(["panda_finger_joint.*"])
```

---

## Typical RL Environment Integration

### Observation Space

Common observations for manipulation tasks:

```python
def _get_observations(self) -> dict:
    # Arm joint positions (7)
    arm_q = self.robot.data.joint_pos[:, :7]

    # Arm joint velocities (7)
    arm_dq = self.robot.data.joint_vel[:, :7]

    # Gripper position (2)
    gripper_q = self.robot.data.joint_pos[:, 7:9]

    # End-effector pose
    ee_pos = self.robot.data.body_pos_w[:, self.ee_body_idx, :]
    ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx, :]

    # Combine into observation
    obs = torch.cat([arm_q, arm_dq, gripper_q, ee_pos, ee_quat], dim=-1)

    return {"policy": obs}
```

**Total observation dimension:** 7 + 7 + 2 + 3 + 4 = **23**

For your current config:
- `observation_space = 14` suggests: 7 joint positions + 7 joint velocities (arm only)

### Action Space

```python
def _apply_action(self) -> None:
    # Actions are typically normalized [-1, 1]
    # Scale to actual control range

    # For position control (delta or absolute)
    target_pos = self.robot.data.joint_pos[:, :7] + self.actions * 0.05
    self.robot.set_joint_position_target(target_pos, joint_ids=self._arm_joint_ids)

    # Or for effort control
    efforts = self.actions * self.cfg.action_scale
    self.robot.set_joint_effort_target(efforts, joint_ids=self._arm_joint_ids)

    self.robot.write_data_to_sim()
```

**Your current config:**
- `action_space = 7` corresponds to 7 arm joints

### Reset Logic

```python
def _reset_idx(self, env_ids):
    # Sample new joint positions
    joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, :7] += torch.randn_like(joint_pos[:, :7]) * 0.1

    # Zero velocities
    joint_vel = torch.zeros_like(joint_pos)

    # Reset root state
    root_state = self.robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += self.scene.env_origins[env_ids]

    # Write to simulation
    self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
    self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

---

## Important Notes

### Effort Limits

The `effort_limit_sim` parameter sets the **physics solver's maximum joint effort**. This is a hard limit enforced by the simulation:

- Shoulder/Elbow: 87 Nm
- Wrist: 12 Nm
- Gripper: 200 N

These differ from the real Franka's motor limits and are tuned for simulation stability.

### Velocity Limits

By default, velocity limits are read from the USD file:
- Shoulder/Elbow: ~2.17 rad/s
- Wrist: ~2.61 rad/s
- Gripper: 0.2 m/s

### Fixed Base

```python
robot_cfg.fix_base = True  # Franka base is fixed to the world
```

The Franka is typically mounted, so its base doesn't move.

### Self-Collisions

```python
articulation_props=ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,  # Arm links can collide with each other
)
```

Important for realistic manipulation where the arm might collide with itself.

### Solver Iterations

```python
solver_position_iteration_count=8  # Higher = more accurate but slower
solver_velocity_iteration_count=0  # Usually 0 for articulations
```

---

## Common Patterns

### End-Effector Control

To find the end-effector body:

```python
# In __init__
self.ee_body_idx, _ = self.robot.find_bodies("panda_hand")

# Access end-effector pose
ee_pos = self.robot.data.body_pos_w[:, self.ee_body_idx, :]
ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx, :]
```

### Differential IK

For task-space control:

```python
# Use high PD gains
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

### Parallel Jaw Gripper

The Panda gripper fingers move symmetrically:
- Both fingers move the same distance from center
- `panda_finger_joint1 = panda_finger_joint2`
- Total opening = 2 * joint_position (max ~8cm)

---

## Debugging

### Enable Actuator Value Debugging

To see what values are actually applied:

```python
robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
robot_cfg.actuator_value_resolution_debug_print = True
```

This prints a table showing USD values vs ActuatorCfg values vs what's actually applied.

### Common Issues

1. **Robot falling through floor:** Check `disable_gravity=False` and ground plane collision
2. **Jittery motion:** Increase solver iterations or reduce stiffness
3. **Slow tracking:** Increase stiffness/damping or use HIGH_PD_CFG
4. **Joints hitting limits:** Check joint position limits in USD file
5. **Unstable simulation:** Reduce `max_depenetration_velocity` or effort limits

---

## Reference Documentation

- Franka config: `/isaac/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/franka.py`
- Articulation tutorial: `/isaac/IsaacLab/docs/source/tutorials/01_assets/run_articulation.rst`
- Actuator guide: `/isaac/IsaacLab/docs/source/how-to/write_articulation_cfg.rst`
- Isaac Lab Articulation API: https://isaac-sim.github.io/IsaacLab/main/source/api/isaaclab_assets/isaaclab_assets.html
