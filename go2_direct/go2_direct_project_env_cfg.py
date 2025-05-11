# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_D1T_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

DT1_JOINTS = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
]
GO2_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
DOG_JOINTS = DT1_JOINTS + GO2_JOINTS


@configclass
class Go2DirectProjectEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 5.0
    # - spaces definition
    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_D1T_CFG
    robot_cfg.prim_path = '/World/envs/env_.*/Robot'

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    dog_dof_name = GO2_JOINTS
    dt1_dof_name = DT1_JOINTS
    all_dof_name = GO2_JOINTS + DT1_JOINTS
    # - action scale
    action_scale = 0.1  # [N]
    # - reward scales
    rew_scale_alive = .1
    rew_scale_terminated = -.2
    rew_scale_dog_pos = .1
    rew_scale_dt1_vel = -.1
    rew_scale_dt1_pos = -.1

    # - reset states/conditions
    initial_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    target_pos = 0  # reset if cart exceeds this position [m]
    target_vel = 23

