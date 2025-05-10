# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import UNITREE_GO2_D1T_CFG  # isort:skip

DT1_JOINTS = [
    "Joint1",
    "Joint2",
    "Joint3",
    "Joint4",
    "Joint5",
    "Joint6",
    "Joint7_1",
    "Joint7_2",
]
GO2_JOINTS = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
DOG_JOINTS = DT1_JOINTS + GO2_JOINTS

##
# Scene definition
##

CAM_CONF = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Link5/camera_bottom_screw_frame/camera_link/camera_depth_frame/camera_depth_optical_frame/depth_camera",
    data_types=["depth"],
    width=800,
    height=600,
    debug_vis=True,
    spawn=None,
)
# /go2/Link5/camera_bottom_screw_frame/camera_link/camera_depth_frame/camera_depth_optical_frame/depth_camera


@configclass
class Go2ProjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_D1T_CFG
    # camera
    depth_camera: CameraCfg = CAM_CONF

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=DT1_JOINTS, scale=5.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # image = ObsTerm(func=mdp.image, params={"sensor_cfg": Camera(cfg=CAM_CONF), "data_type": "depth"})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_dog_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=GO2_JOINTS),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
        # interval_range_s=(0.5, 0.5)
    )

    reset_d1t_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=DT1_JOINTS),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
        # interval_range_s=(1.5, 1.5),
    )

    # reset_robot_pos = EventTerm(
        # func=mdp.reset_root_state_uniform,
        # mode='reset',
        # params={
            # "asset_cfg": SceneEntityCfg("robot"),
            # "pose_range": (-0.1, 0.1),
            # "velocity_range": (-0.1, 0.1),
        # }
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    target = RewTerm(
        func=mdp.point_distance,
        weight=3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DT1_JOINTS), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DT1_JOINTS)},
    )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
        # func=mdp.joint_vel_l1,
        # weight=-0.005,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=DT1_JOINTS)},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Dog out of bounds
    # dog_joints_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "bounds": (-30.0, 30.0)},
    # )
    # dog_joint_inappropriate = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 60, "asset_cfg": SceneEntityCfg("robot")},
    # )
    # joint_velocity_limit = DoneTerm(
    #     func=mdp.joint_vel_out_of_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot")},  # Example bounds
    # )

##
# Environment configuration
##


@configclass
class Go2ProjectEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Go2ProjectSceneCfg = Go2ProjectSceneCfg(num_envs=2, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 240
        self.sim.render_interval = self.decimation
