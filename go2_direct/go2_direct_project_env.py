# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .go2_direct_project_env_cfg import Go2DirectProjectEnvCfg


class Go2DirectProjectEnv(DirectRLEnv):
    cfg: Go2DirectProjectEnvCfg

    def __init__(self, cfg: Go2DirectProjectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._dog_dof_idx, _ = self.robot.find_joints(self.cfg.dog_dof_name)
        self._dt1_dof_idx, _ = self.robot.find_joints(self.cfg.dt1_dof_name)
        self._all_joints_idx, _ = self.robot.find_joints(self.cfg.all_dof_name)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                'rew_alive',
                'rew_termination',
                'rew_dog_pos',
                'rew_dt1_vel',
                'rew_dt1_pos',
            ]
        }
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(182, 182)))
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            (
                self.joint_pos[:, self._dog_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._dog_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._dt1_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._dt1_dof_idx[0]].unsqueeze(dim=1),
                self.actions,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward, rew_alive, rew_termination, rew_dog_pos, rew_dt1_vel, rew_dt1_pos = compute_rewards(
            # rew_pole_pos,
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_dog_pos,
            self.cfg.rew_scale_dt1_vel,
            self.cfg.rew_scale_dt1_pos,
            self.joint_pos[:, self._dog_dof_idx[0]],
            self.joint_pos[:, self._dt1_dof_idx[0]],
            self.joint_vel[:, self._dt1_dof_idx[0]],
            self.cfg.target_pos,
            self.cfg.target_vel,
            self.reset_terminated,
        )
        rew_dict = {
            'rew_alive': rew_alive,
            'rew_termination': rew_termination,
            'rew_dog_pos': rew_dog_pos,
            'rew_dt1_vel': rew_dt1_vel,
            'rew_dt1_pos': rew_dt1_pos,
        }

        for key, value in rew_dict.items():
            self._episode_sums[key] += value
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._dog_dof_idx]) > self.cfg.target_pos, dim=1)
        out_of_bounds = torch.any(torch.abs(self.joint_vel[:, self._dt1_dof_idx]) > self.cfg.target_vel, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self.actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos[:, self._dog_dof_idx] += sample_uniform(
            self.cfg.initial_angle_range[0] * math.pi,
            self.cfg.initial_angle_range[1] * math.pi,
            joint_pos[:, self._dog_dof_idx].shape,
            str(joint_pos.device),
        )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras['Episode_Reward/' + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras['log'] = dict()
        self.extras['log'].update(extras)

        extras = dict()
        extras["Episode_Termination/num_terminations"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_outs"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_dog_pos: float,
    rew_scale_dt1_vel: float,
    rew_scale_dt1_pos: float,
    dog_pos: torch.Tensor,
    dt1_vel: torch.Tensor,
    dt1_pos: torch.Tensor,
    target_pos: float,
    target_vel: float,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_dog_pos = rew_scale_dog_pos * torch.sum(torch.square(target_pos - dog_pos).unsqueeze(dim=1), dim=-1)
    rew_dt1_vel = rew_scale_dt1_vel * torch.sum(torch.abs(target_vel - dt1_vel).unsqueeze(dim=1), dim=-1)
    rew_dt1_pos = rew_scale_dt1_pos * torch.sum(torch.square(target_pos - dt1_pos).unsqueeze(dim=1), dim=-1)
    total_reward = (rew_alive + rew_termination + rew_dt1_vel + rew_dt1_pos)
    return (total_reward, rew_alive, rew_termination,
            rew_dog_pos,
            rew_dt1_vel, rew_dt1_pos)
    
