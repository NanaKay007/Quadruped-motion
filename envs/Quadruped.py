from matplotlib.pylab import f
import numpy as np
from rsl_rl.env.vec_env import VecEnv
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)
from tensordict import TensorDict
import torch
import math


def gs_rand_float(low, upper, shape, device):
    return (upper - low) * torch.rand(shape, device=device) + low


class QuadrupedEnv(VecEnv):
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        device,
        show_viewer=False,
        add_camera=False,
    ):
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.cfg = {
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
        }
        self.device = torch.device(device)
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.num_actions = env_cfg[
            "num_actions"
        ]  # all 12 robot joints for quadruped without arm
        self.num_commands = command_cfg["num_commands"]
        self.num_obs = obs_cfg["num_obs"]
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # Initialize scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=30,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0])
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.cam_0 = None
        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 6.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )


        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=self.device, dtype=gs.tc_float
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=self.device, dtype=gs.tc_float
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="/Users/nkayslaptop/Desktop/Master's Program/Reinforcement learning/Final Project/GPT-Reward/robots/boston_dynamics_spot/scene.xml",
                # pos=self.base_init_pos.cpu().numpy(),
                # quat=self.base_init_quat.cpu().numpy(),
            )
        )
        self.scene.build(n_envs=self.num_envs)
        self.dofs_idx = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]
        self._set_gains_and_damping()

        # prepare reward functions. #TODO why scale by dt?
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt  # scale rewards by dt
            self.reward_functions[name] = getattr(self, f"_reward_{name}")
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=self.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=self.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["dof_pos"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros_like(self.actions)

        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.crouch_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        # self.jump_target_height = torch.zeros((self.num_envs,), device=self.device)

        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def __del__(self):
        self.scene.destroy()
        print("Destroyed scene")

    # required by gymnasium.Env!
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state and return an initial observation.
        Returns:
            observation (object): the initial observation of the space.
            info (dict): a dictionary containing additional information about the reset.
        """
        # print("Resetting environments")
        self.reset_buf[:] = True
        self.reset_indx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), {}

    def reset_indx(self, envs_indx):
        # print(f"Resetting envs: {envs_indx}")
        if len(envs_indx) == 0:
            return

        # print(f"Resetting envs: {envs_indx}")
        # reset robot dofs
        self.dof_pos[envs_indx] = self.default_dof_pos
        self.dof_vel[envs_indx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_indx],
            dofs_idx_local=self.dofs_idx,
            zero_velocity=True,
            envs_idx=envs_indx,
        )
        # reset robot position, orientation, velocity
        self.base_pos[envs_indx] = self.base_init_pos
        # self.robot.set_pos(
        #     self.base_init_pos[envs_indx], zero_velocity=False, envs_idx=envs_indx
        # )
        self.base_quat[envs_indx] = self.base_init_quat.reshape(1, -1)
        # self.robot.set_quat(
        #     self.base_quat[envs_indx], zero_velocity=False, envs_idx=envs_indx
        # )
        # self.robot.zero_all_dofs_velocity(envs_indx)
        
        self.robot.set_qpos(
            torch.cat(
                [self.base_pos[envs_indx], self.base_quat[envs_indx], self.dof_pos[envs_indx]],
                dim=-1,
            ),
            envs_idx=envs_indx,
            zero_velocity=True
        )
        # reset buffers
        self.base_lin_vel[envs_indx] = 0.0
        self.base_ang_vel[envs_indx] = 0.0
        self.last_actions[envs_indx] = 0.0
        self.last_dof_vel[envs_indx] = 0.0
        self.reset_buf[envs_indx] = True
        self.crouch_toggled_buf[envs_indx] = 0.0
        # self.jump_target_height[envs_indx] = 0.0
        self.episode_length_buf[envs_indx] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_indx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_indx] = 0.0

        self._sample_commands(envs_indx)

    def step(self, actions, is_train=True):
        # take action after clipping
        if type(actions) is not torch.Tensor:
            actions = torch.tensor(actions, device=self.device, dtype=gs.tc_float)
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        # exec_actions = self.last_actions if self.simu
        target_dof_pos = (
            self.last_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.dofs_idx)
        self.scene.step()

        # get observations: position, orientation, velocities in base frame and update buffers
        self.episode_length_buf += 1
            
        robot_qpos = self.robot.get_qpos()
        self.base_pos[:] = robot_qpos[:, :3]
        self.base_quat[:] = robot_qpos[:, 3:7]

        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True, degrees=True
        )           
                
        inv_base_quat = inv_quat(self.base_quat)  # transform world to base frame
        dofs_velocity = self.robot.get_dofs_velocity()
        robot_lin_vel = dofs_velocity[:, :3]
        robot_ang_vel = dofs_velocity[:, 3:6]
        self.base_lin_vel[:] = transform_by_quat(
            robot_lin_vel, inv_base_quat
        )  # linear velocity in base frame
        self.base_ang_vel[:] = transform_by_quat(
            robot_ang_vel, inv_base_quat
        )  # angular velocity in base frame
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.dofs_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.dofs_idx)

        # get environments that need to be reset once resampling time is reached
        envs_idx = (
            (
                self.episode_length_buf
                % int(self.env_cfg["resampling_time_s"] / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        if is_train:
            self._sample_commands(envs_idx)

        # check termination conditions
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # if torch.any(self.reset_buf):
            # print(f"terminated envs due to max episode length: {self.reset_buf}")
        
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        # if torch.any(self.reset_buf):
        #     import IPython; IPython.embed()
        #     print(f"terminated envs due to pitch: {self.reset_buf}")
            
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )
        # if torch.any(self.reset_buf):
        #     print(f"terminated envs due to roll: {self.reset_buf}")
        # if torch.any(self.reset_buf):
        #     import IPython; IPython.embed()
        # print(f"terminated envs due to roll: {self.reset_buf}")

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=self.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        self.reset_indx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute rewards
        self.rew_buf[:] = 0.0  # reset reward buffer
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations. Why scale some of them? #TODO
        tensors = [
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 5
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
        ]
        self.obs_buf = torch.cat(
            tensors,
            axis=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        self.extras["observations"]["critic"] = self.obs_buf

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        # active_mask = (self.jump_toggled_buf < 0.01).float()
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        # active_mask = (self.jump_toggled_buf < 0.01).float()
        return torch.sum(
            torch.square(self.last_actions - self.actions), dim=1
        )

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        # active_mask = (self.jump_toggled_buf < 0.01).float()
        return torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos), dim=1
        )

    def _reward_base_height(self):
        # Penalize base height away from target
        target_height = torch.where(
            self.commands[:, 3] > 0.0, self.reward_cfg["crouch_height"], self.reward_cfg["base_height_target"]
        )
        print(target_height)
        base_height_error = torch.square(self.base_pos[:, 2] - target_height)
        return base_height_error
    
    def _reward_orientation(self):
        # Reward upright base orientation
        orientation_error =  torch.abs(self.base_euler[:, 0]) + torch.abs(self.base_euler[:, 1]) + torch.abs(self.base_euler[:, 2])
        return torch.exp(-orientation_error / self.reward_cfg["tracking_sigma"] )

    def get_observations(self):
        return TensorDict(
            {
                "policy": self.obs_buf,
            }
        )

    def _sample_commands(self, envs_idx):
        """
        Sample new commands for each environment
        command format: [lin_vel_x, lin_vel_y, ang_vel, crouch_toggle]
        """
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device
        )

        self.commands[envs_idx, 3] = torch.round(gs_rand_float(
                *self.command_cfg["height_range"], (len(envs_idx),), self.device
            ))
        # print(f"Sampled new commands for envs {envs_idx}: {self.commands[envs_idx]}")

    def _set_gains_and_damping(self):
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.dofs_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.dofs_idx)
        self.robot.set_dofs_force_range(
            lower=[self.env_cfg["force_lower"]] * self.num_actions,
            upper=[self.env_cfg["force_upper"]] * self.num_actions,
            dofs_idx_local=self.dofs_idx,
        )
        return