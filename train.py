import os
import argparse
import pickle
import shutil
from envs.Quadruped import QuadrupedEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "class_name": "PPO",
            "checkpoint": -1,
            "load_run": -1,
            "log_interval": 1,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
        },
        "class_name": "OnPolicyRunner",
        "experiment_name": exp_name,
        "run_name": "quadruped_motion",
        "save_interval": 300,
        "num_steps_per_env": 24,
        "seed": 1,
        "logger": "tensorboard",
        "max_iterations": max_iterations,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "dof_names": [
            "fl_hx",
            "fr_hx",
            "hl_hx",
            "hr_hx",
            "fl_hy",
            "fr_hy",
            "hl_hy",
            "hr_hy",
            "fl_kn",
            "fr_kn",
            "hl_kn",
            "hr_kn",
        ],
        "default_joint_angles": {  # [rad]
            "fl_hx": 0.0,
            "fr_hx": 0.0,
            "hl_hx": 0.0,
            "hr_hx": 0.0,
            "fl_hy": 0.7,
            "fr_hy": 0.7,
            "hl_hy": 0.7,
            "hr_hy": 0.7,
            "fl_kn": -1.3,
            "fr_kn": -1.3,
            "hl_kn": -1.3,
            "hr_kn": -1.3,
        },
        # PD
        "kp": 350,
        "kd": 50,
        "force_lower": -100,
        "force_upper": 100,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.5546],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.5,
        "feet_height_target": 0.075,
        # "jump_upward_velocity": 1.2,
        # "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="quadruped_walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="device to use: 'cpu', 'cuda:0' or 'mps'",
    )
    args = parser.parse_args()

    backend = gs.constants.backend.metal
    gs.init(logging_level="warning", backend=backend)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = QuadrupedEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
