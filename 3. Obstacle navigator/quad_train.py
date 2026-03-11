import os
import argparse
import pickle
import shutil
from envs.QuadrupedNavigator import QuadrupedNavigatorEnv
from envs.Quadruped import QuadrupedEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
import genesis as gs
import yaml
import os


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="navigator")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("-i", "--max_iterations", type=int, default=7200)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--show_viewer", action="store_true")

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
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_config("./navigator_config.yaml").values()
    train_cfg["max_iterations"] = args.max_iterations

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # load locomotion policy
    locomotion_policy_path = "policies/locomotion.pt"
    import pickle as pkl
    cfg = pkl.load(open("policies/cfgs.pkl", "rb"))

    quadruped_env = QuadrupedEnv(
        num_envs=1,
        env_cfg=cfg[0],
        obs_cfg=cfg[1],
        reward_cfg=cfg[2],
        command_cfg=cfg[3],
        device=args.device,
    )
    locomotion_runner = OnPolicyRunner(
        quadruped_env, cfg[4], log_dir, device=args.device
    )
    locomotion_runner.load(locomotion_policy_path)

    locomotion_policy = locomotion_runner.get_inference_policy(device=args.device)

    env = QuadrupedNavigatorEnv(
        num_envs=args.num_envs,
        controler_policy=locomotion_policy,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    if args.ckpt is not None:
        print("Loading checkpoint:", args.ckpt)
        resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
        runner.load(resume_path)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
