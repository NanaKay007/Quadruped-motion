import argparse
import os
import pickle
from envs.Quadruped import QuadrupedEnv

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="quadruped_walking")
    parser.add_argument("--ckpt", type=int, default=3200)
    args = parser.parse_args()

    gs.init(backend=gs.metal)

    log_dir = f"./logs/{args.exp_name}"
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    # import IPython; IPython.embed()
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    reward_cfg["reward_scales"] = {}
    train_cfg["policy"]["class_name"] = "ActorCritic" # temp fix
    train_cfg["algorithm"]["class_name"] = "PPO"  # temp fix
    env = QuadrupedEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device='mps'
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""