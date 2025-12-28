import argparse
import os
import pickle
import torch
from envs.Quadruped import QuadrupedEnv
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import genesis as gs
from pynput import keyboard

# Global variables to store command velocities
lin_x = 0.0
lin_y = 0.0
ang_z = 0.0
crouch_toggle = 0
stop = False

def on_press(key):
    global lin_x, lin_y, ang_z, crouch_toggle, stop
    try:
        if key.char == ';':
            lin_x += 0.1
        elif key.char == '.':
            lin_x -= 0.1
        elif key.char == '/':
            lin_y += 0.1
        elif key.char == ',':
            lin_y -= 0.1
        elif key.char == ']':
            ang_z += 0.1
        elif key.char == '[':
            ang_z -= 0.1
        elif key.char == '8':
            stop = True
        elif key.char == '7':
            crouch_toggle = 1
        elif key.char == '6':
            crouch_toggle = 0
            
        lin_x = np.clip(lin_x, -1.0, 2.0)
        lin_y = np.clip(lin_y, -0.5, 0.5)
        ang_z = np.clip(ang_z, -0.6, 0.6)
        
            
        # Clear the console
        os.system('clear')
        
        print(f"lin_x: {lin_x:.2f}, lin_y: {lin_y:.2f}, ang_z: {ang_z:.2f}, height: {crouch_toggle:.2f}")
    except AttributeError:
        pass

def on_release(key):
    global lin_x, lin_y, ang_z
    lin_x = 0
    lin_y = 0
    ang_z = 0
    
    if key == keyboard.Key.esc:
        print("listener stopped")
        # Stop listener
        return False

def main():
    global lin_x, lin_y, ang_z, crouch_toggle, stop
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="quadruped_walking")
    parser.add_argument("--ckpt", type=int, default=4999)
    parser.add_argument("--save-data", type=bool, default=False)
    args = parser.parse_args()

    gs.init(
        logger_verbose_time = False,
        logging_level="warning",
    )

    log_dir = f"./logs/{args.exp_name}"
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
    # env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"genesis/logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    train_cfg["policy"]["class_name"] = "ActorCritic" # temp fix
    train_cfg["algorithm"]["class_name"] = "PPO"  # temp fix

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    env_cfg["termination_if_roll_greater_than"] =  50  # degree
    env_cfg["termination_if_pitch_greater_than"] = 50  # degree
    num_envs = 1
    env = QuadrupedEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        # add_camera=True,
        device="mps",
    )
    
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="mps")

    obs, _ = env.reset()
    
    env.commands = torch.tensor([[lin_x, lin_y, ang_z, 0.55]]).to("mps").repeat(num_envs, 1)
    iter = 0

    images_buffer = []
    commands_buffer = []
    with torch.no_grad():
        while not stop:
            actions = policy(obs)
            height = 0.55 if crouch_toggle == 0 else 0.3
            env.commands = torch.tensor([[lin_x, lin_y, ang_z, height]], dtype=torch.float).to("mps").repeat(num_envs, 1)
            obs, rews, dones, infos = env.step(actions, is_train=False)

            iter += 1
            
            # Render the camera
            if env.cam_0 is not None:
                rgb, _, _, _ = env.cam_0.render(
                    rgb=True,
                    depth=False,
                    segmentation=False,
                )
                if args.save_data:
                    images_buffer.append(rgb)
                    commands_buffer.append([lin_x, lin_y, ang_z])
            
            if dones.any():
                iter = 0
          
    # if args.save_data:
    #     # save the images and commands
    #     images_buffer = np.array(images_buffer)
    #     commands_buffer = np.array(commands_buffer)
    #     pickle.dump(images_buffer, open("images_buffer.pkl", "wb"))
    #     pickle.dump(commands_buffer, open("commands_buffer.pkl", "wb"))

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""