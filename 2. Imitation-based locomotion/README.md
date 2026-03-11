# Imitation-Based Quadruped Locomotion from Video

## Overview

Train a locomotion policy for a quadruped robot using imitation learning from monocular video. The pipeline extracts motion from video of real animals and transfers it to a simulated/real robot via reinforcement learning.

## Pipeline

```
Dog Video ──▶ 2D Pose Estimation ──▶ 3D Pose Lifting ──▶ Motion Retargeting ──▶ RL in Sim ──▶ Real Robot
```

### Stage 1: Extract 2D Pose from Video

Run a 2D pose estimation model on each frame of a monocular video of a dog performing the target gait (walk, trot, gallop).

- Use **DeepLabCut** or **ViTPose** to detect joint keypoints (hips, knees, ankles, shoulders, spine, etc.)
- Apply **Kalman filtering** to smooth noisy or occluded joint detections across frames
- **Input:** monocular video (3-8 seconds is sufficient)
- **Output:** time series of 2D joint positions per frame

### Stage 2: Lift 2D Poses to 3D

Convert 2D pixel coordinates into 3D joint trajectories.

- Use a **Spatial-Temporal Graph Convolution Network (ST-GCN)** to estimate 3D joint positions from the 2D sequence
  - Spatial graph convolutions encode the skeleton structure (which joints connect)
  - Temporal convolutions (dilated) capture dynamics across frames
- **Input:** 2D joint positions over time
- **Output:** 3D joint trajectories

### Stage 3: Motion Retargeting

Map the animal's 3D motion onto the robot's joint space.

- Define a mapping between anatomical keypoints (e.g., dog's rear knee to robot's rear knee joint)
- Apply **Inverse Kinematics (IK)** to convert 3D animal joint positions into feasible robot joint angles
- Handle differences in body proportions between the animal and robot
- **Input:** 3D joint trajectories (animal)
- **Output:** reference motion trajectory in robot joint space (joint angles over time)

### Stage 4: Train a Policy in Simulation via RL

Use reinforcement learning to train a robust policy that tracks the reference motion.

1. Set up a physics simulator (Isaac Gym/Lab, MuJoCo, or PyBullet) with the robot's URDF/MJCF model
2. Design a reward function:
   - Joint angle similarity (match reference pose)
   - Velocity similarity (match reference velocities)
   - Base height reward (don't fall)
   - Energy penalty (don't waste energy)
   - Smoothness penalty (avoid jerky actions)
3. Train with **PPO** across many parallel environments
   - **Observation:** joint angles, angular velocities, body orientation, phase variable indexing into reference motion
   - **Action:** target joint angles (sent to PD controllers)
4. Apply **domain randomization** (friction, mass, motor strength, terrain, sensor noise) for sim-to-real transfer
5. Use a **curriculum** — start on flat ground, gradually introduce perturbations and rough terrain

- **Input:** reference motion trajectory, robot model, simulator
- **Output:** trained neural network policy (observations to joint commands)

### Stage 5: Sim-to-Real Transfer

Not applicable

## Resources

**[3DDogs dataset for quadruped video pose](https://cvssp.org/data/3DDogs/)**
