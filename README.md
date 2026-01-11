# Deep Q-Network (DQN) for Atari Pong

This repository contains an implementation of a Deep Q-Network (DQN) agent trained to play Atari Pong using the Gymnasium ALE environment and PyTorch.

**Click to play the video below**
https://github.com/sssure02/dqn-pong-atari/pong_trained-episode-1.mp4

## Overview
The agent uses a convolutional neural network to approximate the action-value function from stacked grayscale game frames. Training follows the original DQN algorithm with experience replay and a target network for stability.

## Key Features
- CNN-based DQN architecture for image-based input
- Experience replay buffer
- Target network updates
- ε-greedy exploration strategy
- Frame preprocessing (grayscale, resizing, frame stacking)
- Video recording of trained agent gameplay

## Environment
- **Environment:** `ALE/Pong-v5`
- **Observation:** 4 stacked frames (84 × 84 grayscale)
- **Action Space:** Discrete joystick actions
- **Frameworks:** PyTorch, Gymnasium

## Training
The agent is trained using mean-squared Bellman error with the Adam optimizer. Training can be resumed from a saved checkpoint and periodically saves model weights while monitoring average episode reward.

## Evaluation
After training, the agent can be evaluated in a video-recording environment to visualize learned behavior and measure performance.

## Results

The DQN agent demonstrates steady learning progress on Atari Pong, improving from random play to near-optimal performance.
- Initial performance: ~ −20 average reward (random behavior)
- After training: Average reward of +18 to +21 over the last 100 episodes
- Evaluation: Deterministic policy (ε = 0), gameplay recorded using Gymnasium RecordVideo

These results indicate that the agent has learned effective paddle control, ball tracking, and long-term reward optimization consistent with the original DQN benchmark.
