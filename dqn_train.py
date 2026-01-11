"""
Deep Q-Network (DQN) for Atari Pong (ALE/Pong-v5)

This script:
1. Defines a CNN-based DQN architecture
2. Implements a replay buffer
3. Trains a DQN agent (optionally continuing from a checkpoint)
4. Records gameplay videos of a trained agent
"""
import gymnasium as gym
import numpy as np
import random
import cv2
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import logging
from gymnasium.wrappers import RecordVideo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# DQN Network Definition
class DQN(nn.Module):
    """
    Convolutional Deep Q-Network used for Atari environments.
    Input:  (batch, 4, 84, 84)
    Output: Q-values for each action
    """
    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        return self.net(x / 255.0)


# Replay Buffer
class ReplayBuffer:
    """
    Fixed-size buffer that stores experience tuples
    (state, action, reward, next_state, done)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states      = np.stack([b[0] for b in batch])
        actions     = np.array([b[1] for b in batch], dtype=np.int64)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch])
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Frame Preprocessing
def preprocess_frame(frame):
    """
    Convert RGB frame to grayscale and resize to 84x84
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame

# Action Selection (ε-greedy)
def select_action(state, model, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.tensor(state, device=device).unsqueeze(0)
        return model(state_tensor).argmax(1).item()

# =========================
# Training Setup
# =========================
env = gym.make("ALE/Pong-v5")
n_actions = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Q = DQN(n_actions).to(device)
Q_target = DQN(n_actions).to(device)

# Load checkpoint (optional)
Q.load_state_dict(torch.load("round2_1850.pth", map_location=device))
Q_target.load_state_dict(Q.state_dict())

optimizer = optim.Adam(Q.parameters(), lr=1e-4)
replay_buffer = ReplayBuffer(capacity=100_000)

gamma = 0.99
epsilon = 0.1
epsilon_min = 0.1
epsilon_decay = 1e-6

batch_size = 32
target_update_freq = 10_000
max_episodes = 5000

# =========================
# Training Loop
# =========================
state, _ = env.reset()
frame_stack = deque(maxlen=4)

init_frame = preprocess_frame(state)
for _ in range(4):
    frame_stack.append(init_frame)

state = np.stack(frame_stack, axis=0)

episode_rewards = []
current_episode_reward = 0
episode_count = 0
step = 0

while episode_count < max_episodes:
    step += 1

    # Select and perform action
    action = select_action(state, Q, epsilon, n_actions, device)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # Process next state
    next_frame = preprocess_frame(next_obs)
    frame_stack.append(next_frame)
    next_state = np.stack(frame_stack, axis=0)

    # Store transition
    replay_buffer.push(state, action, reward, next_state, done)
    state = next_state
    current_episode_reward += reward

    # Train only when buffer has enough samples
    if len(replay_buffer) >= batch_size:
        s, a, r, s_next, d = replay_buffer.sample(batch_size)

        s      = torch.tensor(s, device=device)
        a      = torch.tensor(a, device=device)
        r      = torch.tensor(r, device=device)
        s_next = torch.tensor(s_next, device=device)
        d      = torch.tensor(d, device=device)

        with torch.no_grad():
            target = r + gamma * Q_target(s_next).max(1)[0] * (1 - d)

        q_values = Q(s).gather(1, a.unsqueeze(1)).squeeze()
        loss = (q_values - target).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network
    if step % target_update_freq == 0:
        Q_target.load_state_dict(Q.state_dict())

    epsilon = max(epsilon_min, epsilon - epsilon_decay)

    # End of episode
    if done:
        episode_rewards.append(current_episode_reward)
        logger.info(f"Episode {episode_count}, Reward: {current_episode_reward}")
        current_episode_reward = 0
        episode_count += 1

        if episode_count % 50 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            torch.save(Q.state_dict(), f"dqn_pong_episode{episode_count}.pth")
            logger.info(f"Avg Reward (last 100): {avg_reward:.2f}")
            if avg_reward >= 19:
                print("Pong solved!")
                break

        state, _ = env.reset()
        frame_stack.clear()
        init_frame = preprocess_frame(state)
        for _ in range(4):
            frame_stack.append(init_frame)
        state = np.stack(frame_stack, axis=0)

# Evaluation
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda e: True)

Q.eval()

def play_trained_agent(env, model):
    state, _ = env.reset()
    frame_stack = deque(maxlen=4)

    frame = preprocess_frame(state)
    for _ in range(4):
        frame_stack.append(frame)

    state = np.stack(frame_stack, axis=0)
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = model(
                torch.tensor(state, device=device).unsqueeze(0)
            ).argmax(1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        frame_stack.append(preprocess_frame(next_state))
        state = np.stack(frame_stack, axis=0)

    return total_reward


score = play_trained_agent(env, Q)
print(f"Trained agent score: {score}")

