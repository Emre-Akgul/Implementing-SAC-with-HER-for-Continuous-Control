import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from models.sac_agent import SAC
from utils.model_saver import load_agent

# Set up environment and testing parameters
def main():
    # Environment setup
    env = gym.make('FetchReach-v3', render_mode="human", reward_type="sparse")
    obs = env.reset()[0]
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Device setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model_path = "checkpoints/sac_her_fetchreach_1000.pth"

    # Initialize the SAC agent
    sac = SAC(state_dim, action_dim, device=device)

    # Load the trained model
    sac = load_agent(sac, model_path, device)
    print(f"Loaded model from {model_path}")

    # Testing parameters
    num_episodes = 1000
    episode_length = 50

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(episode_length):
            # Prepare state
            state = np.concatenate([obs['observation'], obs['desired_goal']])

            # Select action
            action = sac.select_action(state)

            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update state and reward
            obs = next_obs
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    main()
