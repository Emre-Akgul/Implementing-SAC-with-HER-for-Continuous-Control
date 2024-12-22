import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from models.sac_agent import SAC
from utils.model_saver import save_agent, save_replay_buffer
from buffers.her_replay_buffer import HERReplayBuffer

def main():
    # Set up environment
    env = gym.make('FetchReach-v3', render_mode=None, reward_type="sparse")
    obs = env.reset()[0]
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Device setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Initialize SAC agent
    sac = SAC(state_dim, action_dim, device=device)
    save_dir = "checkpoints/"
    os.makedirs(save_dir, exist_ok=True)

    # Set hyperparameters
    max_episodes = 1000 # max number of episodes to stop training
    episode_length = env._max_episode_steps # the default is 50
    batch_size = 256
    num_random_episodes = batch_size
    save = True

    for episode in range(max_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        trajectory = []

        for t in range(episode_length):
            # Create the state input by concatenating observation and desired goal
            state = np.concatenate([obs['observation'], obs['desired_goal']])

            # Select action
            action = sac.select_action(state)

            # Step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Append transition to trajectory
            trajectory.append((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward

            if done:
                break

        # Store trajectory in the HER replay buffer
        sac.replay_buffer.store_trajectory(trajectory)

        # Train the SAC agent
        if len(sac.replay_buffer) > num_random_episodes:
            for _ in range(episode_length):
                sac.update(batch_size)

        # Define success (if last position is within 0.05 of the goal)
        success = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.05

        print(f"Episode {episode}, Reward: {episode_reward}, Succcess: {success}")
        
        # save model in every 1000 episodes
        if save and (episode + 1) % 100 == 0:
            agent_path = os.path.join(save_dir, f"sac_her_fetchreach_{episode + 1}.pth")
            replay_buffer_path = os.path.join(save_dir, f"replay_buffer_{episode + 1}.pkl")
            save_agent(sac, agent_path)
            save_replay_buffer(sac.replay_buffer, replay_buffer_path) # in case want to train further later
            print("Model and replay buffer saved at episode:", episode + 1)
        
    print("Training completed.")

if __name__ == "__main__":
    main()
