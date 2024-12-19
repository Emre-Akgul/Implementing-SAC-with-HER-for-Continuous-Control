import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from models.sac_agent import SAC
from utils.model_saver import save_agent, save_replay_buffer
from buffers.her_replay_buffer import HERReplayBuffer

# Set up environment and training parameters
def main():
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

    # Training hyperparameters
    max_episodes = 3000
    episode_length = 50
    batch_size = 256

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
        if len(sac.replay_buffer) > batch_size:
            for _ in range(episode_length):
                sac.update(batch_size)

        print(f"Episode {episode}, Reward: {episode_reward}")
        
        # save model in every 1000 episodes
        if episode % 1000 == 0:
            agent_path = os.path.join(save_dir, f"sac_her_fetchreach_{episode}.pth")
            replay_buffer_path = os.path.join(save_dir, f"replay_buffer_{episode}.pkl")
            save_agent(sac, agent_path)
            save_replay_buffer(sac.replay_buffer, replay_buffer_path)
            print("Model and replay buffer saved at episode:", episode)

    # Final save
    agent_path = os.path.join(save_dir, "sac_her_fetchreach.pth")
    replay_buffer_path = os.path.join(save_dir, "replay_buffer.pkl")
    save_agent(sac, agent_path)
    save_replay_buffer(sac.replay_buffer, replay_buffer_path)

    print("Training complete. Model and replay buffer saved.")

if __name__ == "__main__":
    main()
