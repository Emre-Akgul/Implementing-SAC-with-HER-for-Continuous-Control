import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from models.sac_agent import SAC
from utils.model_saver import load_agent

# Evaluate the model
def evaluate_model(env, sac, num_episodes=100, episode_length=50):
    """
    Evaluate the SAC agent on a specified number of episodes and compute metrics.

    Args:
        env (gym.Env): The Gymnasium environment.
        sac (SAC): The SAC agent to evaluate.
        num_episodes (int): Number of evaluation episodes.
        episode_length (int): Maximum number of steps per episode.

    Prints:
        - Average reward over all episodes.
        - Success rate as a percentage.
    """
    success_count = 0
    total_reward = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        for t in range(episode_length):
            # Prepare the input state
            state = np.concatenate([obs['observation'], obs['desired_goal']])

            # Select action
            action = sac.select_action(state)

            # Step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update state and reward
            obs = next_obs
            episode_reward += reward

            if done:
                break

        # Check if the goal was achieved
        if "is_success" in info and info["is_success"]:
            success_count += 1

        total_reward += episode_reward
        print(f"Episode {episode}, Reward: {episode_reward}")

    # Calculate metrics
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes * 100
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")

# Main function for evaluation
def main():
    """
    Main function to set up the environment and evaluate the SAC agent.
    """
    # Environment setup
    env = gym.make('FetchReach-v3', render_mode=None, reward_type="sparse")
    obs = env.reset()[0]
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/sac_her_fetchreach.pth"

    # Initialize and load the SAC agent
    sac = SAC(state_dim, action_dim, device=device)
    sac = load_agent(sac, model_path, device)
    print(f"Loaded model from {model_path}")

    # Evaluate the SAC agent
    evaluate_model(env, sac)

if __name__ == "__main__":
    main()
