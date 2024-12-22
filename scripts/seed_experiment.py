import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.sac_agent import SAC
from utils.model_saver import save_agent, save_replay_buffer
from buffers.her_replay_buffer import HERReplayBuffer

def calculate_success_rate(success_history, window=100):
    if len(success_history) < window:
        return 0
    return np.mean(success_history[-window:]) * 100

def plot_individual_metrics(reward_history, success_history, seed, save_dir):
    # Calculate running success rate
    success_rates = []
    for i in range(len(success_history)):
        success_rates.append(calculate_success_rate(success_history[:i+1]))
    
    # Plot and save episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, 'b-', alpha=0.7)
    plt.title(f'Episode Rewards (Seed {seed})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rewards_seed_{seed}.png'))
    plt.close()
    
    # Plot and save success rate
    plt.figure(figsize=(10, 5))
    plt.plot(success_rates, 'g-', alpha=0.7)
    plt.title(f'Success Rate over 100-Episode Window (Seed {seed})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    plt.ylim([0, 105])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'success_rate_seed_{seed}.png'))
    plt.close()
    
    return success_rates

def plot_combined_metrics(all_metrics, seeds, save_dir):
    # Plot combined rewards
    plt.figure(figsize=(12, 6))
    for seed, metrics in zip(seeds, all_metrics):
        plt.plot(metrics['rewards'], alpha=0.7, label=f'Seed {seed}')
    plt.title('Episode Rewards (All Seeds)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_rewards.png'))
    plt.close()
    
    # Plot combined success rates
    plt.figure(figsize=(12, 6))
    for seed, metrics in zip(seeds, all_metrics):
        plt.plot(metrics['success_rates'], alpha=0.7, label=f'Seed {seed}')
    plt.title('Success Rate over 100-Episode Window (All Seeds)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.grid(True)
    plt.ylim([0, 105])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_success_rates.png'))
    plt.close()

def train_single_seed(seed, save_dir):
    print(f"\nStarting training for seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gym.make('FetchReach-v3', render_mode=None, reward_type="sparse")
    env.reset(seed=seed)
    
    obs = env.reset()[0]
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize SAC agent
    sac = SAC(state_dim, action_dim, device="cpu")
    
    # Training hyperparameters
    max_episodes = 3000
    episode_length = 50
    batch_size = 256
    
    # Metrics tracking
    reward_history = []
    success_history = []
    episode = 0

    while episode < max_episodes:
        obs, _ = env.reset()
        episode_reward = 0
        trajectory = []

        for t in range(episode_length):
            state = np.concatenate([obs['observation'], obs['desired_goal']])
            action = sac.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            trajectory.append((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_reward += reward
            
            if done:
                break

        sac.replay_buffer.store_trajectory(trajectory)

        if len(sac.replay_buffer) > batch_size:
            for _ in range(episode_length):
                sac.update(batch_size)

        success = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.05
        reward_history.append(episode_reward)
        success_history.append(float(success))
        
        success_rate = calculate_success_rate(success_history)
        print(f"Seed {seed} - Episode {episode}, Reward: {episode_reward:.2f}, Success Rate: {success_rate:.2f}%")

        if (episode + 1) % 1000 == 0:
            agent_path = os.path.join(save_dir, f"sac_her_fetchreach_seed_{seed}_episode_{episode}.pth")
            replay_buffer_path = os.path.join(save_dir, f"replay_buffer_seed_{seed}_episode_{episode}.pkl")
            save_agent(sac, agent_path)
            save_replay_buffer(sac.replay_buffer, replay_buffer_path)

        if success_rate == 100 and len(success_history) >= 100:
            print(f"Seed {seed} - Reached 100% success rate at episode {episode}")
            break

        episode += 1
    
    success_rates = plot_individual_metrics(reward_history, success_history, seed, save_dir)
    return {'rewards': reward_history, 'success_rates': success_rates}

def main():
    save_dir = "checkpoints/plots"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = [6, 14, 42, 44, 93]
    all_metrics = []
    
    for seed in seeds:
        metrics = train_single_seed(seed, save_dir)
        all_metrics.append(metrics)
    
    # Create combined plots
    plot_combined_metrics(all_metrics, seeds, save_dir)
    print("\nTraining completed. All plots saved in checkpoints directory.")

if __name__ == "__main__":
    main()