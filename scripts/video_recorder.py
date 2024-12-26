import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import cv2
from models.sac_agent import SAC
from utils.model_saver import load_agent
import glob

def concatenate_videos(video_files, output_path):
    """Concatenate multiple videos into one using cv2."""
    # Get the first video's properties
    first = cv2.VideoCapture(video_files[0])
    fps = int(first.get(cv2.CAP_PROP_FPS))
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first.release()

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write each video
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()

def main():
    # Create videos directory if it doesn't exist
    temp_video_dir = "videos/fetch_reach/temp"
    final_video_dir = "videos/fetch_reach"
    os.makedirs(temp_video_dir, exist_ok=True)
    os.makedirs(final_video_dir, exist_ok=True)

    # Environment setup with video recording wrapper
    env = gym.make('FetchReach-v3', render_mode="rgb_array", reward_type="sparse")
    env = RecordVideo(
        env,
        temp_video_dir,
        episode_trigger=lambda x: True,  # Record every episode
        name_prefix="fetch_reach"
    )
    
    obs = env.reset()[0]
    state_dim = obs['observation'].shape[0] + obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Device setup
    device = "cpu"
    version = 100
    model_path = f"checkpoints/sac_her_fetchreach_{version}.pth"

    # Initialize and load the SAC agent
    sac = SAC(state_dim, action_dim, device=device)
    sac = load_agent(sac, model_path, device)
    print(f"Loaded model from {model_path}")

    # Testing parameters
    num_episodes = 10  # Record 20 episodes
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

    # Close the environment
    env.close()

    # Concatenate all videos into one
    print("Concatenating videos...")
    video_files = sorted(glob.glob(os.path.join(temp_video_dir, "*.mp4")))
    
    if video_files:
        output_path = os.path.join(final_video_dir, f"fetch_reach_{version}.mp4")
        concatenate_videos(video_files, output_path)
        
        # Clean up temporary video files
        for video in video_files:
            os.remove(video)
        os.rmdir(temp_video_dir)
        
        print(f"Combined video saved to: {output_path}")
    else:
        print("No video files found to concatenate")

if __name__ == "__main__":
    main()