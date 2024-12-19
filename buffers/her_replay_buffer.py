import numpy as np
import random
from collections import deque

class HERReplayBuffer:
    """
    Hindsight Experience Replay (HER) Replay Buffer.

    Stores experiences for off-policy reinforcement learning algorithms
    and applies the HER strategy to augment the data.

    Args:
        capacity (int): Maximum number of experiences to store.
        k_future (int): Number of future states to use for HER augmentation.
    """
    def __init__(self, capacity=1000000, k_future=4):
        self.buffer = deque(maxlen=capacity)
        self.k_future = k_future
    
    def store_trajectory(self, trajectory):
        """
        Store a trajectory and apply the HER strategy to augment the data.

        Args:
            trajectory (list): List of transitions (state, action, reward, next_state, done).
        """
        # Store original trajectory
        for t in range(len(trajectory)):
            self.buffer.append(trajectory[t])
        
        # Apply HER - using future strategy
        for t in range(len(trajectory)):
            # Sample k random states from the future of the trajectory
            future_ids = random.choices(
                range(t + 1, len(trajectory)), 
                k=min(self.k_future, len(trajectory) - t - 1)
            )
            
            for future_id in future_ids:
                # Get the achieved goal from the future state
                future_ag = trajectory[future_id][3]['achieved_goal']
                
                # Create new goal-conditioned experience
                state = trajectory[t][0].copy()
                next_state = trajectory[t][3].copy()
                
                # Replace goal with the future achieved goal
                state['desired_goal'] = future_ag
                next_state['desired_goal'] = future_ag
                
                # Calculate new reward based on future goal
                reward = 0.0 if np.linalg.norm(next_state['achieved_goal'] - future_ag) < 0.05 else -1.0
                
                self.buffer.append((state, trajectory[t][1], reward, next_state, trajectory[t][4]))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            tuple: Batch of (state, action, reward, next_state, done).
        """
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # Convert states to network input format
        def process_states(states):
            observations = np.array([s['observation'] for s in states])
            goals = np.array([s['desired_goal'] for s in states])
            return np.concatenate([observations, goals], axis=1)
        
        return (
            process_states(state_batch),
            np.array(action_batch),
            np.array(reward_batch),
            process_states(next_state_batch),
            np.array(done_batch)
        )
    
    def __len__(self):
        """
        Get the current size of the replay buffer.

        Returns:
            int: Number of stored experiences.
        """
        return len(self.buffer)
