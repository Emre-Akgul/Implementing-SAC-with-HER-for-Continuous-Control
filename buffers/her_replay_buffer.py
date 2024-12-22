import numpy as np
import random
from collections import deque

class HERReplayBuffer:
    def __init__(self, capacity=1000000, k_future=4):
        self.buffer = deque(maxlen=capacity)
        self.k_future = k_future
    
    def store_trajectory(self, trajectory):
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
                
                # Calculate reward
                reward = 0.0 if np.linalg.norm(next_state['achieved_goal'] - future_ag) < 0.05 else -1.0
                
                self.buffer.append((state, trajectory[t][1], reward, next_state, trajectory[t][4]))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        

        cur_observations = np.array([s['observation'] for s in state_batch])
        cur_goals = np.array([s['desired_goal'] for s in state_batch])
        cur_state = np.concatenate([cur_observations, cur_goals], axis=1)
        

        next_observations = np.array([s['observation'] for s in next_state_batch])
        next_goals = np.array([s['desired_goal'] for s in next_state_batch])
        next_state = np.concatenate([next_observations, next_goals], axis=1)
        return (
            cur_state,
            np.array(action_batch),
            np.array(reward_batch),
            next_state,
            np.array(done_batch)
        )
    
    def __len__(self):
        return len(self.buffer)
