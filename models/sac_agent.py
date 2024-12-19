import gymnasium
import gymnasium_robotics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

from models.q_network import QNetwork
from models.policy_network import PolicyNetwork
from buffers.her_replay_buffer import HERReplayBuffer

class SAC:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device
        self.gamma = 0.98
        self.tau = 0.005
        self.alpha = 0.2
        
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        self.replay_buffer = HERReplayBuffer()
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.sample(state_tensor)
        return action.cpu().numpy()[0]
    
    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        
        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)
        
        # Update Q networks
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_state_batch)
            next_q1 = self.q1_target(next_state_batch, next_action)
            next_q2 = self.q2_target(next_state_batch, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Q1 update
        current_q1 = self.q1(state_batch, action_batch)
        q1_loss = F.mse_loss(current_q1, target_q)
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Q2 update
        current_q2 = self.q2(state_batch, action_batch)
        q2_loss = F.mse_loss(current_q2, target_q)
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Policy update
        new_action, log_prob = self.policy.sample(state_batch)
        q1_new = self.q1(state_batch, new_action)
        q2_new = self.q2(state_batch, new_action)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_prob - q_new).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)