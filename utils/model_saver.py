import torch
import pickle
import os

def save_agent(agent, filepath):
    """
    Save the SAC agent's model parameters and optimizer states.

    Args:
        agent (SAC): The SAC agent to save.
        filepath (str): Path to save the model.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'policy_state_dict': agent.policy.state_dict(),
        'q1_state_dict': agent.q1.state_dict(),
        'q2_state_dict': agent.q2.state_dict(),
        'q1_target_state_dict': agent.q1_target.state_dict(),
        'q2_target_state_dict': agent.q2_target.state_dict(),
        'policy_optimizer_state_dict': agent.policy_optimizer.state_dict(),
        'q1_optimizer_state_dict': agent.q1_optimizer.state_dict(),
        'q2_optimizer_state_dict': agent.q2_optimizer.state_dict(),
        'log_alpha': agent.log_alpha,
        'alpha': agent.alpha,
        'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),
    }, filepath)
    print(f"Agent saved to {filepath}")

def load_agent(agent, filepath, device='cpu'):
    """
    Load the SAC agent's model parameters and optimizer states.

    Args:
        agent (SAC): The SAC agent to load parameters into.
        filepath (str): Path to the saved model file.
        device (str): Device to map the model to ('cpu' or 'cuda').

    Returns:
        SAC: The SAC agent with loaded parameters.
    """
    checkpoint = torch.load(filepath, map_location=device)
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.q1.load_state_dict(checkpoint['q1_state_dict'])
    agent.q2.load_state_dict(checkpoint['q2_state_dict'])
    agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
    agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
    agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    agent.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
    agent.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
    agent.log_alpha = checkpoint['log_alpha']
    agent.alpha = checkpoint['alpha']
    agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

    print(f"Agent loaded from {filepath}")
    return agent


def save_replay_buffer(replay_buffer, filepath):
    """
    Save the replay buffer to a file.

    Args:
        replay_buffer (HERReplayBuffer): The replay buffer to save.
        filepath (str): Path to save the replay buffer.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(replay_buffer.buffer, f)
    print(f"Replay buffer saved to {filepath}")

def load_replay_buffer(replay_buffer, filepath):
    """
    Load the replay buffer from a file.

    Args:
        replay_buffer (HERReplayBuffer): The replay buffer object to populate.
        filepath (str): Path to the saved replay buffer file.

    Returns:
        HERReplayBuffer: Replay buffer with loaded data.
    """
    with open(filepath, 'rb') as f:
        replay_buffer.buffer = pickle.load(f)
    print(f"Replay buffer loaded from {filepath}")
    return replay_buffer