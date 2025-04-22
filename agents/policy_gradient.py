import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import yaml
from pathlib import Path
from typing import Tuple, List, Dict

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class PolicyGradientAgent:
    def __init__(self, state_space: Tuple[int, int], action_space: int, config_path: str = None):
        if config_path:
            self._load_config(config_path)
        else:
            # Default parameters
            self.learning_rate = 0.001
            self.discount_factor = 0.99
            self.hidden_dim = 128
            self.episodes = 10000
            self.epsilon = 0.9  # Adding epsilon for compatibility with train.py
            self.epsilon_decay = 0.995
            self.min_epsilon = 0.01
        
        self.state_space = state_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy network
        self.policy = PolicyNetwork(
            input_dim=4,  # State representation: (row, col, goal_row, goal_col)
            hidden_dim=self.hidden_dim,
            output_dim=action_space
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Storage for episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        # For tracking exploration vs exploitation
        self.exploration_history = []
    
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.learning_rate = config.get('pg_learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.99)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.episodes = config.get('episodes', 10000)
        self.epsilon = config.get('epsilon', 0.9)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)
    
    def _preprocess_state(self, state: Tuple[int, int], goal: Tuple[int, int] = None) -> torch.Tensor:
        """Convert state tuple to tensor with normalized coordinates"""
        row, col = state
        if goal is None:
            # During evaluation, we don't have access to the goal directly
            # Use a default goal position (bottom right)
            goal_row, goal_col = self.state_space[0]-1, self.state_space[1]-1
        else:
            goal_row, goal_col = goal
            
        # Create a feature vector with current position and goal position
        state_tensor = torch.FloatTensor(
            [row/self.state_space[0], col/self.state_space[1], 
             goal_row/self.state_space[0], goal_col/self.state_space[1]]
        ).to(self.device)
        
        return state_tensor
    
    def get_action(self, state: Tuple[int, int], training: bool = True, goal: Tuple[int, int] = None) -> int:
        state_tensor = self._preprocess_state(state, goal)
        
        # Get action probabilities from policy network
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        
        # Sample action from probability distribution
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            
            # For exploration tracking (stochastic policy is always exploring to some degree)
            # We'll consider it exploration if we didn't take the highest probability action
            max_prob_action = torch.argmax(action_probs).item()
            self.exploration_history.append(1 if action.item() != max_prob_action else 0)
            
            return action.item()
        else:
            # During evaluation, take the most probable action
            return torch.argmax(action_probs).item()
    
    def store_transition(self, state: Tuple[int, int], action: int, reward: float, goal: Tuple[int, int] = None):
        """Store transition for later training"""
        state_tensor = self._preprocess_state(state, goal)
        action_tensor = torch.tensor([action], device=self.device)
        
        # Get log probability of the action taken
        action_probs = self.policy(state_tensor)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action_tensor)
        
        # Store transition
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def update(self, state: Tuple[int, int], action: int, reward: float, 
               next_state: Tuple[int, int], done: bool, goal: Tuple[int, int] = None):
        """Store transition and update policy if episode is done"""
        self.store_transition(state, action, reward, goal)
        
        if done:
            self._update_policy()
    
    def _update_policy(self):
        """Update policy network using collected episode data"""
        if not self.rewards:  # Skip if no data
            return
            
        # Calculate discounted rewards
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)
        
        # Normalize returns for stability
        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)  # Negative because we want to maximize
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate (epsilon)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Clear episode data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def save(self, filepath: str):
        """Save policy network to file"""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load policy network from file"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy.eval()