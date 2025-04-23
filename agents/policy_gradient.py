import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from typing import Tuple, List
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # Remove softmax from here - we'll apply it separately with more control
        )
    
    def forward(self, x):
        return self.network(x)

class PolicyGradientAgent:
    def __init__(self, state_space: Tuple[int, int], action_space: int, config_path: str = None):
        if config_path:
            self._load_config(config_path)
        else:
            # Default parameters
            self.learning_rate = 0.001
            self.gamma = 0.99
            self.epsilon = 0.9
            self.epsilon_decay = 0.995
            self.min_epsilon = 0.01
            self.hidden_dim = 128
        
        self.state_space = state_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced state representation: position, goal position
        self.policy = PolicyNetwork(
            input_dim=4,  # row, col, goal_row, goal_col
            hidden_dim=self.hidden_dim,
            output_dim=action_space
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Save episode history for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.exploration_history = []
        
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.learning_rate = config.get('pg_learning_rate', 0.001)
        self.gamma = config.get('discount_factor', 0.99)
        self.epsilon = config.get('epsilon', 0.9)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.hidden_dim = config.get('hidden_dim', 128)
    
    def _preprocess_state(self, state, goal=None):
        """Preprocess state for neural network input"""
        row, col = state
        
        # Normalize positions to [0,1] range
        height, width = self.state_space
        norm_row = row / height
        norm_col = col / width
        
        if goal is None:
            # If goal not provided, use default goal
            goal_row, goal_col = self.state_space[0]-1, self.state_space[1]-1
        else:
            goal_row, goal_col = goal
            
        norm_goal_row = goal_row / height
        norm_goal_col = goal_col / width
        
        state_tensor = torch.FloatTensor([
            norm_row, norm_col, norm_goal_row, norm_goal_col
        ]).to(self.device)
        
        return state_tensor
    
    def get_action(self, state, training=True, goal=None, maze=None):
        # Check if we've reached the goal state
        if state == goal:
            # Return a "no-op" action or the best action to stay in place
            return np.argmax([0, 0, 0, 0])  # This will just return 0 (up) but won't be used
        
        # Use epsilon-greedy exploration during training
        if training and np.random.random() < self.epsilon:
            self.exploration_history.append(1)  # 1 for exploration
            return np.random.randint(self.action_space)
        
        self.exploration_history.append(0)  # 0 for exploitation
        
        # Preprocess state
        state_tensor = self._preprocess_state(state, goal)
        
        # Get action logits from policy network
        with torch.no_grad():
            logits = self.policy(state_tensor)
            
            # Apply softmax with numerical stability
            logits = logits - torch.max(logits)  # For numerical stability
            action_probs = torch.softmax(logits, dim=0).cpu().numpy()
            
            # Check for NaN values and fix if necessary
            if np.isnan(action_probs).any():
                print("Warning: NaN detected in action probabilities, using uniform distribution")
                action_probs = np.ones(self.action_space) / self.action_space
        
        # During evaluation, always choose the best action
        if not training:
            return np.argmax(action_probs)
        
        # During training, sample from the probability distribution
        try:
            return np.random.choice(self.action_space, p=action_probs)
        except ValueError as e:
            # Fallback to uniform distribution if there's still an issue
            print(f"Error in action selection: {e}. Using uniform distribution.")
            return np.random.randint(self.action_space)
    
    def update(self, state, action, reward, next_state, done, goal=None, maze=None):
        # Store the transition
        self.states.append(self._preprocess_state(state, goal))
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Only update policy at the end of an episode
        if done:
            # Calculate discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(self.rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)
            
            # Normalize rewards for stability
            discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
            if len(discounted_rewards) > 1:
                discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
            
            # Calculate loss and update policy
            policy_loss = []
            for state_tensor, action, reward in zip(self.states, self.actions, discounted_rewards):
                logits = self.policy(state_tensor)
                
                # Apply softmax with numerical stability
                logits = logits - torch.max(logits)  # For numerical stability
                action_probs = torch.softmax(logits, dim=0)
                
                # Check for NaN values
                if torch.isnan(action_probs).any():
                    print("Warning: NaN detected during update, skipping this sample")
                    continue
                
                selected_action_prob = action_probs[action]
                
                # Use log_softmax for better numerical stability
                log_prob = torch.log(selected_action_prob + 1e-10)  # Add small epsilon to prevent log(0)
                policy_loss.append(-log_prob * reward)
            
            if policy_loss:  # Only update if we have valid samples
                policy_loss = torch.stack(policy_loss).sum()
                
                # Add gradient clipping to prevent exploding gradients
                self.optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Clear episode history
            self.states = []
            self.actions = []
            self.rewards = []
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save the policy network to a file"""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        """Load the policy network from a file"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        self.policy.eval()