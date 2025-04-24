import numpy as np
from typing import Tuple, Dict
import yaml
from pathlib import Path
import os

class QLearningAgent:
    def __init__(self, state_space: Tuple[int, int], action_space: int, config_path: str = None):
        if config_path:
            self._load_config(config_path)
        else:
            # Default parameters
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.epsilon = 0.9
            self.epsilon_decay = 0.995
            self.min_epsilon = 0.01
        
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((*state_space, action_space))
        self.exploration_history = []  # Track exploration decisions
        
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.learning_rate = config['learning_rate']
        self.discount_factor = config['discount_factor']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.min_epsilon = config['min_epsilon']
    
    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            self.exploration_history.append(1)  # 1 for exploration
            return np.random.randint(self.action_space)
        self.exploration_history.append(0)  # 0 for exploitation
        return np.argmax(self.q_table[state])
    
    def update(self, state: Tuple[int, int], action: int, reward: float, 
               next_state: Tuple[int, int], done: bool):
        current_q = self.q_table[state][action]
        
        # For terminal states, don't include the future reward estimate
        if done:
            new_q = current_q + self.learning_rate * (reward - current_q)
        else:
            max_next_q = np.max(self.q_table[next_state])
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
        
        self.q_table[state][action] = new_q
        
        # Decay exploration rate
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save_and_clear_exploration_history(self, filepath: str):
        """Save exploration history to a file and clear it"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, np.array(self.exploration_history))
        self.exploration_history = []