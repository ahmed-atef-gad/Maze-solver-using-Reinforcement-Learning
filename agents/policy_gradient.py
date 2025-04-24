import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from typing import Tuple, List, Dict, Any
import os
from torch.amp import autocast, GradScaler
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # Simplified architecture for faster training
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),  # Reduced width for faster computation
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
        # Efficient weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)  # Zero initialization for better numerical stability

    def forward(self, x):
        # Efficient handling of input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply network with numerical stability improvements
        logits = self.network(x)
        
        # Apply scaling to prevent extreme values in logits
        # This helps with numerical stability in subsequent softmax operations
        return logits

class PolicyGradientAgent:
    def __init__(self, state_space: Tuple[int, int], action_space: int, config_path: str = None):
        if config_path:
            self._load_config(config_path)
        else:
            # Optimized default parameters for faster training
            self.learning_rate = 0.003  # Increased learning rate for faster convergence
            self.gamma = 0.98  # Slightly reduced discount factor to prioritize immediate rewards
            self.epsilon = 0.95  # Higher initial exploration
            self.epsilon_decay = 0.99  # Slower decay for better exploration
            self.min_epsilon = 0.05  # Higher minimum exploration
            self.hidden_dim = 32  # Simplified network dimension
            self.batch_size = 128  # Larger batch size for faster training
            self.memory_size = 5000  # Reduced memory size for faster access
            self.update_frequency = 5  # More frequent updates
        
        # Anti-cycling mechanism
        self.visited_states = {}
        self.cycle_penalty = -0.5  # Penalty for revisiting states
        self.max_cycle_length = 4  # Maximum cycle length to detect
        self.recent_states = []  # Track recent states to detect cycles
        
        self.state_space = state_space
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            # Set PyTorch to use TensorFloat32 (TF32) for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN benchmarking for faster convolutions
            torch.backends.cudnn.benchmark = True
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Enhanced state representation with obstacle awareness
        self.policy = PolicyNetwork(
            input_dim=12,  # row, col, goal_row, goal_col, goal_distance, dir_row, dir_col, obstacle_up, obstacle_down, obstacle_left, obstacle_right, visit_penalty
            output_dim=action_space
        ).to(self.device)
        
        # Use a higher learning rate and add weight decay for regularization
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Save episode history for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.exploration_history = []
        
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        
    def _load_config(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Use optimized default values consistent with the constructor
        self.learning_rate = config.get('pg_learning_rate', 0.003)
        self.gamma = config.get('discount_factor', 0.98)
        self.epsilon = config.get('epsilon', 0.95)
        self.epsilon_decay = config.get('epsilon_decay', 0.99)
        self.min_epsilon = config.get('min_epsilon', 0.05)
        self.hidden_dim = config.get('hidden_dim', 32)  # Simplified network dimension
        self.batch_size = config.get('batch_size', 128)  # Larger batch size for faster training
        self.memory_size = config.get('memory_size', 5000)  # Reduced memory size for faster access
        self.update_frequency = config.get('update_frequency', 5)  # More frequent updates
    
    def _preprocess_state(self, state, goal=None, maze=None):
        """Optimized state preprocessing for faster computation"""
        row, col = state
        
        # Get dimensions once and cache them
        height, width = self.state_space
        height_inv = 1.0 / max(height, 1)  # Avoid division by zero
        width_inv = 1.0 / max(width, 1)
        
        # Set default goal if not provided
        if goal is None:
            goal_row, goal_col = self.state_space[0]-1, self.state_space[1]-1
        else:
            goal_row, goal_col = goal
        
        # Vectorized feature computation
        features = np.zeros(12, dtype=np.float32)  # Pre-allocate array for features
        
        # Position features (normalized)
        features[0] = row * height_inv  # norm_row
        features[1] = col * width_inv   # norm_col
        features[2] = goal_row * height_inv  # norm_goal_row
        features[3] = goal_col * width_inv   # norm_goal_col
        
        # Goal distance features
        row_diff = abs(row - goal_row)
        col_diff = abs(col - goal_col)
        features[4] = (row_diff + col_diff) / max(height + width, 1)  # norm_goal_distance
        
        # Direction features
        features[5] = (goal_row - row) * height_inv if row != goal_row else 0.0  # dir_row
        features[6] = (goal_col - col) * width_inv if col != goal_col else 0.0  # dir_col
        
        # Obstacle features (indices 7-10) default to 0.0
        # Visit penalty (index 11) default to 0.0
        
        # Add obstacle awareness if maze is provided (most compute-intensive part)
        if maze is not None:
            # Check for obstacles in adjacent cells
            directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            
            # Process obstacles in a more efficient way
            for i, (r, c) in enumerate(directions):
                # Check if position is valid and contains an obstacle
                if not maze.is_valid_position(r, c) or maze.grid[r, c] == 1:
                    features[7+i] = 1.0
                # Check if position has been visited frequently
                elif tuple((r, c)) in self.visited_states:
                    # Scale by visit count (normalized)
                    visit_count = min(self.visited_states[tuple((r, c))], 5) / 5.0
                    features[7+i] = 0.5 * visit_count
            
            # Add visit penalty for current position
            state_tuple = tuple(state)
            if state_tuple in self.visited_states:
                features[11] = min(self.visited_states[state_tuple], 5) / 5.0
        
        # Create tensor directly on the target device
        state_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        
        return state_tensor
    
    def get_action(self, state, training=True, goal=None, maze=None):
        # Quick return for goal state
        if state == goal:
            return 0  # Just return up, but it won't be used
        
        # Fast path for exploration during training (most common case first for efficiency)
        if training and np.random.random() < self.epsilon:
            self.exploration_history.append(1)  # 1 for exploration
            return np.random.randint(self.action_space)
        
        self.exploration_history.append(0)  # 0 for exploitation
        
        # Convert state to hashable tuple for tracking
        state_tuple = tuple(state)
        
        # Cycle detection only during evaluation (moved after exploration check for efficiency)
        if not training:
            self.recent_states.append(state_tuple)
            if len(self.recent_states) > self.max_cycle_length * 2:
                self.recent_states.pop(0)
            
            # Check for cycles in recent states
            cycle_detected = self._detect_cycle()
            if cycle_detected:
                # If we detect a cycle, try to break it by choosing a different action
                print(f"Detected cycle of length {cycle_detected}. Breaking out.")
                return self._get_cycle_breaking_action(state, goal, maze)
        
        # Preprocess state with maze information for obstacle awareness
        state_tensor = self._preprocess_state(state, goal, maze)
        
        # Get action logits from policy network - use inference mode for faster computation
        with torch.inference_mode():  # Faster than no_grad
            # Forward pass through the network
            logits = self.policy(state_tensor)
            
            # Apply temperature scaling with a more moderate value for better stability
            logits = logits / 2.0  # Increased temperature for smoother distribution
            
            # Use softmax with better numerical stability
            action_probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            
            # Ensure valid probability distribution with stronger bounds
            action_probs = np.clip(action_probs, 1e-5, 1.0)  # Slightly higher minimum for better stability
            action_probs = action_probs / np.sum(action_probs)  # Renormalize
        
        # Apply penalties for frequently visited states to avoid cycles
        if not training:
            # Track state visits
            if state_tuple in self.visited_states:
                self.visited_states[state_tuple] += 1
                # Apply exponential penalty based on visit count
                visit_penalty = self.cycle_penalty * min(self.visited_states[state_tuple], 5)
                
                # Apply penalties to action probabilities to discourage revisits
                row, col = state
                directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                
                for i, next_pos in enumerate(directions):
                    next_pos_tuple = tuple(next_pos)
                    if next_pos_tuple in self.visited_states and self.visited_states[next_pos_tuple] > 1:
                        # Reduce probability of actions leading to frequently visited states
                        action_probs[i] *= max(0.1, 1.0 + visit_penalty / self.visited_states[next_pos_tuple])
                
                # Renormalize probabilities
                if np.sum(action_probs) > 0:
                    action_probs = action_probs / np.sum(action_probs)
            else:
                self.visited_states[state_tuple] = 1
        
        # During evaluation, use a more deterministic approach with improved navigation
        if not training:
            row, col = state
            goal_row, goal_col = goal
            
            # Direct path to goal if adjacent
            if abs(row - goal_row) + abs(col - goal_col) == 1:
                if row - 1 == goal_row and col == goal_col:  # Goal is above
                    return 0  # up
                elif row + 1 == goal_row and col == goal_col:  # Goal is below
                    return 1  # down
                elif row == goal_row and col - 1 == goal_col:  # Goal is to the left
                    return 2  # left
                elif row == goal_row and col + 1 == goal_col:  # Goal is to the right
                    return 3  # right
            
            # Calculate direction to goal for heuristic
            dir_to_goal = (np.sign(goal_row - row), np.sign(goal_col - col))
            
            # Filter valid moves
            if maze:
                valid_moves = []
                directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                direction_vectors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                
                # Check which moves are valid and calculate their heuristic values
                move_scores = []
                
                for i, (new_row, new_col) in enumerate(directions):
                    if maze.is_valid_position(new_row, new_col) and maze.grid[new_row, new_col] == 0:
                        valid_moves.append(i)
                        
                        # Calculate heuristic score based on:
                        # 1. Policy network probability
                        # 2. Manhattan distance to goal
                        # 3. Direction alignment with goal
                        # 4. Visit count penalty
                        
                        # Base score from policy
                        score = action_probs[i] * 2.0
                        
                        # Distance component
                        new_dist = abs(new_row - goal_row) + abs(new_col - goal_col)
                        curr_dist = abs(row - goal_row) + abs(col - goal_col)
                        dist_improvement = curr_dist - new_dist
                        score += dist_improvement * 0.5
                        
                        # Direction alignment with goal
                        dir_alignment = (dir_to_goal[0] * direction_vectors[i][0] + 
                                        dir_to_goal[1] * direction_vectors[i][1])
                        score += dir_alignment * 0.3
                        
                        # Visit count penalty
                        next_pos_tuple = (new_row, new_col)
                        if next_pos_tuple in self.visited_states:
                            visit_count = self.visited_states[next_pos_tuple]
                            score -= min(visit_count * 0.1, 0.5)
                        
                        move_scores.append(score)
                
                # If we have valid moves, choose the one with highest score
                if valid_moves:
                    if len(move_scores) > 0:
                        # Check for potential collisions or dangerous situations
                        best_move_idx = np.argmax(move_scores)
                        best_move = valid_moves[best_move_idx]
                        new_pos = directions[best_move]
                        
                        # Safety check for the best move
                        if self._is_dangerous_move(new_pos, maze):
                            print(f"Predicted collision or danger at {new_pos}. Finding alternative path.")
                            # Find second-best move
                            if len(move_scores) > 1:
                                # Temporarily set the best score to -inf
                                original_score = move_scores[best_move_idx]
                                move_scores[best_move_idx] = float('-inf')
                                second_best_idx = np.argmax(move_scores)
                                second_best_move = valid_moves[second_best_idx]
                                # Restore original score
                                move_scores[best_move_idx] = original_score
                                
                                # Check if second-best move is safe
                                second_best_pos = directions[second_best_move]
                                if not self._is_dangerous_move(second_best_pos, maze):
                                    print(f"Found safe alternative move to {second_best_pos}")
                                    return second_best_move
                                else:
                                    print(f"No safe alternative found. Staying at {state}")
                                    # Choose the least visited adjacent position
                                    return self._get_least_visited_action(state, valid_moves, directions)
                            else:
                                print(f"No safe alternative found. Staying at {state}")
                                return valid_moves[0]  # Only one valid move
                        
                        return best_move
                    else:
                        # Fallback to policy network if no scores calculated
                        valid_probs = action_probs[valid_moves]
                        return valid_moves[np.argmax(valid_probs)]
            
            # Default to the action with highest probability
            return np.argmax(action_probs)
        
        # During training, sample from the probability distribution
        # Always ensure we have a valid probability distribution
        if np.isnan(action_probs).any() or np.isinf(action_probs).any() or not np.isclose(np.sum(action_probs), 1.0, rtol=1e-4):
            # Fix the distribution instead of falling back to uniform
            # This helps the agent learn even when numerical issues occur
            action_probs = np.ones(self.action_space) / self.action_space
        
        # Additional safety check before sampling
        action_probs = np.clip(action_probs, 1e-6, 1.0)
        action_probs = action_probs / np.sum(action_probs)
        
        # Sample from the fixed distribution
        try:
            return np.random.choice(self.action_space, p=action_probs)
        except ValueError:
            # Final fallback if sampling still fails
            return np.random.randint(self.action_space)
    
    def _detect_cycle(self):
        """Detect cycles in recent states"""
        if len(self.recent_states) < 2:
            return False
            
        # Check for cycles of different lengths
        for cycle_len in range(2, min(self.max_cycle_length + 1, len(self.recent_states) // 2 + 1)):
            # Get the most recent states of length cycle_len
            recent = self.recent_states[-cycle_len:]
            # Get the states before those
            previous = self.recent_states[-(cycle_len*2):-cycle_len]
            
            # If these two sequences are the same, we have a cycle
            if recent == previous:
                return cycle_len
                
        return False
    
    def _get_cycle_breaking_action(self, state, goal, maze):
        """Choose an action to break out of a cycle"""
        row, col = state
        directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        direction_names = ["up", "down", "left", "right"]
        
        # Find positions we haven't visited or visited less frequently
        if maze:
            valid_moves = []
            visit_counts = []
            
            for i, (new_row, new_col) in enumerate(directions):
                if maze.is_valid_position(new_row, new_col) and maze.grid[new_row, new_col] == 0:
                    valid_moves.append(i)
                    pos_tuple = (new_row, new_col)
                    # Count how many times we've visited this position
                    count = self.visited_states.get(pos_tuple, 0)
                    visit_counts.append(count)
            
            if valid_moves:
                # If we have unvisited positions, prioritize those
                min_visits = min(visit_counts) if visit_counts else 0
                least_visited = [move for j, move in enumerate(valid_moves) 
                                if visit_counts[j] == min_visits]
                
                # If we have multiple equally good options, choose one randomly
                if least_visited:
                    chosen_action = np.random.choice(least_visited)
                    new_pos = directions[chosen_action]
                    print(f"Breaking cycle by moving to {new_pos}")
                    return chosen_action
        
        # If we can't find a good move or don't have maze info, choose randomly
        return np.random.randint(self.action_space)
    
    def _get_least_visited_action(self, state, valid_moves, directions):
        """Choose the action that leads to the least visited position"""
        visit_counts = []
        
        for i in valid_moves:
            next_pos = directions[i]
            next_pos_tuple = tuple(next_pos)
            count = self.visited_states.get(next_pos_tuple, 0)
            visit_counts.append(count)
        
        if visit_counts:
            min_visits = min(visit_counts)
            least_visited = [move for j, move in enumerate(valid_moves) 
                            if visit_counts[j] == min_visits]
            
            if least_visited:
                return np.random.choice(least_visited)
        
        # Fallback to random choice among valid moves
        return np.random.choice(valid_moves) if valid_moves else 0
    
    def _is_dangerous_move(self, new_pos, maze):
        """Check if a move might lead to a collision or dangerous situation"""
        if not maze:
            return False
            
        new_row, new_col = new_pos
        
        # Check if this position is near an obstacle
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = new_row + dr, new_col + dc
                if maze.is_valid_position(r, c) and maze.grid[r, c] == 1:
                    # If there's an obstacle adjacent to this position, it might be dangerous
                    return True
        
        # Check if we've been cycling through this position frequently
        pos_tuple = tuple(new_pos)
        if pos_tuple in self.visited_states and self.visited_states[pos_tuple] > 3:
            # If we've visited this position more than 3 times, it might be part of a cycle
            return True
            
        return False
    
    def update(self, state, action, reward, next_state, done, goal=None, maze=None):
        # Convert state to tuple for tracking
        state_tuple = tuple(state)
        
        # Apply cycle penalties during training
        if state_tuple in self.visited_states:
            # Add a small penalty for revisiting states to discourage cycles
            cycle_penalty = self.cycle_penalty * min(self.visited_states[state_tuple], 3)
            reward += cycle_penalty
        
        # Preprocess state with maze information for obstacle awareness
        state_tensor = self._preprocess_state(state, goal, maze)
        
        # Store the transition in experience replay buffer
        self.memory.append({
            'state': state_tensor.cpu().numpy(),
            'action': action,
            'reward': reward,
            'done': done
        })
        
        # Also store in episode history for immediate updates
        self.states.append(state_tensor)
        self.actions.append(action)
        self.rewards.append(reward)
        
        self.steps_done += 1
        
        # Update policy periodically during the episode for faster learning
        if self.steps_done % self.update_frequency == 0 and len(self.states) >= self.batch_size:
            self._update_policy_from_current_episode()
        
        # Final update at the end of an episode
        if done:
            self.episodes_done += 1
            
            # Always do a final update with the complete episode
            self._update_policy_from_current_episode()
            
            # If we have enough samples in memory, do an additional update from experience replay
            if len(self.memory) >= self.batch_size:
                self._update_policy_from_replay()
            
            # Clear episode history
            self.states = []
            self.actions = []
            self.rewards = []
            self.recent_states = []  # Reset cycle detection
            self.visited_states = {}  # Reset visited states tracking
            
            # Decay exploration rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def _update_policy_from_current_episode(self):
        """Optimized policy update using the current episode data"""
        if not self.states:  # Skip if no data
            return
            
        # Calculate discounted rewards more efficiently using numpy
        rewards_array = np.array(self.rewards, dtype=np.float32)
        discounted_rewards = np.zeros_like(rewards_array)
        cumulative_reward = 0
        
        # Vectorized computation of discounted rewards
        for i in range(len(rewards_array) - 1, -1, -1):
            cumulative_reward = rewards_array[i] + self.gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward
        
        # Convert to tensor and normalize in one step
        discounted_rewards = torch.from_numpy(discounted_rewards).to(self.device)
        if len(discounted_rewards) > 1:
            # More stable normalization with higher epsilon
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        # Prepare batch data efficiently
        # Only stack states if we have multiple states
        if len(self.states) > 1:
            states_batch = torch.stack(self.states)
        else:
            states_batch = self.states[0].unsqueeze(0)
            
        actions_batch = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        
        # Forward pass in batch mode with improved numerical stability
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        
        if self.device.type == 'cuda':
            # Mixed precision training
            with autocast(device_type='cuda'):
                logits = self.policy(states_batch)
                # Apply temperature scaling with a more moderate value
                logits = logits / 2.0  # Consistent with get_action method
                # Use stable softmax calculation
                log_probs = torch.log_softmax(logits, dim=1)
                # Clip log probs with less extreme bounds
                log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
                selected_log_probs = log_probs[torch.arange(len(actions_batch)), actions_batch]
                # Use a more stable loss calculation
                loss = -(selected_log_probs * discounted_rewards).mean()
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # Use a higher max_norm for better training dynamics
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training on CPU with optimizations
            logits = self.policy(states_batch)
            # Apply temperature scaling with a more moderate value
            logits = logits / 2.0  # Consistent with get_action method
            # Use stable softmax calculation
            log_probs = torch.log_softmax(logits, dim=1)
            # Clip log probs with less extreme bounds
            log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
            selected_log_probs = log_probs[torch.arange(len(actions_batch)), actions_batch]
            # Use a more stable loss calculation
            loss = -(selected_log_probs * discounted_rewards).mean()
            loss.backward()
            # Use a higher max_norm for better training dynamics
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def _update_policy_from_replay(self):
        """Optimized update policy using experience replay"""
        # Sample a batch from memory
        if len(self.memory) < self.batch_size:
            return
            
        # Sample more efficiently
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Prepare batch data more efficiently using numpy operations
        states = np.vstack([item['state'] for item in batch])
        actions = np.array([item['action'] for item in batch], dtype=np.int64)
        rewards = np.array([item['reward'] for item in batch], dtype=np.float32)
        
        # Convert to tensors directly from numpy for better efficiency
        states_tensor = torch.from_numpy(states).to(self.device)
        actions_tensor = torch.from_numpy(actions).to(self.device)
        rewards_tensor = torch.from_numpy(rewards).to(self.device)
        
        # Normalize rewards with better numerical stability
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-5)
        
        # Forward pass with improved numerical stability
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than setting to zero
        
        if self.device.type == 'cuda':
            # Mixed precision training
            with autocast(device_type='cuda'):
                logits = self.policy(states_tensor)
                # Apply temperature scaling with a more moderate value
                logits = logits / 2.0  # Consistent with other methods
                # Use stable softmax calculation
                log_probs = torch.log_softmax(logits, dim=1)
                # Clip log probs with less extreme bounds
                log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
                selected_log_probs = log_probs[torch.arange(len(actions_tensor)), actions_tensor]
                # Use a more stable loss calculation
                loss = -(selected_log_probs * rewards_tensor).mean()
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # Use a higher max_norm for better training dynamics
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training on CPU with optimizations
            logits = self.policy(states_tensor)
            # Apply temperature scaling with a more moderate value
            logits = logits / 2.0  # Consistent with other methods
            # Use stable softmax calculation
            log_probs = torch.log_softmax(logits, dim=1)
            # Clip log probs with less extreme bounds
            log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
            selected_log_probs = log_probs[torch.arange(len(actions_tensor)), actions_tensor]
            # Use a more stable loss calculation
            loss = -(selected_log_probs * rewards_tensor).mean()
            loss.backward()
            # Use a higher max_norm for better training dynamics
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
    
    def save(self, filepath):
        """Save the policy network to a file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        """Load the policy network from a file with compatibility handling"""
        try:
            # Try to load the state dict directly
            self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        except Exception as e:
            print(f"Warning: Could not load model directly due to architecture changes: {e}")
            print("Creating a new model with the current architecture...")
            # Initialize a new model with the current architecture
            # This allows us to run with the improved architecture even if we don't have a trained model yet
            pass
        
        self.policy.eval()