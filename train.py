import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
from agents.q_learning import QLearningAgent
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def train():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment with moving obstacles
    maze = Maze(width=10, height=10, num_moving_obstacles=4, use_seed=40)
    agent = QLearningAgent(
        state_space=(maze.height, maze.width),
        action_space=4,
        config_path='configs/default.yaml'
    )
    
    # Training metrics
    rewards_history = []
    steps_history = []
    exploration_rates = []
    
    # Training loop
    for episode in tqdm(range(config['episodes']), desc="Training"):
        state = maze.start
        total_reward = 0
        steps = 0
        done = False
        
        # Track exploration rate for this episode
        exploration_rates.append(agent.epsilon)
        
        while not done and steps < 1000:
            # Get action from agent
            action = agent.get_action(state)
            maze.update_moving_obstacles()
            
            # Take action
            row, col = state
            if action == 0:   # up
                next_state = (row-1, col)
            elif action == 1:  # down
                next_state = (row+1, col)
            elif action == 2:  # left
                next_state = (row, col-1)
            elif action == 3:  # right
                next_state = (row, col+1)
            
            # Check if move is valid
            if not maze.is_valid_position(*next_state):
                # Strong penalty for invalid moves
                reward = -10.0
                next_state = state  # Stay in place if invalid
                done = False
            else:
                # Get reward based on new position
                reward = maze.get_reward(*next_state)
                
                # Extra positive reward for getting closer to goal
                current_dist = abs(row - maze.goal[0]) + abs(col - maze.goal[1])
                new_dist = abs(next_state[0] - maze.goal[0]) + abs(next_state[1] - maze.goal[1])
                if new_dist < current_dist:
                    reward += 0.5  # Small bonus for progress
                
                # Check if goal reached
                done = (next_state == maze.goal)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record episode metrics
        rewards_history.append(total_reward)
        steps_history.append(steps)
    
    # Save training results
    np.save('results/rewards.npy', rewards_history)
    np.save('results/steps.npy', steps_history)
    np.save('results/q_table.npy', agent.q_table)
    np.save('results/exploration_rates.npy', exploration_rates)
    np.save('results/exploration_history.npy', np.array(agent.exploration_history))
    np.save('results/maze.npy', maze.grid)
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    # Reward plot
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Steps plot
    plt.subplot(1, 3, 2)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Exploration analysis
    plt.subplot(1, 3, 3)
    plt.plot(exploration_rates)
    plt.title('Exploration Rate (ε) Over Time')
    plt.xlabel('Episode')
    plt.ylabel('ε value')
    
    plt.tight_layout()
    plt.savefig('results/training_plot.png')
    plt.close()
    
    # Exploration vs Exploitation histogram
    plt.figure(figsize=(6, 4))
    plt.hist(agent.exploration_history, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ['Exploit', 'Explore'])
    plt.title('Total Explore/Exploit Decisions')
    plt.xlabel('Decision Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/exploration_histogram.png')
    plt.close()
    
    print("Training completed. Results saved in 'results/' directory.")

if __name__ == "__main__":
    train()