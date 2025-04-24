import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
from agents.q_learning import QLearningAgent
from agents.policy_gradient import PolicyGradientAgent
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
import argparse

def train(agent_type="q_learning"):
    # Create agent-specific results directory if it doesn't exist
    results_dir = os.path.join('results', agent_type)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment with moving obstacles
    # Use a different seed each time or no seed for true randomness
    # use 4 obstacles in q learning, 2 in policy gradient for best results
    maze = Maze(width=10, height=10, num_moving_obstacles=4, use_seed=None)
    
    # Initialize agent based on type
    if agent_type == "q_learning":
        agent = QLearningAgent(
            state_space=(maze.height, maze.width),
            action_space=4,
            config_path='configs/default.yaml'
        )
    elif agent_type == "policy_gradient":
        agent = PolicyGradientAgent(
            state_space=(maze.height, maze.width),
            action_space=4,
            config_path='configs/default.yaml'
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    
    # Training metrics
    rewards_history = []
    steps_history = []
    exploration_rates = []
    
    # Training loop with agent-specific episode count
    episodes = config['q_learning_episodes'] if agent_type == 'q_learning' else config['pg_episodes']
    for episode in tqdm(range(episodes), desc="Training"):
        state = maze.start
        total_reward = 0
        steps = 0
        done = False
        
        # Track exploration rate for this episode
        exploration_rates.append(agent.epsilon)
        
        stuck_counter = 0  # Track how many steps the agent is stuck
        visited_states = set()

        # In the training loop, modify the section where the agent takes an action:
        
        while not done and steps < 300:
            # Get action from agent
            if agent_type == "policy_gradient":
                action = agent.get_action(state, training=True, goal=maze.goal, maze=maze)
            else:
                action = agent.get_action(state)
            
            # Update moving obstacles first
            maze.update_moving_obstacles()
            
            # Get current obstacle positions
            current_obstacle_positions = [(obs_row, obs_col) for (obs_row, obs_col, _, _, _) in maze.moving_obstacles]
            
            # Predict the next positions of moving obstacles
            predicted_obstacle_positions = [
                (obs_row + dr, obs_col + dc) for (obs_row, obs_col, dr, dc, _) in maze.moving_obstacles
            ]
            
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
            
            # Check for path crossing - if agent and obstacle would swap positions
            path_crossing = False
            for i, (obs_row, obs_col, dr, dc, _) in enumerate(maze.moving_obstacles):
                if (obs_row, obs_col) == next_state and (obs_row + dr, obs_col + dc) == state:
                    path_crossing = True
                    break
            
            # Check if move is valid and not colliding with obstacles
            if not maze.is_valid_position(*next_state) or next_state in current_obstacle_positions or next_state in predicted_obstacle_positions or path_crossing:
                # Strong penalty for invalid moves or potential collisions
                reward = -10.0
                next_state = state  # Stay in place
                done = False
            else:
                # In the training loop, modify the reward calculation:
                
                # Get reward based on new position
                reward = maze.get_reward(*next_state)
                
                # Extra positive reward for getting closer to goal
                current_dist = abs(row - maze.goal[0]) + abs(col - maze.goal[1])
                new_dist = abs(next_state[0] - maze.goal[0]) + abs(next_state[1] - maze.goal[1])
                if new_dist < current_dist:
                    reward += 1.5  # Increased bonus for progress (was 0.5)
                elif new_dist > current_dist:
                    reward -= 1.0  # Penalty for moving away from goal
                
                # Check if goal reached
                if next_state == maze.goal:
                    reward += 50.0  # Extra bonus for reaching the goal
                    done = True
                
                # Small penalty for staying in place (to encourage movement when safe)
                if next_state == state:
                    reward -= 1.0

            # Detect if agent is stuck
            if state in visited_states:
                stuck_counter += 1
            else:
                stuck_counter = 0
            visited_states.add(state)

            if stuck_counter > 20:  # Reset agent if stuck for too long
                print("Agent is stuck. Resetting position.")
                state = maze.start
                stuck_counter = 0

            # Update agent
            if agent_type == "policy_gradient":
                agent.update(state, action, reward, next_state, done, goal=maze.goal, maze=maze)
            else:
                agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # Record episode metrics
        rewards_history.append(total_reward)
        steps_history.append(steps)
    
    # Save training results
    np.save(os.path.join(results_dir, 'rewards.npy'), rewards_history)
    np.save(os.path.join(results_dir, 'steps.npy'), steps_history)
    np.save(os.path.join(results_dir, 'exploration_rates.npy'), exploration_rates)
    np.save(os.path.join(results_dir, 'exploration_history.npy'), np.array(agent.exploration_history))
    
    # Save agent-specific data
    if agent_type == "q_learning":
        np.save(os.path.join(results_dir, 'q_table.npy'), agent.q_table)
    elif agent_type == "policy_gradient":
        agent.save(os.path.join(results_dir, 'policy_net.pth'))
    
    # Save the complete maze state as a dictionary
    maze_state = {
        'grid': maze.grid,
        'start': maze.start,
        'goal': maze.goal,
        'width': maze.width,
        'height': maze.height,
        'moving_obstacles': maze.moving_obstacles
    }
    np.save(os.path.join(results_dir, 'maze.npy'), maze_state)
    
    # Also save just the grid for backward compatibility
    np.save(os.path.join(results_dir, 'maze_grid.npy'), maze.grid)
    
    # Plot training results
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Training Results for {agent_type.replace("_", " ").title()} Agent')
    
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
    plt.savefig(os.path.join(results_dir, 'training_plot.png'))
    plt.close()
    
    # Exploration vs Exploitation histogram
    plt.figure(figsize=(6, 4))
    plt.hist(agent.exploration_history, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ['Exploit', 'Explore'])
    plt.title('Total Explore/Exploit Decisions')
    plt.xlabel('Decision Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'exploration_histogram.png'))
    plt.close()
    
    print(f"Training completed. Results saved in '{results_dir}' directory.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a reinforcement learning agent on the maze environment')
    parser.add_argument('--agent', type=str, required=True, choices=['q_learning', 'policy_gradient'],
                        help='Type of agent to train (q_learning or policy_gradient)')
    
    try:
        args = parser.parse_args()
        print(f"Training {args.agent} agent...")
        train(agent_type=args.agent)
    except SystemExit:
        print("\nError: You must specify an agent type using --agent")
        print("Available agents: q_learning, policy_gradient")
        print("\nExample usage:")
        print("  python train.py --agent q_learning")
        print("  python train.py --agent policy_gradient")