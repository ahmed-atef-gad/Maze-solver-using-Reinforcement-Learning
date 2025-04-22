import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
from agents.q_learning import QLearningAgent
from agents.policy_gradient import PolicyGradientAgent
import pygame
import time
import sys
import os
import random
import argparse
import torch

def evaluate(agent_type='q_learning', render=True):
    # Initialize environment with dummy obstacles (will load actual state)
    maze = Maze(width=10, height=10, num_moving_obstacles=0, use_seed=None)
    
    # Load the exact maze state from training
    maze_grid_path = os.path.join('results', 'maze.npy')
    if os.path.exists(maze_grid_path):
        maze_state = np.load(maze_grid_path, allow_pickle=True).item()
        maze.grid = maze_state['grid']
        maze.start = maze_state['start']
        maze.goal = maze_state['goal']
        maze.moving_obstacles = maze_state['moving_obstacles']
    
    # Create and load agent based on type
    if agent_type == 'q_learning':
        # Load trained Q-table
        q_table_path = os.path.join('results', 'q_table.npy')
        if not os.path.exists(q_table_path):
            print(f"Error: Q-table file not found at {q_table_path}")
            return
            
        q_table = np.load(q_table_path)
        agent = QLearningAgent(
            state_space=(maze.height, maze.width),
            action_space=4,
            config_path='configs/default.yaml'
        )
        agent.q_table = q_table  # Load the trained Q-table
    elif agent_type == 'policy_gradient':
        # Load trained policy network
        policy_path = os.path.join('results', 'policy_net.pth')
        if not os.path.exists(policy_path):
            print(f"Error: Policy network file not found at {policy_path}")
            return
            
        agent = PolicyGradientAgent(
            state_space=(maze.height, maze.width),
            action_space=4,
            config_path='configs/default.yaml'
        )
        agent.load(policy_path)  # Load the trained policy network
    else:
        print(f"Error: Unknown agent type '{agent_type}'")
        return
    
    renderer = MazeRenderer(maze)
    
    # Initialize font for score display
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)
    
    # Initialize variables for evaluation
    state = maze.start
    done = False
    total_reward = 0
    steps = 0
    visited_states = {}  # Track visited states and count
    stuck_count = 0
    failed_actions = {}  # Track actions that failed at each state
    
    # Keep track of the path taken
    path_history = [state]
    
    # For Q-learning, we can create a temporary Q-table for this evaluation run
    # This allows us to modify the Q-table during evaluation without affecting the original
    if agent_type == 'q_learning':
        temp_q_table = np.copy(agent.q_table)
    
    max_steps = 100
    
    while not done and steps < max_steps:
        # Update moving obstacles before the agent moves
        maze.update_moving_obstacles()

        # Debug: Print positions of moving obstacles
        print("Moving obstacles positions:", [(obs_row, obs_col) for (obs_row, obs_col, _, _, _) in maze.moving_obstacles])

        # Get action using the agent's policy WITHOUT exploration
        if agent_type == 'policy_gradient':
            action = agent.get_action(state, training=False, goal=maze.goal)
        else:
            action = agent.get_action(state, training=False)

        # Determine the next state based on the chosen action
        row, col = state
        if action == 0:   # up
            next_state = (row-1, col)
        elif action == 1:  # down
            next_state = (row+1, col)
        elif action == 2:  # left
            next_state = (row, col-1)
        elif action == 3:  # right
            next_state = (row, col+1)

        # Get current obstacle positions
        current_obstacle_positions = [(obs_row, obs_col) for (obs_row, obs_col, _, _, _) in maze.moving_obstacles]
        
        # Predict the next positions of moving obstacles
        predicted_obstacle_positions = [
            (obs_row + dr, obs_col + dc) for (obs_row, obs_col, dr, dc, _) in maze.moving_obstacles
        ]

        # Check for collision with current and predicted positions of moving obstacles
        collision = next_state in predicted_obstacle_positions or next_state in current_obstacle_positions

        if collision:
            print(f"Predicted collision at {next_state}. Finding alternative path.")
            # Find an alternative action to avoid the obstacle
            possible_actions = [a for a in range(4) if a != action]
            safe_moves = []
            for alt_action in possible_actions:
                if alt_action == 0:   # up
                    alt_next_state = (row-1, col)
                elif alt_action == 1:  # down
                    alt_next_state = (row+1, col)
                elif alt_action == 2:  # left
                    alt_next_state = (row, col-1)
                elif alt_action == 3:  # right
                    alt_next_state = (row, col+1)

                # Check if alternative move is valid and not colliding with predicted positions
                if (maze.is_valid_position(*alt_next_state) and 
                    alt_next_state not in predicted_obstacle_positions and
                    alt_next_state not in current_obstacle_positions):
                    safe_moves.append((alt_action, alt_next_state))

            # Prioritize moves that are farthest from predicted obstacle positions
            if safe_moves:
                safe_moves.sort(key=lambda move: min(abs(move[1][0] - obs_row) + abs(move[1][1] - obs_col) 
                                                  for (obs_row, obs_col) in predicted_obstacle_positions), 
                                reverse=True)
                action, next_state = safe_moves[0]
                print(f"Found safe alternative move to {next_state}")
            else:
                # If no valid alternative found, stay in place
                print(f"No safe alternative found. Staying at {state}")
                next_state = state
                reward = -5.0  # Penalty for being stuck
                stuck_count += 1
                
                # Still count this as a step since the agent made a decision
                steps += 1
                total_reward += reward
                
                # Render only if visualization is enabled
                if render:
                    renderer.update(
                        agent_pos=state,
                        steps=steps,
                        score=total_reward,
                        status="Stuck" if stuck_count > 0 else "In Progress"
                    )
                    pygame.display.flip()
                    time.sleep(0.3)  # Slow down for visualization
                
                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        sys.exit()
                
                # Skip the rest of this iteration and continue to the next step
                continue
        
        # Check if move is valid
        if not maze.is_valid_position(*next_state):
            print(f"Invalid move attempted. Staying at {state}")
            next_state = state
            reward = -10.0
            stuck_count += 1
        else:
            # Check for collision with current obstacle positions (double-check)
            if next_state in current_obstacle_positions:
                print(f"Collision detected at {next_state}. Staying at {state}")
                next_state = state
                reward = -20.0  # Larger penalty for collision
                stuck_count += 1
            else:
                # Get reward based on new position
                reward = maze.get_reward(*next_state)
                print(f"Moving to {next_state} with reward {reward}")

                # Check if goal reached
                if next_state == maze.goal:
                    print(f"Agent reached the goal in {steps+1} steps!")
                    total_reward += reward
                    done = True
                else:
                    total_reward += reward

                # Reset stuck counter only if we're moving to a new state
                if next_state != state:
                    stuck_count = 0

        # Update state and path history
        state = next_state
        path_history.append(state)
        steps += 1
        
        # Detect cycles in the path and break them
        if len(path_history) > 10:
            # Check for cycles of length 2-4
            for cycle_len in range(2, 5):
                if len(path_history) >= cycle_len * 2:
                    recent_path = path_history[-cycle_len:]
                    previous_path = path_history[-(cycle_len*2):-cycle_len]
                    
                    if recent_path == previous_path:
                        print(f"Detected cycle of length {cycle_len}. Breaking out.")
                        # Find a state we haven't visited much
                        least_visited = None
                        min_visits = float('inf')
                        
                        # Check all neighboring states
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            neighbor = (state[0] + dr, state[1] + dc)
                            if 0 <= neighbor[0] < maze.height and 0 <= neighbor[1] < maze.width:
                                if maze.is_valid_position(*neighbor) and neighbor not in current_obstacle_positions:
                                    visits = visited_states.get(neighbor, 0)
                                    if visits < min_visits:
                                        min_visits = visits
                                        least_visited = neighbor
                        
                        # If we found a less-visited neighbor, force movement there
                        if least_visited:
                            state = least_visited
                            print(f"Breaking cycle by moving to {state}")
                            path_history.append(state)
                            stuck_count = 0
                            break
        
        # Render only if visualization is enabled
        if render:
            status = "Success!" if done else "Failed!" if steps >= max_steps else "In Progress"
            renderer.update(
                agent_pos=state,
                steps=steps,
                score=total_reward,
                status=status
            )
            pygame.display.flip()
            time.sleep(0.3)  # Slow down for visualization
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                sys.exit()
    
    # Final score display
    if done:
        final_text = font.render(f"GOAL REACHED! Final Score: {total_reward:.1f}", True, (0, 0, 255))
    else:
        final_text = font.render(f"FAILED TO REACH GOAL. Final Score: {total_reward:.1f}", True, (255, 0, 0))
    renderer.screen.blit(final_text, (renderer.screen_width//2 - 150, renderer.screen_height//2))
    pygame.display.flip()
    
    # Wait for a moment before closing
    time.sleep(3)
    
    # Print final results
    print(f"Evaluation completed in {steps} steps with total score: {total_reward:.1f}")
    
    return total_reward, steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained agent on the maze environment')
    parser.add_argument('--agent', type=str, required=True, choices=['q_learning', 'policy_gradient'],
                        help='Type of agent to evaluate (q_learning or policy_gradient)')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    
    try:
        args = parser.parse_args()
        print(f"Evaluating {args.agent} agent...")
        evaluate(agent_type=args.agent, render=not args.no_render)
    except SystemExit:
        print("\nError: You must specify an agent type using --agent")
        print("Available agents: q_learning, policy_gradient")
        print("\nExample usage:")
        print("  python evaluate.py --agent q_learning")
        print("  python evaluate.py --agent policy_gradient")