import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
from agents.q_learning import QLearningAgent
import pygame
import time
import sys
import os
import random

def evaluate(render=True):
    # Load trained Q-table
    q_table = np.load(os.path.join('results', 'q_table.npy'))
    
    # Initialize environment
    maze = Maze(width=10, height=10, num_moving_obstacles=3, use_seed=None)
    # Load the exact maze grid from training
    maze_grid_path = os.path.join('results', 'maze.npy')
    if os.path.exists(maze_grid_path):
        maze_state = np.load(maze_grid_path, allow_pickle=True).item()
        maze.grid = maze_state['grid']
    
    # Create agent with loaded Q-table
    agent = QLearningAgent(
        state_space=(maze.height, maze.width),
        action_space=4,
        config_path='configs/default.yaml'
    )
    agent.q_table = q_table  # Load the trained Q-table
    
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
    
    # Create a temporary Q-table for this evaluation run
    temp_q_table = np.copy(q_table)
    
    max_steps = 100
    
    while not done and steps < max_steps:
        # Update moving obstacles before the agent moves
        maze.update_moving_obstacles()
        
        # Determine action based on current situation
        if stuck_count > 8:
            # Force exploration when stuck
            if random.random() < 0.7:  # 70% chance of random action when stuck
                print("Taking random action to break out of loop")
                action = random.randint(0, 3)
            else:
                # Use the temporary Q-table for action selection
                action = np.argmax(temp_q_table[state])
        else:
            # Use the original Q-table with some randomness
            if random.random() < 0.1:  # 10% exploration
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_table[state])
        
        # If this action has failed before at this state, try another one
        if state in failed_actions and action in failed_actions[state]:
            # Get actions that haven't failed
            possible_actions = [a for a in range(4) if a not in failed_actions[state]]
            if possible_actions:
                action = random.choice(possible_actions)
                print(f"Avoiding failed action, trying {action} instead")
        
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
            print(f"Invalid move attempted. Staying at {state}")
            
            # Record this failed action
            if state not in failed_actions:
                failed_actions[state] = set()
            failed_actions[state].add(action)
            
            # Heavily penalize this action in the temporary Q-table
            temp_q_table[state][action] -= 20.0
            
            # Stay in place
            next_state = state
            reward = -10.0
            stuck_count += 1
        else:
            # Check for collision with moving obstacles
            collision = False
            for (obs_row, obs_col, _, _, _) in maze.moving_obstacles:
                if next_state[0] == obs_row and next_state[1] == obs_col:
                    print(f"Collision with obstacle at {next_state}! Game over.")
                    total_reward -= 50
                    done = True
                    collision = True
                    break
            
            if not collision:
                # Get reward based on new position
                reward = maze.get_reward(*next_state)
                
                # Check if we're revisiting a state (to prevent loops)
                if next_state in visited_states:
                    visit_count = visited_states[next_state]
                    # Exponential penalty for revisits
                    penalty = -2.0 * (2 ** min(visit_count, 5))
                    print(f"Revisited state: {next_state} ({visit_count} times), applying penalty {penalty}.")
                    reward += penalty
                    
                    # Also penalize this state in the temporary Q-table
                    temp_q_table[state][action] -= visit_count * 2.0
                
                # Check if goal reached
                if next_state == maze.goal:
                    print(f"Moving to {next_state} with reward {reward}")
                    print(f"Agent reached the goal in {steps+1} steps!")
                    total_reward += reward
                    done = True
                else:
                    print(f"Moving to {next_state} with reward {reward}")
                    total_reward += reward
                
                # Reset stuck counter only if we're moving to a new state
                if next_state != state:
                    stuck_count = 0
            
            # Update visited states counter
            if next_state in visited_states:
                visited_states[next_state] += 1
            else:
                visited_states[next_state] = 1
        
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
                                if maze.is_valid_position(*neighbor):
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
            renderer.update(state)
            score_text = font.render(f"Score: {total_reward:.1f}", True, (216, 229, 24))
            steps_text = font.render(f"Steps: {steps}", True, (216, 229, 24))
            renderer.screen.blit(score_text, (10, 10))
            renderer.screen.blit(steps_text, (10, 40))
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
    
    print(f"Evaluation completed in {steps} steps with total score: {total_reward:.1f}")
    time.sleep(3)  # Show final message for 3 seconds
    renderer.close()

if __name__ == "__main__":
    evaluate(render=True)  # Set to False for headless evaluation