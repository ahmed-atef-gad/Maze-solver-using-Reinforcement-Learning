import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
import pygame
import time
import sys

def evaluate():
    # Load trained Q-table
    q_table = np.load('results/q_table.npy')
    
    # Initialize environment
    maze = Maze(width=10, height=10, num_moving_obstacles=0)  # Disable moving obstacles
    renderer = MazeRenderer(maze)
    
    # Initialize font for score display
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)
    
    # Evaluation variables
    state = maze.start
    total_reward = 0
    steps = 0
    done = False
    visited_states = set([state])  # Initialize with start state
    blocked_actions = {}  # Track actions that lead to invalid states
    
    max_steps = 100
    
    while not done and steps < max_steps:
        # Debug print current state
        print(f"Agent Position: {state}, Goal: {maze.goal}")
        
        # Get action from the Q-table
        row, col = state
        
        # Choose action based on Q-table, but avoid previously invalid moves
        valid_actions = np.ones(4, dtype=bool)  # All actions start as valid
        
        # Mark previously blocked actions as invalid
        if state in blocked_actions:
            for blocked_action in blocked_actions[state]:
                valid_actions[blocked_action] = False
        
        # Get the best valid action
        if np.any(valid_actions):
            # Filter Q-values to only consider valid actions
            masked_q_values = q_table[state].copy()
            masked_q_values[~valid_actions] = -np.inf
            action = np.argmax(masked_q_values)
        else:
            # All actions have been tried and failed, pick random one
            print("All known actions blocked. Taking random action.")
            action = np.random.randint(0, 4)
        
        # Execute the selected action
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
            # Mark this action as blocked for future reference
            if state not in blocked_actions:
                blocked_actions[state] = []
            blocked_actions[state].append(action)
            
            # Apply penalty and stay in place
            reward = -5.0
            next_state = state
            print(f"Invalid move attempted. Staying at {state}")
        else:
            # Valid move, get reward
            reward = maze.get_reward(*next_state)
            print(f"Moving to {next_state} with reward {reward}")
            
            # Check if we've reached the goal
            done = (next_state == maze.goal)
            
            # Apply small penalty for revisiting states
            if next_state in visited_states:
                print(f"Revisited state: {next_state}, applying penalty.")
                reward -= 1
            
            # Add to visited states
            visited_states.add(next_state)
        
        # Update metrics
        state = next_state
        total_reward += reward
        steps += 1
        
        # If stuck (making no progress), temporarily decrease the Q-value
        if steps > 20 and len(visited_states) < 10:
            print("Agent appears stuck. Penalizing current action preference.")
            q_table[state][action] *= 0.8  # Temporarily reduce this action's value
        
        # Render
        renderer.update(state)
        
        # Display score and steps
        score_text = font.render(f"Score: {total_reward:.1f}", True, (0, 0, 0))
        steps_text = font.render(f"Steps: {steps}", True, (0, 0, 0))
        renderer.screen.blit(score_text, (10, 10))
        renderer.screen.blit(steps_text, (10, 40))
        
        pygame.display.flip()
        time.sleep(0.3)  # Slow down for visualization
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close()
                sys.exit()
        
        if done:
            print(f"Agent reached the goal in {steps} steps!")
    
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
    evaluate()