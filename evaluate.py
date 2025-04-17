import numpy as np
from maze_env.maze import Maze
from maze_env.render import MazeRenderer
import pygame
import time
import sys

def evaluate():
    # Load trained Q-table
    q_table = np.load('results/q_table.npy')
    
    # Initialize environment with moving obstacles
    maze = Maze(width=10, height=10, num_moving_obstacles=3)
    renderer = MazeRenderer(maze)
    
    # Initialize font for score display
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)
    
    # Evaluation variables
    state = maze.start
    total_reward = 0
    steps = 0
    done = False
    
    visited_states = set()  # Track visited states to detect loops

    while not done and steps < 100:
        # Update moving obstacles
        maze.update_moving_obstacles()

        # Get best action from Q-table
        action = np.argmax(q_table[state])
        row, col = state

        # Try all actions until a valid move is found
        for _ in range(4):  # There are 4 possible actions
            if action == 0:   # up
                next_state = (row-1, col)
            elif action == 1:  # down
                next_state = (row+1, col)
            elif action == 2:  # left
                next_state = (row, col-1)
            elif action == 3:  # right
                next_state = (row, col+1)

            # Check if move is valid
            if maze.is_valid_position(*next_state):
                break
            else:
                # If invalid, try the next best action
                q_table[state][action] = -np.inf  # Penalize invalid action
                action = np.argmax(q_table[state])

        # If no valid move is found, the agent waits (does not move)
        if not maze.is_valid_position(*next_state):
            next_state = state

        # Penalize revisiting the same state
        if next_state in visited_states:
            print(f"Agent revisited state: {next_state}, applying penalty.")
            total_reward -= 1  # Apply a penalty for revisiting
        else:
            visited_states.add(next_state)

        # Add a fallback mechanism to force movement toward the goal
        if steps > 50 and not done:  # If stuck for too long
            print("Agent is stuck. Forcing movement toward the goal.")
            row, col = state
            goal_row, goal_col = maze.goal

            # Generate all possible moves toward the goal
            possible_moves = []
            if row < goal_row:
                possible_moves.append((row + 1, col))
            if row > goal_row:
                possible_moves.append((row - 1, col))
            if col < goal_col:
                possible_moves.append((row, col + 1))
            if col > goal_col:
                possible_moves.append((row, col - 1))

            # Filter valid moves
            valid_moves = [move for move in possible_moves if maze.is_valid_position(*move) and move not in visited_states]

            # Choose the first valid move or stay in place if no valid moves
            if valid_moves:
                next_state = valid_moves[0]
            else:
                next_state = state

        # Debug: Print agent's position and goal
        print(f"Agent Position: {next_state}, Goal: {maze.goal}")

        # Get reward
        reward = maze.get_reward(*next_state)
        done = (next_state == maze.goal)

        # Debug: Check if the agent reached the goal
        if done:
            print(f"Agent reached the goal at step {steps}!")

        # Update metrics
        state = next_state
        total_reward += reward
        steps += 1

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