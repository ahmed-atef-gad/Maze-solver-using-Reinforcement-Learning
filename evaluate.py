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

    # Use agent-specific results directory
    results_dir = os.path.join('results', agent_type)

    # Load the exact maze state from training
    maze_grid_path = os.path.join(results_dir, 'maze.npy')
    if os.path.exists(maze_grid_path):
        maze_state = np.load(maze_grid_path, allow_pickle=True).item()
        maze.grid = maze_state['grid']
        maze.start = maze_state['start']
        maze.goal = maze_state['goal']
        maze.moving_obstacles = maze_state['moving_obstacles']

    # Create and load agent based on type
    if agent_type == 'q_learning':
        # Load trained Q-table
        q_table_path = os.path.join(results_dir, 'q_table.npy')
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
        policy_path = os.path.join(results_dir, 'policy_net.pth')
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

    # Create a safety buffer around obstacles - positions to avoid
    def get_danger_zones(obstacles):
        danger_zones = set()
        for obs_row, obs_col, dr, dc, _ in obstacles:
            # Current position
            danger_zones.add((obs_row, obs_col))

            # Predicted next position
            danger_zones.add((obs_row + dr, obs_col + dc))

            # Buffer zone around current position (adjacent cells)
            for buffer_dr, buffer_dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                buffer_pos = (obs_row + buffer_dr, obs_col + buffer_dc)
                danger_zones.add(buffer_pos)

            # Buffer zone around predicted position
            for buffer_dr, buffer_dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                buffer_pos = (obs_row + dr + buffer_dr,
                              obs_col + dc + buffer_dc)
                danger_zones.add(buffer_pos)

        return danger_zones

    # Function to check if an obstacle will move into the agent's position
    def will_obstacle_enter_agent_position(agent_pos, obstacles):
        for obs_row, obs_col, dr, dc, _ in obstacles:
            next_obs_pos = (obs_row + dr, obs_col + dc)
            if next_obs_pos == agent_pos:
                return True
        return False

    # Inside the evaluation loop, improve the collision detection:
    while not done and steps < max_steps:
        # Update moving obstacles before the agent moves
        maze.update_moving_obstacles()

        # Track this state visit
        if state in visited_states:
            visited_states[state] += 1
        else:
            visited_states[state] = 1

        # Check if any obstacle is about to move into the agent's current position
        # This is critical to prevent obstacles from entering the agent's cell
        if will_obstacle_enter_agent_position(state, maze.moving_obstacles):
            print(
                f"Obstacle approaching agent at {state}. Must move to avoid collision.")
            # Force the agent to move to a safe position
            safe_moves = []
            row, col = state

            for test_action in range(4):
                if test_action == 0:   # up
                    test_next_state = (row-1, col)
                elif test_action == 1:  # down
                    test_next_state = (row+1, col)
                elif test_action == 2:  # left
                    test_next_state = (row, col-1)
                elif test_action == 3:  # right
                    test_next_state = (row, col+1)

                # Get current obstacle positions
                current_obstacle_positions = [(obs_row, obs_col) for (
                    obs_row, obs_col, _, _, _) in maze.moving_obstacles]

                # Predict the next positions of moving obstacles
                predicted_obstacle_positions = [
                    (obs_row + dr, obs_col + dc) for (obs_row, obs_col, dr, dc, _) in maze.moving_obstacles
                ]

                # Check if this move is safe
                if (maze.is_valid_position(*test_next_state) and
                    test_next_state not in current_obstacle_positions and
                        test_next_state not in predicted_obstacle_positions):

                    # Calculate distance to goal as a heuristic
                    goal_dist = abs(
                        test_next_state[0] - maze.goal[0]) + abs(test_next_state[1] - maze.goal[1])
                    # Calculate safety score (distance to nearest obstacle)
                    safety_score = min(
                        abs(test_next_state[0] - obs_row) +
                        abs(test_next_state[1] - obs_col)
                        for obs_row, obs_col, _, _, _ in maze.moving_obstacles
                    ) if maze.moving_obstacles else 10

                    safe_moves.append(
                        (test_action, test_next_state, goal_dist, safety_score))

            if safe_moves:
                # Sort by safety first, then by goal distance
                safe_moves.sort(key=lambda move: (-move[3], move[2]))
                action, next_state, _, _ = safe_moves[0]
                print(
                    f"Emergency move to {next_state} to avoid incoming obstacle")

                # Get reward based on new position
                reward = maze.get_reward(*next_state)

                # Update state and path history
                state = next_state
                path_history.append(state)
                steps += 1
                total_reward += reward

                # Render only if visualization is enabled
                if render:
                    renderer.update(
                        agent_pos=state,
                        steps=steps,
                        score=total_reward,
                        status="Emergency Move"
                    )
                    pygame.display.flip()
                    time.sleep(0.3)  # Slow down for visualization

                # Check for quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        renderer.close()
                        sys.exit()

                # Skip the rest of this iteration
                continue
            else:
                print(
                    f"WARNING: No safe moves available to avoid incoming obstacle at {state}")
                # We'll continue with normal logic, but this is a dangerous situation

        # Get action using the agent's policy WITHOUT exploration
        if agent_type == 'policy_gradient':
            action = agent.get_action(
                state, training=False, goal=maze.goal, maze=maze)
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

        # Get current obstacle positions and their trajectories
        current_obstacle_positions = [(obs_row, obs_col) for (
            obs_row, obs_col, _, _, _) in maze.moving_obstacles]

        # Predict the next positions of moving obstacles
        predicted_obstacle_positions = [
            (obs_row + dr, obs_col + dc) for (obs_row, obs_col, dr, dc, _) in maze.moving_obstacles
        ]

        # Get all danger zones - positions to avoid
        danger_zones = get_danger_zones(maze.moving_obstacles)

        # Check for path crossing - if agent and obstacle would swap positions
        path_crossing = False
        for i, (obs_row, obs_col, dr, dc, _) in enumerate(maze.moving_obstacles):
            if (obs_row, obs_col) == next_state and (obs_row + dr, obs_col + dc) == state:
                path_crossing = True
                break

            # Also check if agent would move into the path of an obstacle
            # This handles the case where they would pass through each other
            if next_state == (obs_row + dr, obs_col + dc) and state != (obs_row, obs_col):
                path_crossing = True
                break

        # Check for collision with current and predicted positions of moving obstacles
        collision = (next_state in predicted_obstacle_positions or
                     next_state in current_obstacle_positions or
                     path_crossing or
                     next_state in danger_zones)

        if collision:
            print(
                f"Predicted collision or danger at {next_state}. Finding alternative path.")
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

                # Check if alternative move is valid and not colliding with obstacles
                alt_path_crossing = False
                for i, (obs_row, obs_col, dr, dc, _) in enumerate(maze.moving_obstacles):
                    if (obs_row, obs_col) == alt_next_state and (obs_row + dr, obs_col + dc) == state:
                        alt_path_crossing = True
                        break
                    # Check for passing through each other
                    if alt_next_state == (obs_row + dr, obs_col + dc) and state != (obs_row, obs_col):
                        alt_path_crossing = True
                        break

                if (maze.is_valid_position(*alt_next_state) and
                    alt_next_state not in predicted_obstacle_positions and
                    alt_next_state not in current_obstacle_positions and
                    alt_next_state not in danger_zones and
                        not alt_path_crossing):
                    # Calculate distance to goal as a heuristic
                    goal_dist = abs(
                        alt_next_state[0] - maze.goal[0]) + abs(alt_next_state[1] - maze.goal[1])
                    # Calculate safety score (distance to nearest obstacle)
                    safety_score = min(
                        abs(alt_next_state[0] - obs_row) +
                        abs(alt_next_state[1] - obs_col)
                        for obs_row, obs_col, _, _, _ in maze.moving_obstacles
                    ) if maze.moving_obstacles else 10

                    safe_moves.append(
                        (alt_action, alt_next_state, goal_dist, safety_score))

            # Prioritize moves that are safe and lead toward the goal
            if safe_moves:
                # Sort by safety first, then by goal distance
                safe_moves.sort(key=lambda move: (-move[3], move[2]))
                action, next_state, _, _ = safe_moves[0]
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
            # Double-check for collision with current obstacle positions and danger zones
            if (next_state in current_obstacle_positions or
                next_state in predicted_obstacle_positions or
                    next_state in danger_zones):
                print(
                    f"Collision detected at {next_state}. Staying at {state}")
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
                        print(
                            f"Detected cycle of length {cycle_len}. Breaking out.")
                        # Find a state we haven't visited much
                        least_visited = None
                        min_visits = float('inf')

                        # Check all neighboring states
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            neighbor = (state[0] + dr, state[1] + dc)
                            if 0 <= neighbor[0] < maze.height and 0 <= neighbor[1] < maze.width:
                                if (maze.is_valid_position(*neighbor) and
                                    neighbor not in current_obstacle_positions and
                                    neighbor not in predicted_obstacle_positions and
                                        neighbor not in danger_zones):
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
        final_text = font.render(
            f"GOAL REACHED! Final Score: {total_reward:.1f}", True, (0, 0, 255))
    else:
        final_text = font.render(
            f"FAILED TO REACH GOAL. Final Score: {total_reward:.1f}", True, (255, 0, 0))
    renderer.screen.blit(final_text, (renderer.screen_width //
                         2 - 150, renderer.screen_height//2))
    pygame.display.flip()

    # Wait for a moment before closing
    time.sleep(3)

    # Print final results
    print(
        f"Evaluation completed in {steps} steps with total score: {total_reward:.1f}")

    return total_reward, steps


if __name__ == "__main__":
    print("Please specify the agent type for evaluation.")
    print("1. Q-Learning")
    print("2. Policy Gradient")
    choice = input("Enter 1 or 2: ")
    if choice == '1':
        agent_type = 'q_learning'
    elif choice == '2':
        agent_type = 'policy_gradient'
    else:
        print("Invalid choice. defaulting to Q-Learning.")
        agent_type = 'q_learning'
    evaluate(agent_type=agent_type, render=True)
