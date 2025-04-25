import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import argparse
from agents.policy_gradient import PolicyNetwork

# Add arrows to show the best action in each cell


def visualize_q_table():
    # Use agent-specific results directory
    results_dir = os.path.join('results', 'q_learning')

    # Load Q-table and maze
    q_table_path = os.path.join(results_dir, 'q_table.npy')
    if not os.path.exists(q_table_path):
        print(
            f"Q-table file not found at {q_table_path}. Skipping visualization.")
        return

    q_table = np.load(q_table_path)

    # Create colormap
    cmap = ListedColormap(['white', 'black'])  # empty, wall

    plt.figure(figsize=(12, 6))

    # Try to load maze from either format
    maze_path = os.path.join(results_dir, 'maze.npy')
    maze_grid_path = os.path.join(results_dir, 'maze_grid.npy')

    if os.path.exists(maze_path):
        try:
            maze_data = np.load(maze_path, allow_pickle=True).item()
            maze_grid = maze_data['grid']
            plt.subplot(1, 2, 1)
            plt.imshow(maze_grid, cmap=cmap)  # Use maze_grid instead of maze
            plt.title('Maze Layout')
        except:
            print("Error loading maze file. Trying alternative format...")
            if os.path.exists(maze_grid_path):
                maze_grid = np.load(maze_grid_path)
                plt.subplot(1, 2, 1)
                # Use maze_grid instead of maze
                plt.imshow(maze_grid, cmap=cmap)
                plt.title('Maze Layout')
            else:
                print("Maze file not found. Skipping maze visualization.")
    elif os.path.exists(maze_grid_path):
        maze_grid = np.load(maze_grid_path)
        plt.subplot(1, 2, 1)
        plt.imshow(maze_grid, cmap=cmap)  # Use maze_grid instead of maze
        plt.title('Maze Layout')
    else:
        print("Maze file not found. Skipping maze visualization.")

    # Plot Q-table (max Q-values)
    plt.subplot(1, 2, 2)
    max_q = np.max(q_table, axis=-1)
    plt.imshow(max_q, cmap='viridis')
    plt.colorbar(label='Max Q-value')
    plt.title('Learned Q-values')

    # Add arrows to show policy (best action) in each cell
    height, width = q_table.shape[:2]
    for i in range(height):
        for j in range(width):
            best_action = np.argmax(q_table[i, j])
            if max_q[i, j] > -5:  # Only show arrows for cells with reasonable Q-values
                if best_action == 0:  # up
                    plt.arrow(j, i, 0, -0.3, head_width=0.1,
                              head_length=0.1, fc='k', ec='k')
                elif best_action == 1:  # down
                    plt.arrow(j, i, 0, 0.3, head_width=0.1,
                              head_length=0.1, fc='k', ec='k')
                elif best_action == 2:  # left
                    plt.arrow(j, i, -0.3, 0, head_width=0.1,
                              head_length=0.1, fc='k', ec='k')
                elif best_action == 3:  # right
                    plt.arrow(j, i, 0.3, 0, head_width=0.1,
                              head_length=0.1, fc='k', ec='k')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'q_table_visualization.png'))
    plt.show()


def visualize_policy_network():
    # Use agent-specific results directory
    results_dir = os.path.join('results', 'policy_gradient')

    # Load policy network and maze
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load maze from either format
    maze_path = os.path.join(results_dir, 'maze.npy')
    maze_grid_path = os.path.join(results_dir, 'maze_grid.npy')

    # Create colormap
    cmap = ListedColormap(['white', 'black'])  # empty, wall

    plt.figure(figsize=(12, 6))

    # Load maze layout
    if os.path.exists(maze_path):
        try:
            maze_data = np.load(maze_path, allow_pickle=True).item()
            maze_grid = maze_data['grid']
            height, width = maze_grid.shape
            plt.subplot(1, 2, 1)
            plt.imshow(maze_grid, cmap=cmap)
            plt.title('Maze Layout')
        except:
            print("Error loading maze file. Trying alternative format...")
            if os.path.exists(maze_grid_path):
                maze_grid = np.load(maze_grid_path)
                height, width = maze_grid.shape
                plt.subplot(1, 2, 1)
                plt.imshow(maze_grid, cmap=cmap)
                plt.title('Maze Layout')
            else:
                print("Maze file not found. Skipping maze visualization.")
                return
    elif os.path.exists(maze_grid_path):
        maze_grid = np.load(maze_grid_path)
        height, width = maze_grid.shape
        plt.subplot(1, 2, 1)
        plt.imshow(maze_grid, cmap=cmap)
        plt.title('Maze Layout')
    else:
        print("Maze file not found. Skipping maze visualization.")
        return

    # Load policy network
    policy_path = os.path.join(results_dir, 'policy_net.pth')
    if not os.path.exists(policy_path):
        print(
            f"Policy network file not found at {policy_path}. Skipping visualization.")
        return

    # Initialize policy network with the same architecture as in training
    policy = PolicyNetwork(
        input_dim=12,  # Match the actual training input dimension
        output_dim=4   # Four possible actions: up, down, left, right
    ).to(device)

    # Load trained weights
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    # Create a heatmap of policy decisions
    plt.subplot(1, 2, 2)

    # Assume goal is at bottom right for visualization
    goal_row, goal_col = height-1, width-1

    # Create a grid to store the confidence of the best action
    confidence_grid = np.zeros((height, width))

    # Add arrows to show policy in each cell
    for i in range(height):
        for j in range(width):
            if maze_grid[i, j] == 1:  # Skip walls
                continue

            # Compute normalized features (must match training)
            norm_row = i / height
            norm_col = j / width
            norm_goal_row = goal_row / height
            norm_goal_col = goal_col / width
            goal_distance = abs(i - goal_row) + abs(j - goal_col)
            norm_goal_distance = goal_distance / (height + width)
            dir_row = (goal_row - i) / height if i != goal_row else 0
            dir_col = (goal_col - j) / width if j != goal_col else 0

            # Create state tensor with all 12 features
            # First 7 features are the same as before
            # For the remaining 5 features (obstacles and visit penalty), use zeros as placeholders
            state_tensor = torch.FloatTensor([
                norm_row, norm_col, norm_goal_row, norm_goal_col,
                norm_goal_distance, dir_row, dir_col,
                0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder values for obstacle features and visit penalty
            ]).to(device)

            # Get action probabilities
            with torch.no_grad():
                output = policy(state_tensor)

                # Handle different possible output shapes
                if isinstance(output, tuple):
                    # If output is a tuple (e.g., network returns multiple values)
                    action_probs = output[0]
                else:
                    action_probs = output

                # Make sure it's the right shape
                if len(action_probs.shape) > 1:
                    # Remove batch dimension if present
                    action_probs = action_probs.squeeze(0)

                action_probs = action_probs.cpu().numpy()

                # If we have a single value instead of action probabilities
                if len(action_probs.shape) == 0 or action_probs.shape[0] == 1:
                    # Just use a fixed value for visualization
                    best_action = 0  # Default action
                    confidence_grid[i, j] = 1.0  # Full confidence
                else:
                    # Normal case with multiple action probabilities
                    best_action = np.argmax(action_probs)
                    confidence_grid[i, j] = action_probs[best_action]

            # Draw arrow for best action
            if best_action == 0:  # up
                plt.arrow(j, i, 0, -0.3, head_width=0.1,
                          head_length=0.1, fc='k', ec='k')
            elif best_action == 1:  # down
                plt.arrow(j, i, 0, 0.3, head_width=0.1,
                          head_length=0.1, fc='k', ec='k')
            elif best_action == 2:  # left
                plt.arrow(j, i, -0.3, 0, head_width=0.1,
                          head_length=0.1, fc='k', ec='k')
            elif best_action == 3:  # right
                plt.arrow(j, i, 0.3, 0, head_width=0.1,
                          head_length=0.1, fc='k', ec='k')

    # Plot confidence grid
    plt.imshow(confidence_grid, cmap='viridis')
    plt.colorbar(label='Action Confidence')
    plt.title('Learned Policy')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'policy_network_visualization.png'))
    plt.show()


if __name__ == "__main__":
    print("Select visualization type:")
    print("1. Q-Learning Q-table")
    print("2. Policy Gradient Policy Network")
    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        visualize_q_table()
    elif choice == '2':
        visualize_policy_network()
    else:
        print("Invalid choice. defaulting to Q-Learning Q-table.")
        visualize_q_table()