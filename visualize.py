import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import argparse
from agents.policy_gradient import PolicyNetwork

# Add arrows to show the best action in each cell
def visualize_q_table():
    # Load Q-table and maze
    q_table_path = os.path.join('results', 'q_table.npy')
    if not os.path.exists(q_table_path):
        print("Q-table file not found. Skipping visualization.")
        return
        
    q_table = np.load(q_table_path)
    
    # Create colormap
    cmap = ListedColormap(['white', 'black'])  # empty, wall
    
    plt.figure(figsize=(12, 6))
    
    # Try to load maze from either format
    maze_path = os.path.join('results', 'maze.npy')
    maze_grid_path = os.path.join('results', 'maze_grid.npy')
    
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
                plt.imshow(maze_grid, cmap=cmap)  # Use maze_grid instead of maze
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
                    plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 1:  # down
                    plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 2:  # left
                    plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif best_action == 3:  # right
                    plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    plt.tight_layout()
    plt.savefig('results/q_table_visualization.png')
    plt.show()

def visualize_policy_network():
    # Load policy network and maze
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load maze from either format
    maze_path = os.path.join('results', 'maze.npy')
    maze_grid_path = os.path.join('results', 'maze_grid.npy')
    
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
    policy_path = os.path.join('results', 'policy_net.pth')
    if not os.path.exists(policy_path):
        print("Policy network file not found. Skipping visualization.")
        return
    
    # Initialize policy network with the same architecture as in training
    policy = PolicyNetwork(
        input_dim=4,  # State representation: (row, col, goal_row, goal_col)
        hidden_dim=128,  # Using default value, should match training
        output_dim=4  # Four possible actions: up, down, left, right
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
                
            # Create state tensor
            state_tensor = torch.FloatTensor(
                [i/height, j/width, goal_row/height, goal_col/width]
            ).to(device)
            
            # Get action probabilities
            with torch.no_grad():
                action_probs = policy(state_tensor).cpu().numpy()
            
            # Store confidence (probability) of best action
            best_action = np.argmax(action_probs)
            confidence_grid[i, j] = action_probs[best_action]
            
            # Draw arrow for best action
            if best_action == 0:  # up
                plt.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 1:  # down
                plt.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 2:  # left
                plt.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            elif best_action == 3:  # right
                plt.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    
    # Plot confidence grid
    plt.imshow(confidence_grid, cmap='viridis')
    plt.colorbar(label='Action Confidence')
    plt.title('Learned Policy')
    
    plt.tight_layout()
    plt.savefig('results/policy_network_visualization.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trained agent')
    parser.add_argument('--agent', type=str, default='q_learning', choices=['q_learning', 'policy_gradient'],
                        help='Agent type to visualize (q_learning or policy_gradient)')
    args = parser.parse_args()
    
    if args.agent == 'q_learning':
        visualize_q_table()
    elif args.agent == 'policy_gradient':
        visualize_policy_network()
    else:
        print(f"Unknown agent type: {args.agent}")
        print("Available options: q_learning, policy_gradient")