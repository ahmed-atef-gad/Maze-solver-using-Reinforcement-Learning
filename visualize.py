import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add arrows to show the best action in each cell
def visualize_q_table():
    # Load Q-table and maze
    q_table = np.load('results/q_table.npy')
    
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

if __name__ == "__main__":
    visualize_q_table()