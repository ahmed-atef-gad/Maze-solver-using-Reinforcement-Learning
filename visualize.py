import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add arrows to show the best action in each cell
def visualize_q_table():
    # Load Q-table and maze
    q_table = np.load('results/q_table.npy')
    maze = np.load('results/maze.npy') if 'results/maze.npy' in os.listdir('results') else None
    
    # Create colormap
    cmap = ListedColormap(['white', 'black'])  # empty, wall
    
    plt.figure(figsize=(12, 6))
    
    # Plot maze
    if maze is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(maze, cmap=cmap)
        plt.title('Maze Layout')
    
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