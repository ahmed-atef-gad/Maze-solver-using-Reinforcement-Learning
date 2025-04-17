import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
    
    plt.tight_layout()
    plt.savefig('results/q_table_visualization.png')
    plt.show()

if __name__ == "__main__":
    visualize_q_table()