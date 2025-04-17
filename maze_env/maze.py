import numpy as np
import pygame
from typing import Tuple, Dict, List

class Maze:
    def __init__(self, width: int = 10, height: int = 10, num_moving_obstacles: int = 3):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0 = empty, 1 = wall
        self.start = (0, 0)
        self.goal = (height-1, width-1)
        self.moving_obstacles: List[Tuple[int, int, int, int]] = []  # (row, col, dr, dc)
        self._create_maze()
        self._add_moving_obstacles(num_moving_obstacles)
        
    def _create_maze(self):
        """Create a maze with guaranteed path to goal"""
        # Clear all walls first
        self.grid[:, :] = 0
        
        # Add borders (leave bottom-right open)
        self.grid[0, :] = 1                      # Top wall
        self.grid[:, 0] = 1                      # Left wall
        self.grid[:, -1] = 1                     # Right wall
        self.grid[-1, :-1] = 1                   # Bottom wall (except last cell)
        self.grid[0, 1] = 0
        self.grid[1, 0] = 0                      
        self.grid[-1, 8] = 0                  
        # Create a winding path to goal
        path = [
            (-1,8),(1,0),(0,1),(1, 1), (1, 2), (1, 3),             # Rightward path
            (2, 3), (3, 3), (4, 3),              # Downward
            (4, 4), (4, 5),                      # Rightward
            (3, 5), (2, 5),                      # Upward
            (2, 6), (2, 7),                      # Rightward
            (3, 7), (4, 7), (5, 7), (6, 7),     # Downward
            (6, 8), (7, 8), (8, 8), (8, 9)      # Right then down to goal
        ] 
        
        # Add some random obstacles that don't block the path
        for _ in range(15):  # Number of obstacles
            while True:
                row, col = np.random.randint(1, self.height-1), np.random.randint(1, self.width-1)
                if (row, col) not in path and (row, col) != self.start:
                    self.grid[row, col] = 1
                    break
        
    def _add_moving_obstacles(self, num_obstacles: int):
        """Add obstacles that move in set directions"""
        for _ in range(num_obstacles):
            while True:
                row = np.random.randint(1, self.height-1)
                col = np.random.randint(1, self.width-1)
                # Ensure not placed on static wall or start/goal
                if self.grid[row, col] == 0 and (row, col) != self.start and (row, col) != self.goal:
                    dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                    steps_to_change = np.random.randint(3, 7)  # Random steps before changing direction
                    self.moving_obstacles.append((row, col, dr, dc, steps_to_change))
                    break

    def update_moving_obstacles(self):
        """Update positions of moving obstacles"""
        new_obstacles = []
        for row, col, dr, dc, steps_to_change in self.moving_obstacles:
            # Try to move in current direction
            new_row, new_col = row + dr, col + dc

            # Check if new position is out of bounds or hits a wall
            if (new_row < 0 or new_row >= self.height or
                new_col < 0 or new_col >= self.width or
                self.grid[new_row, new_col] == 1):
                # Reverse direction if hitting wall or border
                dr, dc = -dr, -dc
                new_row, new_col = row + dr, col + dc

            # Randomly change direction after a certain number of steps
            steps_to_change -= 1
            if steps_to_change <= 0:
                dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                steps_to_change = np.random.randint(3, 7)

            # Ensure the new position is within bounds
            new_row = max(0, min(self.height - 1, new_row))
            new_col = max(0, min(self.width - 1, new_col))

            new_obstacles.append((new_row, new_col, dr, dc, steps_to_change))

        self.moving_obstacles = new_obstacles
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is valid (not wall and not moving obstacle)"""
        # Check static walls and borders
        if not (0 <= row < self.height and 0 <= col < self.width and self.grid[row, col] == 0):
            return False
        
        # Check moving obstacles
        for (obs_row, obs_col, _, _, _) in self.moving_obstacles:  # Unpack 5 elements
            if row == obs_row and col == obs_col:
                return False
                
        return True
    
    def get_reward(self, row: int, col: int) -> float:
        """Get reward for current position"""
        if (row, col) == self.goal:
            return 20.0  # High reward for reaching the goal

        # Calculate Manhattan distance to the goal
        current_distance = abs(row - self.goal[0]) + abs(col - self.goal[1])
        previous_distance = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])

        # Reward for moving closer to the goal
        if current_distance < previous_distance:
            return 1.0  # Positive reward for reducing distance

        return -1.0  # Stronger negative reward for each step