import numpy as np
import pygame
from typing import Tuple, Dict, List

class Maze:
    def __init__(self, width: int = 10, height: int = 10, num_moving_obstacles: int = 3, use_seed: int = None):
        """
        Initialize a maze environment.
        
        Args:
            width: Width of the maze
            height: Height of the maze
            num_moving_obstacles: Number of moving obstacles to add
            use_seed: If provided, uses this seed for random number generation to create reproducible mazes.
                      Set to None for truly random mazes.
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0 = empty, 1 = wall
        self.start = (0, 0)
        self.goal = (height-1, width-1)
        self.moving_obstacles: List[Tuple[int, int, int, int, int]] = []  # (row, col, dr, dc, steps_to_change)
        if use_seed is not None:
            np.random.seed(use_seed)
        self._create_maze()

        self._add_moving_obstacles(num_moving_obstacles)
        
    def _create_maze(self):
        """Create a random maze with guaranteed path to goal"""
        # Clear all walls first - fill with walls
        self.grid[:, :] = 1
        
        # Create a path using randomized DFS
        def carve_path(row, col):
            # Mark current cell as passage
            self.grid[row, col] = 0
            
            # Define possible directions (up, right, down, left)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            # Shuffle directions for randomness
            np.random.shuffle(directions)
            
            # Try each direction
            for dr, dc in directions:
                new_row, new_col = row + 2*dr, col + 2*dc
                # Check if the new position is within bounds and not visited
                if (0 <= new_row < self.height and 0 <= new_col < self.width and 
                    self.grid[new_row, new_col] == 1):
                    # Carve passage by making the wall and the cell beyond it passages
                    self.grid[row + dr, col + dc] = 0
                    carve_path(new_row, new_col)
        
        # Start from a random position
        start_row, start_col = np.random.randint(1, self.height-1), np.random.randint(1, self.width-1)
        carve_path(start_row, start_col)
        
        # Ensure start and goal are open
        self.grid[self.start[0], self.start[1]] = 0
        self.grid[self.goal[0], self.goal[1]] = 0
        
        # Ensure there's a path from start to goal using A* pathfinding
        if not self._has_path_to_goal():
            self._ensure_path_to_goal()
        
        # Add some random openings to make the maze less dense (more paths)
        for _ in range(int(self.width * self.height * 0.2)):  # Open about 20% of walls
            row, col = np.random.randint(1, self.height-1), np.random.randint(1, self.width-1)
            self.grid[row, col] = 0
    
    def _has_path_to_goal(self):
        """Check if there's a path from start to goal using A* algorithm"""
        # A* implementation
        from heapq import heappush, heappop
        
        # Heuristic: Manhattan distance
        def heuristic(pos):
            return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        
        # Initialize
        open_set = []
        heappush(open_set, (heuristic(self.start), 0, self.start))  # (f, g, position)
        came_from = {}
        g_score = {self.start: 0}
        
        while open_set:
            _, _, current = heappop(open_set)
            
            if current == self.goal:
                return True
            
            # Check neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check if valid position
                if not (0 <= neighbor[0] < self.height and 0 <= neighbor[1] < self.width):
                    continue
                if self.grid[neighbor[0], neighbor[1]] == 1:  # Wall
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heappush(open_set, (f_score, tentative_g, neighbor))
        
        return False
    
    def _ensure_path_to_goal(self):
        """Ensure there's a path from start to goal by carving a direct path"""
        row, col = self.start
        while (row, col) != self.goal:
            # Move towards goal
            if row < self.goal[0]:
                row += 1
            elif row > self.goal[0]:
                row -= 1
            elif col < self.goal[1]:
                col += 1
            elif col > self.goal[1]:
                col -= 1
            
            # Carve path
            self.grid[row, col] = 0
        
        # Add some random obstacles that don't block the path
        # First identify the path we just created
        path = []
        row, col = self.start
        while (row, col) != self.goal:
            path.append((row, col))
            if row < self.goal[0]:
                row += 1
            elif row > self.goal[0]:
                row -= 1
            elif col < self.goal[1]:
                col += 1
            elif col > self.goal[1]:
                col -= 1
        path.append(self.goal)
        
        # Add some random obstacles that don't block the path
        for _ in range(15):  # Number of obstacles
            attempts = 0
            while attempts < 100:
                row, col = np.random.randint(1, self.height-1), np.random.randint(1, self.width-1)
                if (row, col) not in path and (row, col) != self.start:
                    self.grid[row, col] = 1
                    break
                attempts += 1

    def _add_moving_obstacles(self, num_obstacles: int):
        """Add obstacles that move in set directions"""
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 100:
                row = np.random.randint(1, self.height-1)
                col = np.random.randint(1, self.width-1)
                # Ensure not placed on static wall or start/goal
                if self.grid[row, col] == 0 and (row, col) != self.start and (row, col) != self.goal:
                    dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                    steps_to_change = np.random.randint(3, 7)  # Random steps before changing direction
                    self.moving_obstacles.append((row, col, dr, dc, steps_to_change))
                    break
                attempts += 1

    def update_moving_obstacles(self):
        """Update positions of moving obstacles"""
        new_obstacles = []
        for row, col, dr, dc, steps_to_change in self.moving_obstacles:
            # Try to move in the current direction
            new_row, new_col = row + dr, col + dc

            # Check if the new position is valid
            if (0 <= new_row < self.height and 0 <= new_col < self.width and
                self.grid[new_row, new_col] == 0 and  # Ensure it's not a wall
                (new_row, new_col) != self.start and
                (new_row, new_col) != self.goal):
                # Valid move
                steps_to_change -= 1
                if steps_to_change <= 0:
                    # Randomly change direction after a certain number of steps
                    dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                    steps_to_change = np.random.randint(3, 7)
                new_obstacles.append((new_row, new_col, dr, dc, steps_to_change))
            else:
                # Current direction is blocked, choose a new random valid direction
                valid_move_found = False
                for _ in range(4):  # Try all 4 possible directions
                    dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < self.height and 0 <= new_col < self.width and
                        self.grid[new_row, new_col] == 0 and
                        (new_row, new_col) != self.start and
                        (new_row, new_col) != self.goal):
                        # Found a valid move
                        steps_to_change = np.random.randint(3, 7)
                        new_obstacles.append((new_row, new_col, dr, dc, steps_to_change))
                        valid_move_found = True
                        break
                if not valid_move_found:
                    # If no valid move is found, keep the obstacle in its current position
                    new_obstacles.append((row, col, dr, dc, steps_to_change))

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
            return 100.0  # Higher reward for reaching the goal
        
        # Calculate Manhattan distance to the goal
        current_distance = abs(row - self.goal[0]) + abs(col - self.goal[1])
        max_distance = self.width + self.height
        
        # Stronger signal for getting closer to the goal
        return -0.1 * current_distance  # Reward gets less negative as we get closer