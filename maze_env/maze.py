import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, List

class Maze(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width: int = 10, height: int = 10, num_moving_obstacles: int = 3, use_seed: int = None):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0 = empty, 1 = wall
        self.start = (0, 0)
        self.goal = (height-1, width-1)
        self.moving_obstacles: List[Tuple[int, int, int, int, int]] = []
        if use_seed is not None:
            np.random.seed(use_seed)
        self._create_maze()
        self._add_moving_obstacles(num_moving_obstacles)

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(
            low=0, high=max(self.height-1, self.width-1),
            shape=(2,), dtype=np.int32
        )
        self.agent_pos = self.start

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._create_maze()
        self._add_moving_obstacles(len(self.moving_obstacles))
        self.agent_pos = self.start
        return np.array(self.agent_pos, dtype=np.int32)

    def step(self, action):
        row, col = self.agent_pos
        if action == 0:   # up
            next_state = (row-1, col)
        elif action == 1:  # down
            next_state = (row+1, col)
        elif action == 2:  # left
            next_state = (row, col-1)
        elif action == 3:  # right
            next_state = (row, col+1)
        else:
            next_state = (row, col)
    
        # Get current obstacle positions before moving
        current_obstacle_positions = [(obs_row, obs_col) for (obs_row, obs_col, _, _, _) in self.moving_obstacles]
        
        # Predict the next positions of moving obstacles
        predicted_obstacle_positions = [
            (obs_row + dr, obs_col + dc) for (obs_row, obs_col, dr, dc, _) in self.moving_obstacles
        ]
    
        # Check for collision with current and predicted positions of moving obstacles
        collision = next_state in predicted_obstacle_positions or next_state in current_obstacle_positions
        
        # NEW: Check for path crossing - if agent and obstacle would swap positions
        path_crossing = False
        for i, (obs_row, obs_col, dr, dc, _) in enumerate(self.moving_obstacles):
            if (obs_row, obs_col) == next_state and (obs_row + dr, obs_col + dc) == self.agent_pos:
                path_crossing = True
                break
    
        if collision or path_crossing:
            reward = -10.0
            next_state = self.agent_pos  # Stay in current position
            done = False
        else:
            # Safe to move - get reward and check if goal reached
            reward = self.get_reward(*next_state)
            done = (next_state == self.goal)

        self.agent_pos = next_state
        obs = np.array(self.agent_pos, dtype=np.int32)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human'):
        # Optional: You can call your MazeRenderer here if you want to visualize
        print(f"Agent position: {self.agent_pos}")

    def close(self):
        pass

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
        occupied_positions = set()
        
        # Add agent position to occupied positions to prevent obstacles from moving there
        occupied_positions.add(self.agent_pos)
        
        for obstacle in self.moving_obstacles:
            row, col, dr, dc, steps_to_change = obstacle
            original_pos = (row, col)
            
            # Try current direction first
            new_row, new_col = row + dr, col + dc
            
            # CRITICAL FIX: Explicitly check if the new position would be the agent's position
            if (new_row, new_col) == self.agent_pos:
                valid_move = False
            else:
                valid_move = (
                    0 <= new_row < self.height and
                    0 <= new_col < self.width and
                    self.grid[new_row, new_col] == 0 and
                    (new_row, new_col) not in occupied_positions and
                    (new_row, new_col) != self.start and
                    (new_row, new_col) != self.goal
                )
            
            # If current direction invalid, try random directions
            if not valid_move:
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                np.random.shuffle(directions)
                for new_dr, new_dc in directions:
                    potential_row = row + new_dr
                    potential_col = col + new_dc
                    
                    # CRITICAL FIX: Explicitly check if the potential position would be the agent's position
                    if (potential_row, potential_col) == self.agent_pos:
                        continue
                        
                    if (0 <= potential_row < self.height and
                        0 <= potential_col < self.width and
                        self.grid[potential_row, potential_col] == 0 and
                        (potential_row, potential_col) not in occupied_positions and
                        (potential_row, potential_col) != self.start and
                        (potential_row, potential_col) != self.goal):
                        new_row, new_col = potential_row, potential_col
                        dr, dc = new_dr, new_dc
                        valid_move = True
                        break
            
            if valid_move:
                # Update steps_to_change counter
                steps_to_change -= 1
                if steps_to_change <= 0:
                    dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][np.random.randint(4)]
                    steps_to_change = np.random.randint(3, 7)
                occupied_positions.add((new_row, new_col))
                new_obstacles.append((new_row, new_col, dr, dc, steps_to_change))
            else:
                # Keep original position if no valid moves
                occupied_positions.add(original_pos)
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
            return 100.0
        
        # Penalty for being near moving obstacles
        obstacle_penalty = 0.0
        for (obs_row, obs_col, _, _, _) in self.moving_obstacles:
            distance = abs(row - obs_row) + abs(col - obs_col)
            if distance <= 1:  # Immediate vicinity
                obstacle_penalty -= 5.0
        
        current_distance = abs(row - self.goal[0]) + abs(col - self.goal[1])
        return -0.1 * current_distance + obstacle_penalty