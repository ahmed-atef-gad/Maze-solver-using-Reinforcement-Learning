import pygame
import numpy as np
from typing import Tuple

class MazeRenderer:
    def __init__(self, maze, cell_size: int = 40):
        self.maze = maze
        self.cell_size = cell_size
        self.screen_width = maze.width * cell_size
        self.screen_height = maze.height * cell_size
        self.colors = {
            'empty': (255, 255, 255),
            'wall': (0, 0, 0),
            'start': (0, 255, 0),
            'goal': (255, 0, 0),
            'agent': (0, 0, 255),
            'path': (100, 100, 255),
            'moving_obstacle': (255, 165, 0)  # Orange
        }
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Maze Solver with Moving Obstacles")
        
    def draw_maze(self):
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                if (row, col) == self.maze.start:
                    color = self.colors['start']
                elif (row, col) == self.maze.goal:
                    color = self.colors['goal']
                elif self.maze.grid[row, col] == 1:
                    color = self.colors['wall']
                else:
                    color = self.colors['empty']
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
    
    def draw_moving_obstacles(self):
        for (row, col, _, _, _) in self.maze.moving_obstacles:  # Unpack 5 elements
            center = (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(
                self.screen, 
                self.colors['moving_obstacle'], 
                center, 
                self.cell_size // 3
            )
    
    def draw_agent(self, position: Tuple[int, int]):
        row, col = position
        center = (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.colors['agent'], center, self.cell_size // 3)
    
    def update(self, agent_pos=None):
        self.screen.fill((255, 255, 255))
        self.draw_maze()
        self.draw_moving_obstacles()
        if agent_pos:
            self.draw_agent(agent_pos)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()