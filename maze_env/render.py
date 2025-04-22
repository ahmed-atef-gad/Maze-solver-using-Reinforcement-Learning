import pygame
import numpy as np
from typing import Tuple
import math

class MazeRenderer:
    def __init__(self, maze, cell_size: int = 40):
        self.maze = maze
        self.cell_size = cell_size
        # Main maze area dimensions
        self.maze_width = maze.width * cell_size
        self.maze_height = maze.height * cell_size
        
        # Status panel dimensions
        self.panel_height = 100
        self.panel_padding = 20
        
        # Total window dimensions
        self.screen_width = self.maze_width
        self.screen_height = self.maze_height + self.panel_height
        
        self.colors = {
            'empty': (240, 240, 245),    # Light bluish white
            'wall': (48, 48, 75),        # Dark blue-gray
            'start': (46, 204, 113),     # Emerald green
            'goal': (231, 76, 60),       # Flat red
            'agent': (52, 152, 219),     # Bright blue
            'path': (155, 89, 182),      # Purple
            'moving_obstacle': (243, 156, 18),  # Orange
            'grid_lines': (189, 195, 199),     # Light gray
            'arrow': (231, 76, 60),            # Red arrows
            'panel': (44, 62, 80),            # Dark slate for panel
            'text': (236, 240, 241)           # Light gray for text
        }
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Dynamic Maze Solver")
        
        # Initialize fonts
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 20)
        
        # Initialize animation variables
        self.animation_tick = 0
        self.pulse_rate = 0.1
        
        # Initialize game stats
        self.steps = 0
        self.score = 0
        self.status = "In Progress"

    def draw_maze(self):
        # Draw background with gradient
        for y in range(self.screen_height):
            gradient = (
                max(20, min(255, self.colors['empty'][0] - y // 4)),
                max(20, min(255, self.colors['empty'][1] - y // 4)),
                max(20, min(255, self.colors['empty'][2] - y // 4))
            )
            pygame.draw.line(self.screen, gradient, (0, y), (self.screen_width, y))
        
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                rect = pygame.Rect(
                    col * self.cell_size,
                    row * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                
                if (row, col) == self.maze.start:
                    self.draw_special_cell(rect, self.colors['start'], "S")
                elif (row, col) == self.maze.goal:
                    self.draw_special_cell(rect, self.colors['goal'], "G")
                elif self.maze.grid[row, col] == 1:
                    self.draw_wall(rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['empty'], rect)
                    
                pygame.draw.rect(self.screen, self.colors['grid_lines'], rect, 1)
    
    def draw_wall(self, rect):
        """Draw wall with 3D effect"""
        dark_wall = tuple(max(0, c - 30) for c in self.colors['wall'])
        light_wall = tuple(min(255, c + 30) for c in self.colors['wall'])
        
        # Main wall
        pygame.draw.rect(self.screen, self.colors['wall'], rect)
        
        # Top highlight
        pygame.draw.line(self.screen, light_wall, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, light_wall, rect.topleft, rect.bottomleft, 2)
        
        # Bottom shadow
        pygame.draw.line(self.screen, dark_wall, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, dark_wall, rect.topright, rect.bottomright, 2)
    
    def draw_special_cell(self, rect, color, text):
        """Draw start/goal cells with pulsing effect"""
        pulse = abs(math.sin(self.animation_tick * self.pulse_rate))
        glow_color = tuple(int(c * (0.7 + 0.3 * pulse)) for c in color)
        
        # Draw glowing background
        pygame.draw.rect(self.screen, glow_color, rect)
        
        # Draw label
        if not hasattr(self, 'font'):
            self.font = pygame.font.SysFont('Arial', int(self.cell_size * 0.6), bold=True)
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def draw_moving_obstacles(self):
        for (row, col, dr, dc, _) in self.maze.moving_obstacles:
            center = (
                col * self.cell_size + self.cell_size // 2,
                row * self.cell_size + self.cell_size // 2
            )
            
            # Calculate arrow direction angle
            angle = math.atan2(dr, dc)
            
            # Draw obstacle with ripple effect
            pulse = abs(math.sin(self.animation_tick * self.pulse_rate))
            for i in range(3):
                radius = (self.cell_size // 3) * (1 + i * 0.2) * (0.8 + 0.2 * pulse)
                alpha = int(255 * (1 - i * 0.3) * pulse)
                color = (*self.colors['moving_obstacle'], alpha)
                pygame.draw.circle(self.screen, color, center, int(radius))
            
            # Draw direction arrow
            arrow_length = self.cell_size // 2
            end_pos = (
                center[0] + int(arrow_length * math.cos(angle)),
                center[1] + int(arrow_length * math.sin(angle))
            )
            
            # Draw arrow shaft
            pygame.draw.line(
                self.screen,
                self.colors['arrow'],
                center,
                end_pos,
                3
            )
            
            # Draw arrow head
            head_length = self.cell_size // 4
            angle_head = math.pi / 6  # 30 degrees
            
            head_point1 = (
                end_pos[0] - head_length * math.cos(angle - angle_head),
                end_pos[1] - head_length * math.sin(angle - angle_head)
            )
            head_point2 = (
                end_pos[0] - head_length * math.cos(angle + angle_head),
                end_pos[1] - head_length * math.sin(angle + angle_head)
            )
            
            pygame.draw.polygon(
                self.screen,
                self.colors['arrow'],
                [end_pos, head_point1, head_point2]
            )
    
    def draw_agent(self, position: Tuple[int, int]):
        row, col = position
        center = (
            col * self.cell_size + self.cell_size // 2,
            row * self.cell_size + self.cell_size // 2
        )
        
        # Draw agent with dynamic glow effect
        pulse = abs(math.sin(self.animation_tick * self.pulse_rate))
        base_radius = self.cell_size // 3
        
        # Outer glow
        for i in range(4):
            radius = base_radius * (1 + i * 0.2) * (0.8 + 0.2 * pulse)
            alpha = int(255 * (1 - i * 0.25))
            color = (*self.colors['agent'], alpha)
            pygame.draw.circle(self.screen, color, center, int(radius))
        
        # Core
        pygame.draw.circle(self.screen, self.colors['agent'], center, base_radius)
        
        # Highlight
        highlight_pos = (
            center[0] - base_radius // 3,
            center[1] - base_radius // 3
        )
        pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos, base_radius // 4)
    
    def draw_status_panel(self):
        """Draw the status panel below the maze"""
        panel_rect = pygame.Rect(0, self.maze_height, self.screen_width, self.panel_height)
        
        # Draw panel background with gradient
        for y in range(self.maze_height, self.screen_height):
            progress = (y - self.maze_height) / self.panel_height
            color = tuple(int(c * (1 - progress * 0.3)) for c in self.colors['panel'])
            pygame.draw.line(self.screen, color, (0, y), (self.screen_width, y))
        
        # Draw status information
        y_offset = self.maze_height + self.panel_padding
        
        # Draw steps
        steps_text = f"Steps: {self.steps}"
        text_surface = self.info_font.render(steps_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.panel_padding, y_offset))
        
        # Draw score
        score_text = f"Score: {self.score}"
        text_surface = self.info_font.render(score_text, True, self.colors['text'])
        self.screen.blit(text_surface, (self.screen_width // 3, y_offset))
        
        # Draw status with dynamic color
        status_color = (46, 204, 113) if self.status == "Success!" else \
                      (231, 76, 60) if self.status == "Failed!" else \
                      self.colors['text']
        
        status_text = self.title_font.render(self.status, True, status_color)
        status_rect = status_text.get_rect(
            center=(self.screen_width * 2 // 3 + self.panel_padding, 
                   self.maze_height + self.panel_height // 2)
        )
        
        # Add glow effect for success/failed status
        if self.status in ["Success!", "Failed!"]:
            pulse = abs(math.sin(self.animation_tick * self.pulse_rate))
            for i in range(3):
                glow_surface = self.title_font.render(self.status, True, (*status_color, 100 - i * 30))
                glow_rect = glow_surface.get_rect(center=(
                    status_rect.centerx + math.cos(pulse + i) * 2,
                    status_rect.centery + math.sin(pulse + i) * 2
                ))
                self.screen.blit(glow_surface, glow_rect)
        
        self.screen.blit(status_text, status_rect)

    def update(self, agent_pos=None, steps=0, score=0, status="In Progress"):
        """Update the display with current game state"""
        self.animation_tick += 1
        self.steps = steps
        self.score = score
        self.status = status
        
        self.screen.fill(self.colors['empty'])
        self.draw_maze()
        self.draw_moving_obstacles()
        if agent_pos:
            self.draw_agent(agent_pos)
        self.draw_status_panel()
        pygame.display.flip()
    
    def close(self):
        pygame.quit()