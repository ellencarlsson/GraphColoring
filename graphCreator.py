import pygame
import time
import threading
import random
import os
import math
from concurrent.futures import ThreadPoolExecutor

pygame.init()
width, height = 1400, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Graph Coloring")
#runtime_info_font = pygame.font.SysFont(None, 25)


# Function to draw the graph
def draw_graph(color_assignment, nodes, edges):
    screen.fill((255, 255, 255))  # Clear screen with white background

    # Draw edges first
    for edge in edges:
        node1, node2 = edge
        pygame.draw.line(screen, (0, 0, 0),nodes[node1],nodes[node2], 2)  # Black edges

    # Draw nodes with the color assigned from the evolutionary algorithm
    for i, pos in enumerate(nodes):
        pygame.draw.circle(screen, (0, 0, 0), pos, 22)  # Draw outline in black
        pygame.draw.circle(screen, color_assignment[i], pos, 20)  # Draw node with its color
        # Display node index inside the node
        text = pygame.font.SysFont(None, 20).render(str(i), True, (0, 0, 0))
        screen.blit(text, (pos[0] - 5, pos[1] - 10))

    # Update the display after drawing everything
    pygame.display.update()

# Function to generate a random color
def generate_random_color():
    """Generates a random RGB color."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))