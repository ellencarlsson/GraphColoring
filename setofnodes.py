
nodes_pentagon = [(300, 100), (450, 200), (400, 350), (200, 350), (150, 200)]
edges_pentagon = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]

nodes_star = [(400, 300), (250, 150), (550, 150), (600, 300), (550, 450), (250, 450)]
edges_star = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]

nodes_complete = [(200, 100), (400, 100), (300, 250), (200, 400), (400, 400), (500, 250)]
edges_complete = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                   (1, 2), (1, 3), (1, 4), (1, 5),
                   (2, 3), (2, 4), (2, 5),
                   (3, 4), (3, 5),
                   (4, 5)]

nodes_grid = [(200, 100), (300, 100), (400, 100),
               (200, 200), (300, 200), (400, 200),
               (200, 300), (300, 300), (400, 300)]

edges_grid = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5),
               (3, 4), (4, 5), (3, 6), (4, 7), (5, 8),
               (6, 7), (7, 8)]

nodes_bipartite = [(200, 100), (300, 100), (400, 100), (500, 100),
                    (200, 300), (300, 300), (400, 300), (500, 300)]

edges_bipartite = [(0, 4), (0, 5), (0, 6), (0, 7),
                    (1, 4), (1, 5), (1, 6), (1, 7),
                    (2, 4), (2, 5), (2, 6), (2, 7),
                    (3, 4), (3, 5), (3, 6), (3, 7)]

nodes_large = [(100, 100), (200, 100), (300, 100), (400, 100), (500, 100),
                (100, 300), (200, 300), (300, 300), (400, 300), (500, 300)]

edges_large = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
                (3, 4), (3, 5), (4, 6), (5, 7), (5, 8),
                (6, 7), (7, 9), (8, 9)]

import random

# Generate 200 node positions in a grid-like structure
nodes_XL = [(x * 60 + 50, y * 60 + 50) for y in range(10) for x in range(20)]

# Initialize the list of edges connecting adjacent nodes in the grid
edges_XL = []

# Connect horizontal edges (node n to n+1 in the same row)
for y in range(10):  # Loop through each row
    for x in range(19):  # Only up to the second-to-last node in each row (x = 0 to 18)
        node1 = y * 20 + x  # The current node
        node2 = node1 + 1  # The node to the right
        edges_XL.append((node1, node2))

# Connect vertical edges (node n to n+20, meaning the node below)
for y in range(9):  # Only up to the second-to-last row (y = 0 to 8)
    for x in range(20):  # Loop through each column
        node1 = y * 20 + x  # The current node
        node2 = node1 + 20  # The node below
        edges_XL.append((node1, node2))

# Optionally, add some random diagonal edges for variety
for _ in range(4):  # Add 10 random diagonal connections
    node1 = random.randint(0, 199)
    node2 = random.randint(0, 199)
    if node1 != node2 and (node1, node2) not in edges_XL and (node2, node1) not in edges_XL:
        edges_XL.append((node1, node2))


# Define all the graph examples (nodes and edges)
nodes = [nodes_pentagon, nodes_star, nodes_complete, nodes_grid, nodes_bipartite, nodes_large]
edges = [edges_pentagon, edges_star, edges_complete, edges_grid, edges_bipartite, edges_XL]
