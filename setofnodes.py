
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

# Generate 100 node positions in a grid-like structure
nodes_XL = [(x * 100 + 50, y * 100 + 50) for y in range(10) for x in range(10)]

# Generate edges connecting adjacent nodes in a grid
edges_XL = []

# Connect horizontal edges (node n to n+1)
for y in range(10):
    for x in range(9):  # 9 because we're connecting adjacent horizontally, no edge for the last node in a row
        node1 = y * 10 + x
        node2 = y * 10 + x + 1
        edges_XL.append((node1, node2))

# Connect vertical edges (node n to n+10)
for x in range(10):
    for y in range(9):  # 9 because we're connecting vertically, no edge for the last row
        node1 = y * 10 + x
        node2 = (y + 1) * 10 + x
        edges_XL.append((node1, node2))

# Optionally add some random diagonal edges for variety
for _ in range(20):  # Add 20 random diagonal connections
    node1 = random.randint(0, 99)
    node2 = random.randint(0, 99)
    if node1 != node2 and (node1, node2) not in edges_XL and (node2, node1) not in edges_XL:
        edges_XL.append((node1, node2))

# Print nodes and edges to visualize or use in a graph algorithm
print("Nodes:", nodes_large)
print("Edges:", edges_XL)


# Define all the graph examples (nodes and edges)
nodes = [nodes_pentagon, nodes_star, nodes_complete, nodes_grid, nodes_bipartite, nodes_large]
edges = [edges_pentagon, edges_star, edges_complete, edges_grid, edges_bipartite, edges_XL]
