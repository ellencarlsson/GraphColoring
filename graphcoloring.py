import pygame
import sys
import time
import psutil
import threading
import random
import setofnodes
import csv
import os

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Graph Coloring")

# Variables
nodes = setofnodes.nodes_star # Store node positions (e.g., [(x1, y1), (x2, y2), ...])
edges = setofnodes.edges_star  # Store edges as tuples of node indices (e.g., [(0, 1), (1, 2)])
adj_matrix = []  # Adjacency matrix will be built dynamically
selected_node = None  # Used to track when a node is selected to create edges
stop_flag = False  # Flag to control runtime updates
runtime_info_font = pygame.font.SysFont(None, 25)

# Performance Monitoring Variables
max_cpu_usage = 0.0
max_mem_usage = 0.0
monitor_stop_flag = False
monitor_lock = threading.Lock()

# CSV File Path
CSV_FILE = "performance_data.csv"
AVERAGE_FILE = "performance_averages.txt"

def init_adj_matrix():
    n = len(nodes)
    global adj_matrix
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for edge in edges:
        node1, node2 = edge
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1  # Assuming an undirected graph

# Function to draw the graph nodes and edges
def draw_graph(color_assignment):
    screen.fill((255, 255, 255))  # Clear screen

    # Draw edges
    for edge in edges:
        node1, node2 = edge
        pygame.draw.line(screen, (0, 0, 0), nodes[node1], nodes[node2], 3)

    # Draw nodes
    for i, pos in enumerate(nodes):
        pygame.draw.circle(screen, color_assignment[i], pos, 30)
        text = pygame.font.SysFont(None, 25).render(str(i), True, (0, 0, 0))
        screen.blit(text, (pos[0] - 10, pos[1] - 10))

    pygame.display.update()

# Utility function to find if a position is within a node's circle
def find_node(pos):
    for i, node_pos in enumerate(nodes):
        if (node_pos[0] - pos[0]) ** 2 + (node_pos[1] - pos[1]) ** 2 <= 30 ** 2:
            return i
    return None

# Add node at the clicked position
def add_node(pos):
    nodes.append(pos)
    adj_matrix.append([0] * len(nodes))  # Add a new row
    for row in adj_matrix:
        row.append(0)  # Add a new column to each row

# Add edge between two selected nodes
def add_edge(node1, node2):
    if node1 != node2 and adj_matrix[node1][node2] == 0:
        adj_matrix[node1][node2] = adj_matrix[node2][node1] = 1
        edges.append((node1, node2))

# Graph coloring functions
def isSafe(v, color_assignment, c):
    for i in range(len(nodes)):
        if adj_matrix[v][i] == 1 and color_assignment[i] == c:
            return False
    return True

def graphColoringUtil(color_assignment, v, colors):
    if v == len(nodes):
        return True

    for c in range(len(colors)):
        if isSafe(v, color_assignment, colors[c]):
            color_assignment[v] = colors[c]
            draw_graph(color_assignment)
            if graphColoringUtil(color_assignment, v + 1, colors):
                return True
            color_assignment[v] = (255, 255, 255)  # Reset the color
            draw_graph(color_assignment)

    return False

def graphColoring(m):
    color_assignment = [(255, 255, 255)] * len(nodes)
    draw_graph(color_assignment)

    start_time_coloring = time.time()

    # Generate random colors based on the minimum required (m)
    colors = [generate_random_color() for _ in range(m)]

    if not graphColoringUtil(color_assignment, 0, colors):
        print("Solution does not exist")

    elapsed_time = time.time() - start_time_coloring
    return elapsed_time

def generate_random_color():
    """Generates a random RGB color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to calculate the minimum number of colors (chromatic number)
def calculate_min_colors():
    global min_colors
    color_assignment = [-1] * len(nodes)

    # Assign the first color to the first node
    color_assignment[0] = 0

    # A temporary array to store the available colors, initialized to false (not available)
    available = [False] * len(nodes)

    # Assign colors to remaining nodes
    for u in range(1, len(nodes)):
        # Mark colors of adjacent nodes as unavailable
        for i in range(len(nodes)):
            if adj_matrix[u][i] == 1 and color_assignment[i] != -1:
                available[color_assignment[i]] = True

        # Find the first available color
        cr = 0
        while cr < len(nodes) and available[cr]:
            cr += 1

        color_assignment[u] = cr  # Assign the found color

        # Reset the available array for the next iteration
        available = [False] * len(nodes)

    # The number of colors used is the chromatic number
    min_colors = max(color_assignment) + 1
    return max(color_assignment) + 1

def display_runtime_and_usage(start_time):
    global stop_flag
    while not stop_flag:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        runtime = time.time() - start_time

        runtime_text = runtime_info_font.render(f"Runtime: {runtime:.3f} s", True, (0, 0, 0))
        cpu_text = runtime_info_font.render(f"CPU Usage: {cpu_usage}%", True, (0, 0, 0))
        memory_text = runtime_info_font.render(f"Memory Usage: {memory_info.percent}%", True, (0, 0, 0))
        color_text = runtime_info_font.render(f"Colors Used: {min_colors}", True, (0,0,0))

        screen.fill((255, 255, 255), (width - 220, 10, 210, 90))  # Clear the area before writing new info
        screen.blit(runtime_text, (width - 200, 20))
        screen.blit(cpu_text, (width - 200, 50))
        screen.blit(memory_text, (width - 200, 80))
        screen.blit(color_text, (width - 200, 110))

        pygame.display.update()

# Performance Monitoring Function
def monitor_performance():
    global max_cpu_usage, max_mem_usage, monitor_stop_flag
    process = psutil.Process()
    while not monitor_stop_flag:
        cpu = process.cpu_percent(interval=0.1)
        mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        with monitor_lock:
            if cpu > max_cpu_usage:
                max_cpu_usage = cpu
            if mem > max_mem_usage:
                max_mem_usage = mem

# Function to calculate averages from the CSV file
def calculate_averages(csv_file):
    runtime_sum = 0.0
    cpu_sum = 0.0
    mem_sum = 0.0
    colors_sum = 0
    count = 0

    with open(csv_file, "r", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runtime_sum += float(row["Runtime_s"])
            cpu_sum += float(row["CPU_Usage_percent"])
            mem_sum += float(row["Memory_Usage_MB"])
            colors_sum += int(row["Colors_Used"])
            count += 1

    if count == 0:
        return None  # Avoid division by zero

    avg_runtime = runtime_sum / count
    avg_cpu = cpu_sum / count
    avg_mem = mem_sum / count
    avg_colors = colors_sum / count

    return avg_runtime, avg_cpu, avg_mem, avg_colors

# Main loop for the pygame window
def main():
    global selected_node, start_time, stop_flag
    global max_cpu_usage, max_mem_usage, monitor_stop_flag

    runtime = 0
    font = pygame.font.SysFont(None, 25)

    init_adj_matrix()

    graph_colored = False  # Flag to track if the graph has been colored

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                monitor_stop_flag = True  # Ensure the monitor thread stops
                pygame.quit()
                sys.exit()

            # Handle mouse click to add nodes and edges
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked_node = find_node(pos)

                if clicked_node is None:
                    # If no node is clicked, add a new node
                    add_node(pos)
                else:
                    if selected_node is None:
                        # First node selection for edge creation
                        selected_node = clicked_node
                    else:
                        # Second node selection, create edge
                        add_edge(selected_node, clicked_node)
                        selected_node = None  # Reset selected node

                # Only redraw the whole graph with white if it has not been colored yet
                if not graph_colored:
                    draw_graph([(255, 255, 255)] * len(nodes))

            # Press space to start coloring
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and len(nodes) > 0:
                    # Reset performance metrics
                    with monitor_lock:
                        max_cpu_usage = 0.0
                        max_mem_usage = 0.0
                    monitor_stop_flag = False

                    # Start the performance monitoring thread
                    monitor_thread = threading.Thread(target=monitor_performance, daemon=True)
                    monitor_thread.start()

                    # Start the runtime timer
                    start_time_coloring = time.time()

                    # Calculate minimum number of colors
                    min_colors = calculate_min_colors()

                    # Start graph coloring
                    elapsed_time = graphColoring(min_colors)

                    # Set flags to stop monitoring and display
                    graph_colored = True
                    stop_flag = True  # Stop the display_runtime_and_usage thread if it's running
                    monitor_stop_flag = True  # Stop the performance monitoring thread

                    # Wait for the monitor thread to finish
                    monitor_thread.join()

                    # Prepare timestamp
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                    # Log the performance data to CSV
                    file_exists = os.path.isfile(CSV_FILE)
                    with open(CSV_FILE, "a", newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            # Write header if the file does not exist
                            writer.writerow(["Timestamp", "Runtime_s", "CPU_Usage_percent", "Memory_Usage_MB", "Colors_Used"])
                        writer.writerow([timestamp, f"{elapsed_time:.3f}", f"{max_cpu_usage}", f"{max_mem_usage:.2f}", min_colors])

                    print(f"Performance data logged at {timestamp}")

                    # Calculate averages from the CSV file
                    averages = calculate_averages(CSV_FILE)
                    if averages:
                        avg_runtime, avg_cpu, avg_mem, avg_colors = averages

                        # Append the averages to a text file
                        avg_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        with open(AVERAGE_FILE, "a") as f:
                            f.write(f"{avg_timestamp}, Average Runtime: {avg_runtime:.3f} s, "
                                    f"Average CPU Usage: {avg_cpu:.2f}%, "
                                    f"Average Memory Usage: {avg_mem:.2f} MB, "
                                    f"Average Colors Used: {avg_colors:.2f}\n")

                        print(f"Averages logged at {avg_timestamp}")
                    else:
                        print("No data available to calculate averages.")

        # Only redraw the graph with white if not yet colored
        if not graph_colored:
            draw_graph([(255, 255, 255)] * len(nodes))

if __name__ == "__main__":
    main()
