import pygame
import sys
import time
import psutil
import threading
import random
import setofnodes
import csv
import os
import math

# Initialize Pygame
pygame.init()
width, height = 1400, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Graph Coloring")

# pentagon
# bipartite
# grid
# large
# star
# complete
# XL

figure = "XL"

# Variables
nodes = getattr(setofnodes, f"nodes_{figure}")  # Store node positions
edges = getattr(setofnodes, f"edges_{figure}")  # Store edges as tuples of node indices
adj_matrix = []  # Adjacency matrix will be built dynamically
adj_list = []     # Adjacency list for faster fitness computation
selected_node = None  # Used to track when a node is selected to create edges
stop_flag = False  # Flag to control runtime updates
runtime_info_font = pygame.font.SysFont(None, 25)

# Performance Monitoring Variables
max_cpu_usage = 0.0
max_mem_usage = 0.0
monitor_stop_flag = False
monitor_lock = threading.Lock()

# CSV File Path
CSV_FILE = f"{figure}_performance_data.csv"
AVERAGE_FILE = "performance_averages.txt"

mainpath = "DIMACS_graphs/"
dimacs_path = mainpath + "small_2.txt"

# Define fieldnames for CSV
CSV_FIELDNAMES = ["Timestamp", "Runtime_s", "CPU_Usage_percent", "Memory_Usage_MB", "Colors_Used"]

def init_adj_matrix_and_list():
    """Load the graph from the DIMACS file and initialize adjacency matrix and adjacency list."""
    global nodes, edges, adj_matrix, adj_list

    nodes = []
    edges = []

    with open(dimacs_path, 'r') as f:
        for line in f:
            # Skip comment lines
            if line.startswith('c'):
                continue
            # Parse the problem line (p edge <num_vertices> <num_edges>)
            elif line.startswith('p edge'):
                _, _, num_vertices, num_edges = line.strip().split()
                num_vertices = int(num_vertices)  # Get the number of vertices

                # Use a circular layout to evenly space nodes around the screen
                radius = min(width, height) // 2 - 50  # Set a radius smaller than half the screen size
                center_x, center_y = width // 2, height // 2  # Center of the screen
                
                for i in range(num_vertices):
                    angle = 2 * math.pi * i / num_vertices  # Angle for each node
                    x = int(center_x + radius * math.cos(angle))  # Polar to Cartesian
                    y = int(center_y + radius * math.sin(angle))
                    nodes.append((x, y))

            # Parse edge lines (e <node1> <node2>)
            elif line.startswith('e'):
                _, node1, node2 = line.strip().split()
                node1 = int(node1) - 1  # Convert to 0-based index
                node2 = int(node2) - 1  # Convert to 0-based index
                edges.append((node1, node2))  # Add the edge to the edges list

    # Initialize adjacency matrix and adjacency list
    n = len(nodes)
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    
    for edge in edges:
        node1, node2 = edge
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1  # Assuming an undirected graph
        adj_list[node1].append(node2)
        adj_list[node2].append(node1)

# Function to draw the graph nodes and edges
def draw_graph(color_assignment):
    screen.fill((255, 255, 255))  # Clear screen

    # Draw edges
    for edge in edges:
        node1, node2 = edge
        pygame.draw.line(screen, (0, 0, 0), nodes[node1], nodes[node2], 2)

    # Draw nodes with outlines
    for i, pos in enumerate(nodes):
        pygame.draw.circle(screen, (0, 0, 0), pos, 22)  # Draw outline
        pygame.draw.circle(screen, color_assignment[i], pos, 20)
        text = pygame.font.SysFont(None, 20).render(str(i), True, (0, 0, 0))
        screen.blit(text, (pos[0] - 5, pos[1] - 10))

    pygame.display.update()

# Utility function to find if a position is within a node's circle
def find_node(pos):
    for i, node_pos in enumerate(nodes):
        if (node_pos[0] - pos[0]) ** 2 + (node_pos[1] - pos[1]) ** 2 <= 20 ** 2:
            return i
    return None

# Add node at the clicked position
def add_node(pos):
    nodes.append(pos)
    # Update adjacency structures
    for row in adj_matrix:
        row.append(0)
    adj_matrix.append([0] * len(nodes))
    adj_list.append([])

# Add edge between two selected nodes
def add_edge(node1, node2):
    if node1 != node2 and adj_matrix[node1][node2] == 0:
        adj_matrix[node1][node2] = adj_matrix[node2][node1] = 1
        adj_list[node1].append(node2)
        adj_list[node2].append(node1)
        edges.append((node1, node2))

# Function to generate a random color
def generate_random_color():
    """Generates a random RGB color."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))  # Avoid very dark colors

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

# Verification Function
def verify_coloring(candidate):
    """Verify that no two adjacent nodes share the same color."""
    for node1, node2 in edges:
        if candidate[node1] == candidate[node2]:
            return False
    return True

# Evolutionary Graph Coloring Functions
def evolutionary_graph_coloring():
    population_size = 50  # Balanced population size
    max_generations = 1000  # Sufficient generations for convergence
    mutation_rate = 0.005  # Lower mutation rate for stability
    max_colors = len(nodes)  # Maximum possible colors

    # Start timing
    start_time_coloring = time.time()

    # Initialize variables to track the best solution across all color counts
    overall_best_candidate = None
    overall_best_colors = max_colors
    found_valid_coloring = False

    # Iteratively reduce the number of colors
    for k in range(1, max_colors + 1):
        print(f"Trying to color with {k} colors.")
        # Initialize population with k colors
        population = [
            [random.randint(0, k - 1) for _ in range(len(nodes))]
            for _ in range(population_size)
        ]

        best_candidate = None
        best_fitness = float('inf')

        for generation in range(max_generations):
            # Evaluate fitness of each candidate
            fitness_values = [fitness(candidate) for candidate in population]

            # Update best candidate
            min_fitness_in_population = min(fitness_values)
            if min_fitness_in_population < best_fitness:
                best_fitness = min_fitness_in_population
                best_candidate = population[fitness_values.index(min_fitness_in_population)]

                # If a valid coloring is found, verify and update
                if best_fitness == 0 and verify_coloring(best_candidate):
                    print(f"Valid coloring found with {k} colors at generation {generation}.")
                    overall_best_candidate = best_candidate
                    overall_best_colors = k
                    found_valid_coloring = True
                    break

            # Selection
            selected_parents = tournament_selection(population, fitness_values)

            # Create new population
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1 if i + 1 < population_size else 0]

                # Crossover
                child1, child2 = crossover(parent1, parent2)

                # Mutation
                child1 = mutate(child1, k, mutation_rate)
                child2 = mutate(child2, k, mutation_rate)

                new_population.extend([child1, child2])

            # Elitism: Preserve the best candidate
            new_population[0] = best_candidate.copy()

            population = new_population[:population_size]

            # Optional: Display progress every certain generations
            if generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # If a valid coloring was found for current k, stop searching for lower k
        if found_valid_coloring:
            break
        else:
            print(f"No valid coloring found with {k} colors.")

    elapsed_time = time.time() - start_time_coloring

    if overall_best_candidate and verify_coloring(overall_best_candidate):
        # Generate colors based on the number of colors used in the best candidate
        num_colors_used = overall_best_colors
        colors = [generate_random_color() for _ in range(num_colors_used)]
        color_map = {color_index: colors[i] for i, color_index in enumerate(range(num_colors_used))}
        color_assignment = [color_map[color_index] for color_index in overall_best_candidate]

        draw_graph(color_assignment)

        # Display runtime and usage
        runtime = elapsed_time
        with monitor_lock:
            cpu_usage = max_cpu_usage
            memory_info = max_mem_usage
        runtime_text = runtime_info_font.render(f"Runtime: {runtime:.3f} s", True, (0, 0, 0))
        cpu_text = runtime_info_font.render(f"CPU Usage: {cpu_usage}%", True, (0, 0, 0))
        memory_text = runtime_info_font.render(f"Memory Usage: {memory_info:.2f} MB", True, (0, 0, 0))
        color_text = runtime_info_font.render(f"Colors Used: {num_colors_used}", True, (0, 0, 0))
        # Clear the area before writing new info
        pygame.draw.rect(screen, (255, 255, 255), (width - 220, 10, 210, 130))
        screen.blit(runtime_text, (width - 200, 20))
        screen.blit(cpu_text, (width - 200, 50))
        screen.blit(memory_text, (width - 200, 80))
        screen.blit(color_text, (width - 200, 110))
        pygame.display.update()

        print(f"Evolutionary coloring completed in {elapsed_time:.3f} seconds using {num_colors_used} colors.")
        return elapsed_time, num_colors_used
    else:
        print("No valid coloring found after verification.")
        return elapsed_time, 0

def fitness(candidate):
    """Fitness function counts the number of color conflicts by iterating over edges."""
    conflicts = 0
    for node1, node2 in edges:
        if candidate[node1] == candidate[node2]:
            conflicts += 1
    return conflicts

# Tournament Selection with reduced tournament size
def tournament_selection(population, fitness_values, tournament_size=2):
    selected = []
    for _ in range(len(population)):
        # Randomly select tournament_size individuals
        participants = random.sample(range(len(population)), tournament_size)
        # Select the best among them
        best = participants[0]
        for p in participants[1:]:
            if fitness_values[p] < fitness_values[best]:
                best = p
        selected.append(population[best].copy())
    return selected

# Crossover Function
def crossover(parent1, parent2):
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation Function
def mutate(candidate, current_num_colors, mutation_rate):
    for i in range(len(candidate)):
        if random.random() < mutation_rate:
            candidate[i] = random.randint(0, current_num_colors - 1)
    return candidate

# Main loop for the pygame window
def main():
    global selected_node, stop_flag
    global max_cpu_usage, max_mem_usage, monitor_stop_flag

    runtime = 0
    font = pygame.font.SysFont(None, 25)

    init_adj_matrix_and_list()

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

                    # Start evolutionary graph coloring
                    elapsed_time, num_colors_used = evolutionary_graph_coloring()

                    # Set flags to stop monitoring and display
                    graph_colored = True
                    stop_flag = True  # Stop the display_runtime_and_usage thread if it's running
                    monitor_stop_flag = True  # Stop the performance monitoring thread

                    # Wait for the monitor thread to finish
                    monitor_thread.join()

                    if num_colors_used > 0:
                        # Prepare timestamp
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                        # Log the performance data to CSV using DictWriter
                        file_exists = os.path.isfile(CSV_FILE)
                        with open(CSV_FILE, "a", newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                            if not file_exists or os.stat(CSV_FILE).st_size == 0:
                                # Write header if the file does not exist or is empty
                                writer.writeheader()
                            writer.writerow({
                                "Timestamp": timestamp,
                                "Runtime_s": f"{elapsed_time:.3f}",
                                "CPU_Usage_percent": f"{max_cpu_usage}",
                                "Memory_Usage_MB": f"{max_mem_usage:.2f}",
                                "Colors_Used": num_colors_used
                            })

                        print(f"Performance data logged at {timestamp}")

                        # Calculate averages from the CSV file
                        averages = calculate_averages(CSV_FILE)
                        if averages:
                            avg_runtime, avg_cpu, avg_mem, avg_colors = averages

                            # Read the existing averages file, or start with an empty list
                            if os.path.isfile(AVERAGE_FILE):
                                with open(AVERAGE_FILE, "r") as f:
                                    lines = f.readlines()
                            else:
                                lines = []

                            figure_found = False
                            new_lines = []
                            run_count = 1  # Start with 1 if figure is not found

                            # Check if the figure's average data already exists in the file
                            for line in lines:
                                if f"Figure: {figure}" in line:
                                    # Extract the current run count from the line (if present)
                                    split_line = line.split(", ")
                                    if "Run Count" in line:
                                        count_part = split_line[-1]
                                        current_count = int(count_part.split(": ")[1])
                                        run_count = current_count + 1
                                    else:
                                        run_count = 1  # Handle the case where no count is present

                                    # Overwrite the line with new average data and increment run count
                                    new_line = (f"Figure: {figure}, "
                                                f"Average Runtime: {avg_runtime:.3f} s, "
                                                f"Average CPU Usage: {avg_cpu:.2f}%, "
                                                f"Average Memory Usage: {avg_mem:.2f} MB, "
                                                f"Average Colors Used: {avg_colors:.2f}, "
                                                f"Run Count: {run_count}\n")
                                    new_lines.append(new_line)
                                    figure_found = True
                                else:
                                    # Keep other lines unchanged
                                    new_lines.append(line)

                            # If the figure was not found, append the new average data and set run count to 1
                            if not figure_found:
                                new_line = (f"Figure: {figure}, "
                                            f"Average Runtime: {avg_runtime:.3f} s, "
                                            f"Average CPU Usage: {avg_cpu:.2f}%, "
                                            f"Average Memory Usage: {avg_mem:.2f} MB, "
                                            f"Average Colors Used: {avg_colors:.2f}, "
                                            f"Run Count: {run_count}\n")
                                new_lines.append(new_line)

                            # Write the updated data back to the average file
                            with open(AVERAGE_FILE, "w") as f:
                                f.writelines(new_lines)

                            avg_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                            print(f"Averages for Figure {figure} logged/updated at {avg_timestamp}")
                            print(f"Run Count for Figure {figure}: {run_count}")
                        else:
                            print("No data available to calculate averages.")
                    else:
                        print("No valid coloring was found. Please try again or adjust the graph.")

        # Only redraw the graph with white if not yet colored
        if not graph_colored:
            draw_graph([(255, 255, 255)] * len(nodes))

if __name__ == "__main__":
    main()
