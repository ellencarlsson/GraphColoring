import pygame
import time
import threading
import random
import os
import math
from concurrent.futures import ThreadPoolExecutor

# Initialize Pygame
pygame.init()
width, height = 1400, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Interactive Graph Coloring")

adj_matrix = []  # Adjacency matrix will be built dynamically
adj_list = []     # Adjacency list for faster fitness computation
selected_node = None  # Used to track when a node is selected to create edges
stop_flag = False  # Flag to control runtime updates
runtime_info_font = pygame.font.SysFont(None, 25)

monitor_stop_flag = False
monitor_lock = threading.Lock()

mainpath = "DIMACS_graphs/"
dimacs = "large_5"
dimacs_path = mainpath + dimacs + ".txt"


# Function to generate a random color
def generate_random_color():
    """Generates a random RGB color."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

# Parallel Fitness Evaluation
def evaluate_population(population, fitness_values, changed_nodes_list):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_population.fitness, candidate, changed_nodes_list[i])
            for i, candidate in enumerate(population)
        ]
        for i, future in enumerate(futures):
            fitness_values[i] = future.result()

def draw_graph(color_assignment):
    screen.fill((255, 255, 255)) 

    for edge in edges:
        node1, node2 = edge
        pygame.draw.line(screen, (0, 0, 0), nodes[node1], nodes[node2], 2) 

    # Draw nodes with the color assigned from the evolutionary algorithm
    for i, pos in enumerate(nodes):
        pygame.draw.circle(screen, (0, 0, 0), pos, 22)  # Draw outline in black
        pygame.draw.circle(screen, color_assignment[i], pos, 20)  # Draw node with its color
        # Display node index inside the node
        text = pygame.font.SysFont(None, 20).render(str(i), True, (0, 0, 0))
        screen.blit(text, (pos[0] - 5, pos[1] - 10))

    # Update the display after drawing everything
    pygame.display.update()

# Evolutionary Graph Coloring with parallelization and incremental evaluation
def evolutionary_graph_coloring():
    population_size = 50
    max_generations = 1000
    mutation_rate = 0.01
    max_colors = len(nodes)

    # Start timing
    start_time_coloring = time.time()

    # Initialize the population
    population = [
        [random.randint(0, max_colors - 1) for _ in range(len(nodes))]
        for _ in range(population_size)
    ]

    best_candidate = None
    best_fitness = float('inf')
    found_valid_coloring = False
    fitness_evaluations = 0

    for generation in range(max_generations):
        # Create a list to track nodes that changed due to mutation
        changed_nodes_list = [list(range(len(nodes))) for _ in range(population_size)]
        fitness_values = [0] * population_size

        # Evaluate fitness in parallel
        evaluate_population(population, fitness_values, changed_nodes_list)
        fitness_evaluations += population_size  # Increment fitness evaluations counter

        # Find the best candidate
        min_fitness_in_population = min(fitness_values)
        if min_fitness_in_population < best_fitness:
            best_fitness = min_fitness_in_population
            best_candidate = population[fitness_values.index(min_fitness_in_population)]
            if best_fitness == 0:
                found_valid_coloring = True
                break

        # Select parents using tournament selection
        selected_parents = tournament_selection(population, fitness_values)

        # Generate new population
        new_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1 if i + 1 < population_size else 0]

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1, changed_nodes_child1 = mutate(child1, max_colors, mutation_rate)
            child2, changed_nodes_child2 = mutate(child2, max_colors, mutation_rate)

            new_population.extend([child1, child2])
            changed_nodes_list.extend([changed_nodes_child1, changed_nodes_child2])

        # Elitism: Keep the best candidate from the previous generation
        new_population[0] = best_candidate.copy()
        population = new_population

        # Early stopping if valid coloring is found
        if found_valid_coloring:
            break

    elapsed_time = time.time() - start_time_coloring
    num_colors_used = len(set(best_candidate)) if found_valid_coloring else 0

        # Return detailed information
    if found_valid_coloring:
            # Ensure the number of colors generated matches the number of unique colors in the best candidate
            max_color_index = max(best_candidate)  # Find the highest color index used in best_candidate
            colors = [generate_random_color() for _ in range(max_color_index + 1)]  # Generate enough colors
            
            # Map each color index to a color
            color_assignment = [colors[color_index] for color_index in best_candidate]

            # Draw the graph with the found color assignment
            draw_graph(color_assignment)


    return {
        "Elapsed Time": elapsed_time,
        "Colors Used": num_colors_used if found_valid_coloring else 0,
        "Best Fitness": best_fitness,
        "Fitness Evaluations": fitness_evaluations,
        "Nodes": numOfNodes,
        "Edges": numOfEdges
    }





# Tournament Selection
def tournament_selection(population, fitness_values, tournament_size=2):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(range(len(population)), tournament_size)
        best = min(participants, key=lambda x: fitness_values[x])
        selected.append(population[best].copy())
    return selected

# Crossover Function
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Mutation Function
def mutate(candidate, current_num_colors, mutation_rate):
    changed_nodes = []
    for i in range(len(candidate)):
        if random.random() < mutation_rate:
            candidate[i] = random.randint(0, current_num_colors - 1)
            changed_nodes.append(i)
    return candidate, changed_nodes

# Function to log results to a dynamically named text file
# Function to log results to a dynamically named text file
def log_results_to_file(results, dimacs):
    file_path = f"{dimacs}_performance_test.txt"  # Use dimacs name for the log file
    with open(file_path, "a") as f:
        f.write(f"Elapsed Time: {results['Elapsed Time']:.3f} seconds\n")
        f.write(f"Colors Used: {results['Colors Used']}\n")
        f.write(f"Best Fitness: {results['Best Fitness']}\n")
        f.write(f"Total Fitness Evaluations: {results['Fitness Evaluations']}\n")
        f.write(f"Nodes: {results['Nodes']}\n")
        f.write(f"Edges: {results['Edges']}\n")
        f.write("------\n")  # Separator for each run

    print(f"Results logged to {file_path}")

# Function to log results to a dynamically named text file
def log_results_to_file(results, dimacs):
    file_path = f"{dimacs}_performance_test.txt"  # Use dimacs name for the log file
    with open(file_path, "a") as f:
        f.write(f"Elapsed Time: {results['Elapsed Time']:.3f} seconds\n")
        f.write(f"Colors Used: {results['Colors Used']}\n")
        f.write(f"Best Fitness: {results['Best Fitness']}\n")
        f.write(f"Total Fitness Evaluations: {results['Fitness Evaluations']}\n")
        f.write(f"Nodes: {results['Nodes']}\n")
        f.write(f"Edges: {results['Edges']}\n")
        f.write("------\n")  # Separator for each run

    print(f"Results logged to {file_path}")

# Function to calculate averages and update or add them to the master file `all_figures_averages.txt`
def calculate_averages_from_log(dimacs):
    log_file_path = f"{dimacs}_performance_test.txt"
    master_file_path = "all_figures_averages.txt"  # Master file for all figures

    total_time = 0.0
    total_colors = 0
    total_fitness = 0
    total_evaluations = 0
    run_count = 0
    nodes, edges = None, None

    # Read the performance test log file to calculate averages
    with open(log_file_path, "r") as f:
        for line in f:
            if line.startswith("Elapsed Time"):
                total_time += float(line.split(": ")[1].split()[0])
            elif line.startswith("Colors Used"):
                total_colors += int(line.split(": ")[1])
            elif line.startswith("Best Fitness"):
                total_fitness += float(line.split(": ")[1])
            elif line.startswith("Total Fitness Evaluations"):
                total_evaluations += int(line.split(": ")[1])
            elif line.startswith("Nodes"):
                nodes = int(line.split(": ")[1])  # Extract node count
            elif line.startswith("Edges"):
                edges = int(line.split(": ")[1])  # Extract edge count
                run_count += 1  # Increment run count only when both nodes and edges are found

    if run_count == 0:
        print("No data to calculate averages.")
        return

    # Calculate averages
    avg_time = total_time / run_count
    avg_colors = total_colors / run_count
    avg_fitness = total_fitness / run_count
    avg_evaluations = total_evaluations / run_count

    # Prepare the new figure's average entry
    figure_entry = (f"Figure: {dimacs}, Nodes: {nodes}, Edges: {edges}, "
                    f"Avg Elapsed Time: {avg_time:.3f} seconds, "
                    f"Avg Colors Used: {avg_colors:.2f}, "
                    f"Avg Best Fitness: {avg_fitness}, "
                    f"Avg Fitness Evaluations: {avg_evaluations}, "
                    f"Run Count: {run_count}\n")

    # Read the current contents of the master file
    if os.path.exists(master_file_path):
        with open(master_file_path, "r") as master_file:
            lines = master_file.readlines()
    else:
        lines = []

    # Check if the figure already exists in the master file
    updated_lines = []
    figure_exists = False
    for line in lines:
        if line.startswith(f"Figure: {dimacs},"):
            # Replace the existing entry with the new averages
            updated_lines.append(figure_entry)
            figure_exists = True
        else:
            updated_lines.append(line)

    # If the figure does not exist, append it as a new entry
    if not figure_exists:
        updated_lines.append(figure_entry)

    # Write the updated contents back to the master file
    with open(master_file_path, "w") as master_file:
        master_file.writelines(updated_lines)

    print(f"Averages updated in {master_file_path}")

def main():
    global selected_node, stop_flag

    # Initialize adjacency matrix and list
    init_adj_matrix_and_list(dimacs_path)

    # Start evolutionary graph coloring and get detailed results
    results = evolutionary_graph_coloring()

    # Log the results to the dynamically named text file
    log_results_to_file(results, dimacs)

    # Update the master file with averages from all figures
    calculate_averages_from_log(dimacs)

    # Event loop to keep the window open
    running = True
    while running:
        pygame.QUIT
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()  # Keep updating the screen
    pygame.quit()

if __name__ == "__main__":
    main()


