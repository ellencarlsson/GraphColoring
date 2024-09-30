import time
import threading
import random
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os
from concurrent.futures import ThreadPoolExecutor

# Global variables
adj_matrix = []  
adj_list = []     
selected_node = None  
stop_flag = False  

monitor_stop_flag = False
monitor_lock = threading.Lock()


def init_adj_matrix_and_list(dimacs_path):
    global nodes, edges, adj_matrix, adj_list, numOfNodes, numOfEdges 

    nodes = []
    edges = []
    numOfNodes = 0  
    numOfEdges = 0

    with open(dimacs_path, 'r') as f:
        for line in f:

            if line.startswith('p edge'):
                _, _, num_nodes, num_edges = line.strip().split()
                numOfNodes = int(num_nodes)  
                numOfEdges = int(num_edges)  
                num_nodes = int(num_nodes)

                num_columns = math.ceil(math.sqrt(num_nodes))  
                num_rows = math.ceil(num_nodes / num_columns)  

                x_spacing = 800 // (num_columns + 1)
                y_spacing = 800 // (num_rows + 1)

                for i in range(num_nodes):
                    row = i // num_columns
                    col = i % num_columns
                    x = (col + 1) * x_spacing 
                    y = (row + 1) * y_spacing  
                    nodes.append((x, y)) 

            elif line.startswith('e'):
                _, node1, node2 = line.strip().split()
                node1 = int(node1) - 1  
                node2 = int(node2) - 1 
                edges.append((node1, node2))  

    n = len(nodes)
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    
    for edge in edges:
        node1, node2 = edge
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1 
        adj_list[node1].append(node2)
        adj_list[node2].append(node1)



def fitness(candidate, changed_nodes):
    conflicts = 0
    for node in changed_nodes:
        for neighbor in adj_list[node]:
            if candidate[node] == candidate[neighbor]:
                conflicts += 1
    return conflicts

# Parallel Fitness Evaluation
def evaluate_population(population, fitness_values, changed_nodes_list):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fitness, candidate, changed_nodes_list[i])
            for i, candidate in enumerate(population)
        ]
        for i, future in enumerate(futures):
            fitness_values[i] = future.result()

def evolutionary_graph_coloring(figure):
    population_size = 100  # Reduced to balance computational cost and diversity
    max_generations = 1000  # Reduced due to the small size of the graph
    mutation_rate = 0.02
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

    # Initialize lists to store fitness values for plotting
    average_fitness_per_generation = []
    best_fitness_per_generation = []

    for generation in range(max_generations):
        # Create a list to track nodes that changed due to mutation
        changed_nodes_list = [list(range(len(nodes))) for _ in range(population_size)]
        fitness_values = [0] * population_size

        # Evaluate fitness in parallel
        evaluate_population(population, fitness_values, changed_nodes_list)
        fitness_evaluations += population_size  # Increment fitness evaluations counter

        # Track fitness values for plotting
        avg_fitness = sum(fitness_values) / population_size
        best_gen_fitness = min(fitness_values)
        
        average_fitness_per_generation.append(avg_fitness)
        best_fitness_per_generation.append(best_gen_fitness)

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

    print(average_fitness_per_generation, best_fitness_per_generation)

    saveRec(figure, average_fitness_per_generation, best_fitness_per_generation, max_generations, population_size, mutation_rate)
    #plot_fitness_vs_generation(figure, average_fitness_per_generation, best_fitness_per_generation, max_generations, population_size, mutation_rate)

    return {
        "Elapsed Time": elapsed_time,
        "Colors Used": num_colors_used if found_valid_coloring else 0,
        "Best Fitness": best_fitness,
        "Fitness Evaluations": fitness_evaluations,
        "Nodes": numOfNodes,
        "Edges": numOfEdges
    }

def saveRec(figure, average_fitness, best_fitness, max_generations, population_size, mutation_rate, run_number=None, output_dir="output"):
    generations = range(len(average_fitness))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique filename for each run based on run_number or timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_number}" if run_number is not None else f"run_{timestamp}"

    # Saving fitness values to a text file with unique run identifier
    fitness_file_path = os.path.join(output_dir, f"{figure}_{run_id}_fitness_values.txt")
    with open(fitness_file_path, "w") as f:
        f.write("Generation,Average Fitness,Best Fitness\n")
        for gen, avg_fit, best_fit in zip(generations, average_fitness, best_fitness):
            f.write(f"{gen},{avg_fit},{best_fit}\n")
    print(f"Fitness values saved to {fitness_file_path}")

def plot_fitness_vs_generation(figure, average_fitness, best_fitness, max_generations, population_size, mutation_rate, run_number=None, output_dir="output"):
    generations = range(len(average_fitness))

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a unique filename for each run based on run_number or timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{run_number}" if run_number is not None else f"run_{timestamp}"

    # Saving fitness values to a text file with unique run identifier
    fitness_file_path = os.path.join(output_dir, f"{figure}_{run_id}_fitness_values.txt")
    with open(fitness_file_path, "w") as f:
        f.write("Generation,Average Fitness,Best Fitness\n")
        for gen, avg_fit, best_fit in zip(generations, average_fitness, best_fitness):
            f.write(f"{gen},{avg_fit},{best_fit}\n")
    print(f"Fitness values saved to {fitness_file_path}")

    # Plot the average and best fitness with line style
    plt.figure(figsize=(10, 6))
    plt.plot(generations, average_fitness, label="Average Fitness", color="blue", linestyle="-", linewidth=2)
    plt.plot(generations, best_fitness, label="Best Fitness", color="green", linestyle="-", linewidth=2)

    # Title and labels
    plt.title(f"Fitness and Generations for {figure}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    # Plotting a single point for average and best fitness
    plt.scatter(generations, average_fitness, color="blue", label="Average Fitness")
    plt.scatter(generations, best_fitness, color="green", label="Best Fitness")

    # Ensure the Y-axis starts from 0 to show zero values
    plt.ylim(bottom=0)

    # Highlight when best fitness reaches zero (optional)
    zero_fitness_generations = [gen for gen, fitness in enumerate(best_fitness) if fitness == 0]
    for gen in zero_fitness_generations:
        plt.axvline(x=gen, color='red', linestyle='--', label="Best Fitness = 0")

    # Adding the parameters to the graph as text annotations
    annotation_text = (
        f"Population Size: {population_size}\n"
        f"Max Generations: {max_generations}\n"
        f"Mutation Rate: {mutation_rate}"
    )

    # Position the text box in the plot
    plt.text(0.71, 0.75, annotation_text, horizontalalignment='left', verticalalignment='center',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=1), fontsize=12)

    # Adding grid and legend
    plt.grid(True)
    plt.legend()

    # Save the plot as an image with unique run identifier
    plot_file_path = os.path.join(output_dir, f"{figure}_{run_id}_fitness_plot.png")
    plt.savefig(plot_file_path)
    print(f"Plot saved to {plot_file_path}")

    # Show the plot
    plt.show()

# Evolutionary Graph Coloring with parallelization and incremental evaluation (no visualization)
def evolutionary_graph_coloring_with_convergence():
    population_size = 50
    max_generations = 20  # Adjust this for testing purposes
    mutation_rate = 0.01
    max_colors = len(nodes)

    # Initialize lists to store convergence data for different EAs
    generations = list(range(max_generations))
    fitness_values_list = []
    ea_labels = ['EA1', 'EA2', 'EA3']  # You can change this to more EAs if necessary

    for ea_run in range(3):  # Running EA1, EA2, EA3
        # Start timing
        start_time_coloring = time.time()

        fitness_values = []  # To store fitness values per generation for this EA run

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
            fitness_values_generation = [0] * population_size

            # Evaluate fitness in parallel
            evaluate_population(population, fitness_values_generation, changed_nodes_list)
            fitness_evaluations += population_size  # Increment fitness evaluations counter

            # Find the best candidate
            min_fitness_in_population = min(fitness_values_generation)
            if min_fitness_in_population < best_fitness:
                best_fitness = min_fitness_in_population
                best_candidate = population[fitness_values_generation.index(min_fitness_in_population)]
                if best_fitness == 0:
                    found_valid_coloring = True
                    break

            # Track best fitness value for this generation
            fitness_values.append(min_fitness_in_population)

            # Select parents using tournament selection
            selected_parents = tournament_selection(population, fitness_values_generation)

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

        # Append this EA's fitness values to the list for plotting later
        fitness_values_list.append(fitness_values)

    # No convergence plot, just return the data
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
