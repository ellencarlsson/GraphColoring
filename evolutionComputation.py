# evolutionComputation.py
import time
import random
import numpy as np
import calculations
import math
import re
import threading

# Global variables
adj_matrix = []  
adj_list = []     
selected_node = None  
stop_flag = False  

monitor_stop_flag = False
monitor_lock = threading.Lock()

population_size = 100  # Adjusted for balance between performance and diversity
max_generations = 1000  # Adjusted for performance
mutation_rate = 0.01

edges_np_array = None  # Declare at module level

def init_adj_matrix_and_list(dimacs_path):
    global nodes, edges, adj_matrix, adj_list, numOfNodes, numOfEdges, edges_np_array

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

    # Convert edges to NumPy array for vectorized operations
    edges_np_array = np.array(edges)

def evaluate_population(population):
    """Vectorized fitness evaluation using NumPy arrays."""
    conflicts = np.sum(population[:, edges_np_array[:, 0]] == population[:, edges_np_array[:, 1]], axis=1)
    return conflicts

def tournament_selection(population, fitness_values, tournament_size=2):
    """Vectorized tournament selection."""
    selected_indices = []
    for _ in range(population.shape[0]):
        participants = np.random.choice(population.shape[0], tournament_size, replace=False)
        best = participants[np.argmin(fitness_values[participants])]
        selected_indices.append(best)
    selected_population = population[selected_indices]
    return selected_population

def crossover_population(parents):
    """Vectorized crossover operation."""
    num_parents = parents.shape[0]
    np.random.shuffle(parents)
    if num_parents % 2 != 0:
        parents = parents[:-1]
    half = num_parents // 2
    parent1 = parents[:half]
    parent2 = parents[half:2*half]

    crossover_points = np.random.randint(1, parents.shape[1], size=half)
    children = np.empty_like(parents[:2*half])

    for i in range(half):
        cp = crossover_points[i]
        children[2*i] = np.concatenate([parent1[i, :cp], parent2[i, cp:]])
        children[2*i + 1] = np.concatenate([parent2[i, :cp], parent1[i, cp:]])
    return children

def mutate(population, mutation_rate, num_colors):
    """Vectorized mutation operation."""
    mutation_mask = np.random.rand(*population.shape) < mutation_rate
    random_colors = np.random.randint(0, num_colors, size=population.shape)
    population[mutation_mask] = random_colors[mutation_mask]
    return population

def evolutionary_graph_coloring_min_colors(figure, num_trials=10):
    max_colors = len(nodes) # Set the initial maximum number of colors to the number of nodes in the graph

    start_time_coloring = time.time() # Record start time  
    improvement_time = 0 #Initialize a variable to keep time spent on improving solution

    # Variables to keep tack of best overall candidate solution
    best_overall_candidate = None
    best_overall_num_colors = max_colors
    best_fitness_overall = float('inf')

    num_nodes = len(nodes) #Total number of nodes

    for trial in range(num_trials): #Start the trials for fiding diffrent solutions
        print(f"Starting trial {trial + 1}/{num_trials}...") #Print out trial data

        if best_overall_num_colors != max_colors:
            local_max_colors = best_overall_num_colors
        else:
            local_max_colors = max_colors

        while local_max_colors > 1:
            phase_start_time = time.time()
            population = np.random.randint(0, local_max_colors, size=(population_size, num_nodes))

            best_candidate = None
            best_fitness = float('inf')
            found_valid_coloring = False

            average_fitness_per_generation = []
            best_fitness_per_generation = []

            for generation in range(max_generations):
                fitness_values = evaluate_population(population)
                avg_fitness = np.mean(fitness_values)
                best_gen_fitness = np.min(fitness_values)

                average_fitness_per_generation.append(avg_fitness)
                best_fitness_per_generation.append(best_gen_fitness)

                min_fitness_in_population = np.min(fitness_values)
                if min_fitness_in_population < best_fitness:
                    best_fitness = min_fitness_in_population
                    best_candidate = population[np.argmin(fitness_values)].copy()
                    if best_fitness == 0:
                        found_valid_coloring = True
                        break

                selected_parents = tournament_selection(population, fitness_values)
                children = crossover_population(selected_parents)
                population = mutate(children, mutation_rate, local_max_colors)
                population[0] = best_candidate.copy()  # Elitism

            phase_elapsed_time = time.time() - phase_start_time
            num_colors_used = len(set(best_candidate)) if found_valid_coloring else 0
            print(f"Trial {trial + 1}, Solution with {num_colors_used} colors.")

            if not found_valid_coloring:
                print("Unable to find a conflict-free solution.")
                break

            if found_valid_coloring and num_colors_used < best_overall_num_colors:
                improvement_time += phase_elapsed_time
                best_overall_num_colors = num_colors_used
                best_overall_candidate = best_candidate.copy()
                best_fitness_overall = best_fitness
            else:
                break

            local_max_colors = num_colors_used - 1

        if not found_valid_coloring:
            break

    elapsed_time = time.time() - start_time_coloring
    print(f"Best coloring uses {best_overall_num_colors} colors. Total time: {elapsed_time:.2f} seconds.")

    # Store the best results
    calculations.store_fitness_values(
        True,
        figure, 
        average_fitness_per_generation, 
        best_fitness_per_generation, 
        max_generations, 
        population_size, 
        mutation_rate, 
        improvement_time,  
        best_overall_num_colors,
        numOfNodes, 
        numOfEdges
    )
