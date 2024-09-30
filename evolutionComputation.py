import time
import threading
import random
import matplotlib.pyplot as plt
import math
from datetime import datetime
import os
import re
from concurrent.futures import ThreadPoolExecutor
import calculations

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
    print(elapsed_time)
    num_colors_used = len(set(best_candidate)) if found_valid_coloring else 0
    print("heeej", num_colors_used)

    calculations.store_fitness_values(False, figure, average_fitness_per_generation, best_fitness_per_generation, max_generations, population_size, mutation_rate, elapsed_time, num_colors_used)

def evolutionary_graph_coloring_min_colors(figure, num_trials=10):
    population_size = 100  # Reduced to balance computational cost and diversity
    max_generations = 1000  # Adjust based on the size of the graph and performance
    mutation_rate = 0.02
    max_colors = len(nodes)

    # Start overall timing (just for reference, may not be needed)
    start_time_coloring = time.time()
    
    # Timer for improvement phases
    improvement_time = 0

    best_overall_candidate = None
    best_overall_num_colors = max_colors
    best_fitness_overall = float('inf')

    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}...")

        found_valid_coloring = False
        fitness_evaluations = 0
        if best_overall_num_colors != max_colors:
            local_max_colors = best_overall_num_colors
        else:
            local_max_colors = max_colors

        # Keep decreasing max_colors until we get a solution with conflicts > 0
        while local_max_colors > 1:
            # Start measuring the time for this improvement phase
            phase_start_time = time.time()

            # Initialize the population
            population = [
                [random.randint(0, local_max_colors - 1) for _ in range(len(nodes))]
                for _ in range(population_size)
            ]

            best_candidate = None
            best_fitness = float('inf')
            found_valid_coloring = False

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
                    child1, changed_nodes_child1 = mutate(child1, local_max_colors, mutation_rate)
                    child2, changed_nodes_child2 = mutate(child2, local_max_colors, mutation_rate)

                    new_population.extend([child1, child2])
                    changed_nodes_list.extend([changed_nodes_child1, changed_nodes_child2])

                # Elitism: Keep the best candidate from the previous generation
                new_population[0] = best_candidate.copy()
                population = new_population

                # Early stopping if valid coloring is found
                if found_valid_coloring:
                    break

            # Measure the elapsed time for this improvement phase
            phase_elapsed_time = time.time() - phase_start_time
            num_colors_used = len(set(best_candidate)) if found_valid_coloring else 0
            print(f"Trial {trial + 1}, Solution with {num_colors_used} colors.")

            if not found_valid_coloring:
                print("Unable to find a conflict-free solution.")
                break

            # If this trial resulted in an improvement, add the phase time to the total improvement time
            if found_valid_coloring and num_colors_used < best_overall_num_colors:
                improvement_time += phase_elapsed_time
                best_overall_num_colors = num_colors_used
                best_overall_candidate = best_candidate
                best_fitness_overall = best_fitness
            else:
                # If no improvement, we stop the timing here for the current trial
                break

            # Decrease the number of colors for the next iteration
            local_max_colors = num_colors_used - 1

            # If conflicts arise (i.e., we can't find a valid solution with the reduced color count), stop
            if best_fitness > 0:
                print(f"Lowest possible number of colors found in this trial: {num_colors_used + 1}")
                break

    # Output the best result across all trials
    print(f"Best overall solution uses {best_overall_num_colors}")

    # Store fitness values only for the best overall solution after all trials are complete
    calculations.store_fitness_values(
        True,
        figure, 
        average_fitness_per_generation, 
        best_fitness_per_generation, 
        max_generations, 
        population_size, 
        mutation_rate, 
        improvement_time,  # Pass the improvement time here
        best_overall_num_colors
    )

