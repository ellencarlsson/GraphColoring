import time
import threading
import random
import os
import math
from concurrent.futures import ThreadPoolExecutor
import graphCreator

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

                x_spacing = graphCreator.width // (num_columns + 1)
                y_spacing = graphCreator.height // (num_rows + 1)

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

# Fitness function calculates the number of color conflicts and we want value 0
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
            colors = [graphCreator.generate_random_color() for _ in range(max_color_index + 1)]  # Generate enough colors
            
            # Map each color index to a color
            color_assignment = [colors[color_index] for color_index in best_candidate]

            # Draw the graph with the found color assignment
            graphCreator.draw_graph(color_assignment, nodes, edges)


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



