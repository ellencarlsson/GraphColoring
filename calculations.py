import os
import re

# Function to find the latest file number and increment it
def get_next_file_number(output_dir, figureCropped, numberCropped):
    # Path to the directory containing the files
    base_dir = os.path.join(output_dir, figureCropped, numberCropped)
    
    # Ensure the base directory exists, or create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1  # If no previous files, start with 1

    # Get the list of existing .txt files
    existing_files = [int(f.split(".")[0]) for f in os.listdir(base_dir) if f.endswith(".txt") and f.split(".")[0].isdigit()]
    
    # Return the next file number (if no files, return 1)
    return max(existing_files) + 1 if existing_files else 1

# Function to save input parameters and fitness values for each run as .txt files
def store_fitness_values(minColors, figure, average_fitness, best_fitness, max_generations, population_size, mutation_rate, elapsed_time, num_colors_used, output_dir="output"):
    generations = range(len(average_fitness))

    print("hoo", num_colors_used)
    # Extract figure and number before and after the underscore
    figureCropped = re.match(r"([^_]+)_", figure)
    if figureCropped:
        figureCropped = figureCropped.group(1)
    
    numberCropped = re.match(r".*_(\d+)", figure)
    if numberCropped:
        numberCropped = numberCropped.group(1)

    if minColors:
        figureCropped = figureCropped + "_min_colors"

    # Get the next file number based on existing .txt files
    file_number = get_next_file_number(output_dir, figureCropped, numberCropped)
    
    # Define the new file path: output/figureCropped/numberCropped/file_number.txt
    figure_dir = os.path.join(output_dir, figureCropped, numberCropped)
    fitness_file_path = os.path.join(figure_dir, f"{file_number}.txt")

    # Saving input parameters and fitness values to a text file
    
    with open(fitness_file_path, "w") as f:
        # Store the input parameters at the top of the file
        f.write(f"Figure: {figure}\n")
        f.write(f"File Number: {file_number}\n")
        f.write(f"Max Generations: {max_generations}\n")
        f.write(f"Population Size: {population_size}\n")
        f.write(f"Mutation Rate: {mutation_rate}\n")
        f.write(f"Colors Used: {num_colors_used}\n")
        f.write(f"Runtime: {elapsed_time:.4f} s\n")
        f.write("\n")  # Separate parameters from the data
        
        # Store the fitness values per generation
        f.write("Generation,Average Fitness,Best Fitness\n")
        for gen, avg_fit, best_fit in zip(generations, average_fitness, best_fitness):
            f.write(f"{gen},{avg_fit},{best_fit}\n")



# Evolutionary Graph Coloring with parallelization and incremental evaluation (no visualization)
"""def evolutionary_graph_coloring_with_convergence():
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
"""