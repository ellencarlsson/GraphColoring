import os
import re

import os
import re
import matplotlib.pyplot as plt

def extract_data_from_txt(file_path):
    """Extracts the generation, average fitness, and best fitness from a given txt file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract the metadata from the file (Figure, Population Size, etc.)
    figure_name = None
    max_generations = None
    population_size = None
    mutation_rate = None
    colors_used = None
    runtime = None

    # Loop through the top lines to extract metadata
    for line in lines:
        if "Figure" in line:
            figure_name = line.split(":")[-1].strip()
        elif "Max Generations" in line:
            max_generations = int(line.split(":")[-1].strip())
        elif "Population Size" in line:
            population_size = int(line.split(":")[-1].strip())
        elif "Mutation Rate" in line:
            mutation_rate = float(line.split(":")[-1].strip())
        elif "Colors Used" in line:
            colors_used = int(line.split(":")[-1].strip())
        elif "Runtime" in line:
            runtime = float(line.split(":")[-1].strip())
    
    # Extract the fitness data (Generation, Average Fitness, Best Fitness)
    data_start = lines.index("Generation,Average Fitness,Best Fitness\n") + 1
    data_lines = lines[data_start:]
    
    # Store the data in lists
    generations = []
    avg_fitness = []
    best_fitness = []
    
    for data_line in data_lines:
        gen, avg_fit, best_fit = map(float, data_line.strip().split(","))
        generations.append(gen)
        avg_fitness.append(avg_fit)
        best_fitness.append(best_fit)
    
    return {
        "figure_name": figure_name,
        "max_generations": max_generations,
        "population_size": population_size,
        "mutation_rate": mutation_rate,
        "colors_used": colors_used,
        "runtime": runtime,
        "generations": generations,
        "avg_fitness": avg_fitness,
        "best_fitness": best_fitness
    }
import numpy as np

def calculate_average_data(data_list):
    """Calculates the average values of generations, average fitness, and best fitness across multiple files."""
    # Find the minimum length of data across all files to avoid dimension mismatch
    min_length = min(len(data['generations']) for data in data_list)
    
    # Initialize arrays to accumulate fitness values
    avg_fitness_sum = np.zeros(min_length)
    best_fitness_sum = np.zeros(min_length)

    # Accumulate data for each generation, limiting to the minimum length found
    for data in data_list:
        avg_fitness_sum += np.array(data['avg_fitness'][:min_length])
        best_fitness_sum += np.array(data['best_fitness'][:min_length])
    
    # Calculate average values by dividing by the number of files
    num_files = len(data_list)
    avg_fitness_avg = avg_fitness_sum / num_files
    best_fitness_avg = best_fitness_sum / num_files

    # Extract the corresponding generations (limited to min_length)
    generations = data_list[0]['generations'][:min_length]
    
    return generations, avg_fitness_avg, best_fitness_avg

def plot_fitness_data(generations, avg_fitness, best_fitness, figure_name, output_path, max_generations, population_size, mutation_rate, colors_used, runtime):
    """Plots the fitness data and saves the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.plot(generations, best_fitness, label="Best Fitness", color="green", linestyle="--")
    
    plt.suptitle(figure_name)
    plt.suptitle(figure_name, fontsize=16, fontweight='bold')
    plt.title(f"Max Generations: {max_generations}  Population Size: {population_size}  Mutation Rate: {mutation_rate * 100}%  Colors Used: {colors_used}  Runtime: {runtime:.4f} s")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    
    plt.savefig(output_path)
    plt.close()

def getAvgFile(folder_path, output_dir):
    """Processes all txt files in the folder, calculates averages, and stores the final plot."""
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    # Sort files by the file number embedded in the file name (assuming files have numbers in their names)
    txt_files.sort(key=lambda f: int(re.findall(r'\d+', f)[-1]))
    
    # Extract data from all txt files
    data_list = []
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        data = extract_data_from_txt(file_path)
        data_list.append(data)
    
    # Calculate average fitness values
    generations, avg_fitness_avg, best_fitness_avg = calculate_average_data(data_list)
    
    # Use the figure name from the first file
    figure_name = data_list[0]["figure_name"]
    max_generations = data_list[1]["max_generations"]
    population_size = data_list[2]["population_size"]
    mutation_rate = data_list[3]["mutation_rate"]
    colors_used = data_list[4]["colors_used"]
    runtime = data_list[4]["runtime"]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the output path for the plot image
    output_path = os.path.join(output_dir, f"{figure_name}.png")
    
    # Plot the fitness data and save the image
    plot_fitness_data(generations, 
                      avg_fitness_avg, 
                      best_fitness_avg, 
                      figure_name, 
                      output_path,
                      max_generations,
                      population_size,
                      mutation_rate,
                      colors_used,
                      runtime
                      )

    print(f"Plot saved to {output_path}")

# Example usage
# folder_path = "path_to_your_folder_containing_txt_files"
# output_dir = "path_to_output_directory"
# process_folder(folder_path, output_dir)



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

    print("figure_dir", figure_dir)
    print("fitness_file_path", fitness_file_path)
    # Saving input parameters and fitness values to a text file
    
    with open(fitness_file_path, "w") as f:
        # Store the input parameters at the top of the file
        f.write(f"Figure: {figureCropped}\n")
        f.write(f"File Number: {file_number}\n")
        f.write(f"Max Generations: {max_generations}\n")
        f.write(f"Population Size: {population_size}\n")
        f.write(f"Mutation Rate: {mutation_rate}\n")
        f.write(f"Colors Used: {num_colors_used}\n")
        f.write(f"Runtime: {elapsed_time:.4f}\n")
        f.write("\n")  # Separate parameters from the data
        
        # Store the fitness values per generation
        f.write("Generation,Average Fitness,Best Fitness\n")
        for gen, avg_fit, best_fit in zip(generations, average_fitness, best_fitness):
            f.write(f"{gen},{avg_fit},{best_fit}\n")
    
    outputPath = os.path.join(figure_dir, f"average")
    getAvgFile(figure_dir, outputPath)
    #def getAvgFile(folder_path, output_dir):




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