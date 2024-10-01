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
    numOfNodes = None  # Changed to match the rest of the code
    numOfEdges = None  # Changed to match the rest of the code

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
        elif "Nodes" in line:
            numOfNodes = int(line.split(":")[-1].strip())  
        elif "Edges" in line:
            numOfEdges = int(line.split(":")[-1].strip())  
    
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
        "best_fitness": best_fitness,  # Updated with the new 0 appended if necessary
        "numOfNodes": numOfNodes,  
        "numOfEdges": numOfEdges   
    }


import numpy as np

def calculate_average_data(data_list):
    """Calculates the average values of generations, average fitness, best fitness, colors used, and runtime across multiple files."""
    
    # Find the minimum length of data across all files to avoid dimension mismatch
    min_length = min(len(data['generations']) for data in data_list)
    
    # Initialize arrays to accumulate fitness values
    avg_fitness_sum = np.zeros(min_length)
    best_fitness_sum = np.zeros(min_length)
    
    # Variables to accumulate colors used and runtime
    colors_used_sum = 0
    runtime_sum = 0
    
    # Accumulate data for each generation, limiting to the minimum length found
    for data in data_list:
        avg_fitness_sum += np.array(data['avg_fitness'][:min_length])
        best_fitness_sum += np.array(data['best_fitness'][:min_length])
        
        # Accumulate colors used and runtime
        colors_used_sum += data['colors_used']
        runtime_sum += data['runtime']
    
    # Calculate average values by dividing by the number of files
    num_files = len(data_list)
    avg_fitness_avg = avg_fitness_sum / num_files
    best_fitness_avg = best_fitness_sum / num_files
    avg_colors_used = colors_used_sum / num_files
    avg_runtime = runtime_sum / num_files

    # Extract the corresponding generations (limited to min_length)
    generations = data_list[0]['generations'][:min_length]
    
    return generations, avg_fitness_avg, best_fitness_avg, avg_colors_used, avg_runtime


def plot_fitness_data(generations, avg_fitness, best_fitness, figure_name, output_path, max_generations, population_size, mutation_rate, colors_used, runtime, nodes, edges):
    """Plots the fitness data and saves the figure."""
    plt.figure(figsize=(10, 6))
    
    # Ensure that the last best fitness is 0
    if best_fitness[-1] != 0:
        best_fitness[-1] = 0
    
    # Check if generations and best_fitness have the same length
    if len(generations) != len(best_fitness):
        # Make sure they match by appending a generation
        next_generation = len(generations)
        generations = np.append(generations, next_generation)
    
    # Plot average and best fitness lines
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.plot(generations, best_fitness, label="Best Fitness", color="green", linestyle="--")
    
    # Add title and other details
    plt.suptitle(f"{figure_name} - Nodes: {nodes} Edges: {edges}", fontsize=16, fontweight='bold')
    plt.title(f"Max Generations: {max_generations}  Population Size: {population_size}  Mutation Rate: {mutation_rate * 100}%  Colors Used: {int(colors_used)}  Runtime: {runtime:.2f} s")

 # Mark start and end points of Average Fitness
    plt.scatter(generations[0], avg_fitness[0], color="blue", s=100, zorder=5, label="Start Avg Fitness", marker='o')
    plt.scatter(generations[-1], avg_fitness[-1], color="blue", s=100, zorder=5, label="End Avg Fitness", marker='o')
    
    # Mark start and end points of Best Fitness
    plt.scatter(generations[0], best_fitness[0], color="green", s=100, zorder=5, label="Start Best Fitness", marker='o')
    plt.scatter(generations[-1], best_fitness[-1], color="green", s=100, zorder=5, label="End Best Fitness", marker='o')
    
    # Ensure y-axis starts at 0
    plt.ylim(bottom=0)

    # Add labels and legend
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()

    # Save and close the plot
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
    
    # Calculate average fitness values, colors used, and runtime
    generations, avg_fitness_avg, best_fitness_avg, avg_colors_used, avg_runtime = calculate_average_data(data_list)
    
    # Use the figure name from the first file
    figure_name = data_list[0]["figure_name"]
    max_generations = data_list[0]["max_generations"]
    population_size = data_list[0]["population_size"]
    mutation_rate = data_list[0]["mutation_rate"]
    nodes = data_list[0]["numOfNodes"]
    edges = data_list[0]["numOfEdges"]

    
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
                      avg_colors_used,
                      avg_runtime,
                      nodes,
                      edges
                      )

    print(f"Plot saved to {output_path}")


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
def store_fitness_values(minColors, figure, average_fitness, best_fitness, max_generations, population_size, mutation_rate, elapsed_time, num_colors_used, numOfNodes, numOfEdges, output_dir="output"):
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

    # Saving input parameters and fitness values to a text file
    
    with open(fitness_file_path, "w") as f:
        # Store the input parameters at the top of the file
        f.write(f"Figure: {figure}\n")
        f.write(f"File Number: {file_number}\n")
        f.write(f"Nodes: {numOfNodes}\n")
        f.write(f"Edges: {numOfEdges}\n")
        f.write(f"Max Generations: {max_generations}\n")
        f.write(f"Population Size: {population_size}\n")
        f.write(f"Mutation Rate: {mutation_rate}\n")
        f.write(f"Colors Used: {num_colors_used}\n")
        f.write(f"Runtime: {elapsed_time:.2f}\n")
        f.write("\n")  # Separate parameters from the data
        
        # Store the fitness values per generation
        f.write("Generation,Average Fitness,Best Fitness\n")
        for gen, avg_fit, best_fit in zip(generations, average_fitness, best_fitness):
            f.write(f"{gen},{avg_fit},{best_fit}\n")
    
    outputPath = os.path.join(figure_dir, f"average")
    getAvgFile(figure_dir, outputPath)
    #def getAvgFile(folder_path, output_dir):
