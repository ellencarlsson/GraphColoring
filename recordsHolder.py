import os

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