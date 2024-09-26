import matplotlib.pyplot as plt
import time
import random

# Function to plot convergence data
def plot_convergence(generations, fitness_values_list, labels, title='Convergence Analysis', xlabel='Generation', ylabel='Fitness function value'):
    """
    Plots the convergence of different algorithms over generations.
    
    :param generations: List of generation numbers (X-axis)
    :param fitness_values_list: List of lists where each list contains fitness values for a specific EA run
    :param labels: List of labels for each EA run (e.g., ['EA1', 'EA2', 'EA3'])
    :param title: Title of the plot (default: 'Convergence Analysis')
    :param xlabel: Label for the X-axis (default: 'Generation')
    :param ylabel: Label for the Y-axis (default: 'Fitness function value')
    """
    # Check if inputs are valid
    assert len(fitness_values_list) == len(labels), "Each EA run must have a label."

    # Plot each EA run
    for fitness_values, label in zip(fitness_values_list, labels):
        plt.plot(generations, fitness_values, label=label, marker='o')

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Show legend
    plt.legend()
    
    # Show grid
    plt.grid(True)

    # Show plot
    plt.show()