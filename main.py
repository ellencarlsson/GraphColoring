import logging
import evolutionComputation
import pygame

mainpath = "DIMACS_graphs/"

dimacs = "large_1"
dimacs_path = mainpath + dimacs + ".txt"

"""def main():
    global selected_node, stop_flag

    # Initialize adjacency matrix and list
    evolutionComputation.init_adj_matrix_and_list(dimacs_path)

    # Start evolutionary graph coloring and get detailed results
    results = evolutionComputation.evolutionary_graph_coloring()

    # Log the results to the dynamically named text file
    recordsHolder.log_results_to_file(results, dimacs)

    # Update the master file with averages from all figures
    recordsHolder.calculate_averages_from_log(dimacs)

   

if __name__ == "__main__":
    for i in range(60):
        main()"""


def main():
    global selected_node, stop_flag

    # Initialize adjacency matrix and list
    evolutionComputation.init_adj_matrix_and_list(dimacs_path)

    evolutionComputation.evolutionary_graph_coloring(dimacs)
    evolutionComputation.evolutionary_graph_coloring_min_colors(dimacs)


if __name__ == "__main__":
    for i in range(60):
        main()
