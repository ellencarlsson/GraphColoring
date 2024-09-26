import logging
import evolutionComputation
import pygame
import recordsHolder
import graphCreator

mainpath = "DIMACS_graphs/"
dimacs = "small_3"
dimacs_path = mainpath + dimacs + ".txt"

def main():
    global selected_node, stop_flag

    # Initialize adjacency matrix and list
    evolutionComputation.init_adj_matrix_and_list(dimacs_path)

    # Start evolutionary graph coloring and get detailed results
    results = evolutionComputation.evolutionary_graph_coloring()

    # Log the results to the dynamically named text file
    recordsHolder.log_results_to_file(results, dimacs)

    # Update the master file with averages from all figures
    recordsHolder.calculate_averages_from_log(dimacs)

    # Event loop to keep the window open
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()  # Keep updating the screen
    pygame.quit()

if __name__ == "__main__":
    for i in range(60):
        main()
