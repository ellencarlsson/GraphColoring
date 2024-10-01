import logging
import evolutionComputation
import pygame

mainpath = "DIMACS_graphs/"

dimacs = "small_5"
dimacs_path = mainpath + dimacs + ".txt"

def main():
    global selected_node, stop_flag

    # Initialize adjacency matrix and list
    evolutionComputation.init_adj_matrix_and_list(dimacs_path)

    #evolutionComputation.evolutionary_graph_coloring(dimacs) #100
    evolutionComputation.evolutionary_graph_coloring_min_colors(dimacs) #10


if __name__ == "__main__":
    for i in range(100):
        main()
