import logging
import evolutionComputation
import pygame

mainpath = "DIMACS_graphs/"

#figures = ["small_", "medium_", "large_"]
figures = ["small_"]

numbers = ["1", "2", "3", "4", "5"]


def main():
    global selected_node, stop_flag

    for figure in figures:
        for number in numbers:
            dimacs = figure + number
            dimacs_path = mainpath + dimacs + ".txt"

            print(dimacs, dimacs_path)

            evolutionComputation.init_adj_matrix_and_list(dimacs_path)

            evolutionComputation.evolutionary_graph_coloring(dimacs)
            #evolutionComputation.evolutionary_graph_coloring_min_colors(dimacs)


if __name__ == "__main__":
    for i in range(100):
        main()
