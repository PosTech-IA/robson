import random
import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithm import genetic_algorithm

# Parameters for the genetic algorithm
population_size = 100
lower_bound = -50
upper_bound = 50
tournament_size=4
generations = 20
mutation_rate = 2
crossover_rate= 0.7

def main():
    best_solution = genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate,tournament_size,crossover_rate)
    print(f"Melhor solução encontrada: a = {best_solution[0]}, b = {best_solution[1]}, c = {best_solution[2]}")

if __name__ == "__main__":
    main()