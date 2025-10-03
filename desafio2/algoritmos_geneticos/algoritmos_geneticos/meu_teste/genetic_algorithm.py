# genetic_algorithm.py

import random
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# Placeholder functions for the genetic algorithm.
# You will need to define these in your main script.
def fitness_function(params):
    a, b, c = params
    if a <= 0:
        return -float('inf')  # Penalize downward facing u-shapes heavily
    vertex_x = -b / (2 * a) #x value at vertex
    vertex_y = a * (vertex_x ** 2) + b * vertex_x + c #y value at vertex
    y_left = a * (-1) ** 2 + b * (-1) + c #y-coordinate at x = -1
    y_right = a * (1) ** 2 + b * (1) + c #y-coordinate at x = 1
    curviness = abs(y_left - vertex_y) + abs(y_right - vertex_y)
    return -curviness  # Negate to minimize curviness

def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2):
    alpha = random.random()
    child1 = tuple(alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2))
    child2 = tuple(alpha * p2 + (1 - alpha) * p1 for p1, p2 in zip(parent1, parent2))
    return child1, child2

# Mutation function
def mutation(individual, mutation_rate, lower_bound, upper_bound):
    individual = list(individual)
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutation_amount = random.uniform(-1, 1)
            individual[i] += mutation_amount
            # Ensure the individual stays within bounds
            individual[i] = max(min(individual[i], upper_bound), lower_bound)
    return tuple(individual)

# Main genetic algorithm function
def create_initial_population(size, lower_bound, upper_bound):
    """Creates the initial population of individuals."""
    population = []
    for _ in range(size):
        individual = (random.uniform(lower_bound, upper_bound),
                      random.uniform(lower_bound, upper_bound),
                      random.uniform(lower_bound, upper_bound))
        population.append(individual)
    print(population)
    return population

def genetic_algorithm(population_size, lower_bound, upper_bound, generations, mutation_rate, tournament_size,crossover_rate, plot_all=True):
    """Runs the genetic algorithm and returns the best solution."""
    
    population = create_initial_population(population_size, lower_bound, upper_bound)
    
    best_performers = []
    all_populations = []
    table = PrettyTable()
    table.field_names = ["Generation", "a", "b", "c", "Fitness"]

    for generation in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]
        best_individual = max(population, key=fitness_function)
        best_fitness = fitness_function(best_individual)
        
        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        table.add_row([generation + 1, round(best_individual[0], 4), round(best_individual[1], 4), round(best_individual[2], 4), round(best_fitness, 4)])

        next_population = []
        
        # Elitism: Add the best individual to the new population
        next_population.append(best_individual)

        # Fill the rest of the new population
        while len(next_population) < population_size:
            # Select two parents to create offspring
            tournament = random.sample(list(zip(population, fitnesses)), 2 * tournament_size)
            parent1 = max(tournament[:tournament_size], key=lambda x: x[1])[0]
            parent2 = max(tournament[tournament_size:], key=lambda x: x[1])[0]

            # Perform crossover or cloning
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                next_population.append(mutation(child1, mutation_rate, lower_bound, upper_bound))
                if len(next_population) < population_size:
                    next_population.append(mutation(child2, mutation_rate, lower_bound, upper_bound))
            else:
                next_population.append(mutation(parent1, mutation_rate, lower_bound, upper_bound))
                if len(next_population) < population_size:
                    next_population.append(mutation(parent2, mutation_rate, lower_bound, upper_bound))
        
        population = next_population

    print(table)
    
    if plot_all:
        # Assuming you've defined plot_all_results in another file
        from plot_functions import plot_all_results
        plot_all_results(best_performers, all_populations, generations, lower_bound, upper_bound, fitness_function)

    return max(population, key=fitness_function)