# ga_logic.py

import random
import numpy as np

def create_initial_population(n_cities, pop_size):
    """Cria a população inicial de rotas aleatórias."""
    population = []
    for _ in range(pop_size):
        individual = list(range(n_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_distance(city1, city2):
    """Calcula a distância euclidiana entre duas cidades."""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_total_distance(path, cities):
    """Calcula o comprimento total de uma rota."""
    distance = 0
    for i in range(len(path)):
        city1_index = path[i]
        city2_index = path[(i + 1) % len(path)]
        distance += calculate_distance(cities[city1_index], cities[city2_index])
    return distance

def calculate_fitness(path, cities):
    """Calcula a aptidão de uma rota (inverso da distância total)."""
    distance = calculate_total_distance(path, cities)
    return 1 / (distance + 1e-10)

def order_crossover(parent1, parent2):
    """Realiza o crossover de ordem."""
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end+1] = parent1[start:end+1]
    
    current_index = 0
    for gene in parent2:
        if gene not in child:
            while child[current_index] != -1:
                current_index = (current_index + 1) % size
            child[current_index] = gene
            
    return child

def swap_mutation(individual, mutation_prob):
    """Aplica mutação por troca de genes."""
    mutated_individual = list(individual)
    if random.random() < mutation_prob:
        idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
    return tuple(mutated_individual)

# Adicione esta função auxiliar dentro de run_simulation():
def select_parent_by_tournament(population, population_fitness, k):
    """Seleciona um pai usando o método de Seleção por Torneio (k participantes)."""
    
    # 1. Seleciona K índices aleatórios de toda a população
    participants_indices = random.sample(range(len(population)), k)
    
    best_fitness_found = -1 # Fitness é sempre positivo (inverso da distância)
    winner_index = -1
    
    # 2. Encontra o vencedor (o indivíduo com o maior fitness)
    for index in participants_indices:
        current_fitness = population_fitness[index]
        if current_fitness > best_fitness_found:
            best_fitness_found = current_fitness
            winner_index = index
            
    # Retorna o indivíduo (rota) vencedor
    return population[winner_index]

# ...
# O resto do seu código run_simulation
# ...