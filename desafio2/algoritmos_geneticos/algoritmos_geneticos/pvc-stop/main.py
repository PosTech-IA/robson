# main.py

import itertools
import pygame
import sys
import matplotlib.pyplot as plt
import random
import numpy as np

# Importar as funções dos módulos
from ga_logic import create_initial_population, calculate_fitness, calculate_total_distance, order_crossover, swap_mutation
from visualization import setup_pygame_display, draw_all_elements, update_performance_plots_at_end

# --- Parâmetros ---
WIDTH, HEIGHT = 1200, 1000
N_CITIES = 70
POPULATION_SIZE = 1000
N_GENERATIONS = 800
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.95
CONVERGENCE_GENERATIONS = 100
TSP_DISPLAY_OFFSET = 60
TOURNAMENT_SIZE = 5 
def run_simulation():
    # Inicialização
    screen, clock = setup_pygame_display(WIDTH, HEIGHT)
    cities_locations = [(random.randint(TSP_DISPLAY_OFFSET, WIDTH - TSP_DISPLAY_OFFSET),
                         random.randint(TSP_DISPLAY_OFFSET, HEIGHT - TSP_DISPLAY_OFFSET))
                        for _ in range(N_CITIES)]
    
    population = create_initial_population(N_CITIES, POPULATION_SIZE)
    
    # Listas para armazenar dados de performance
    best_fitness_history = []
    best_distance_history = []
    avg_distance_history = []
    
    generation = 0
    
    # Loop Principal da Simulação
    running_simulation = True
    while running_simulation:
        # Verificação de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_simulation = False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_q or event.key == pygame.K_ESCAPE):
                running_simulation = False

        if generation >= N_GENERATIONS:
            running_simulation = False
            continue
        
        generation += 1

        # Avaliação da população e verificação de convergência
        population_fitness = [calculate_fitness(ind, cities_locations) for ind in population]
        population_distances = [calculate_total_distance(ind, cities_locations) for ind in population]
        
        sorted_population = sorted(population, key=lambda ind: calculate_fitness(ind, cities_locations), reverse=True)
        best_individual = sorted_population[0]
        
        best_fitness = calculate_fitness(best_individual, cities_locations)
        best_distance = calculate_total_distance(best_individual, cities_locations)
        avg_distance = np.mean(population_distances)
        
        best_fitness_history.append(best_fitness)
        best_distance_history.append(best_distance)
        avg_distance_history.append(avg_distance)

        if generation > CONVERGENCE_GENERATIONS:
            if abs(best_distance_history[generation - 1] - best_distance_history[generation - 1 - CONVERGENCE_GENERATIONS]) < 1e-6:
                print(f"Convergência detectada na Geração {generation}. Parando a simulação.")
                running_simulation = False
                    
        # Atualiza apenas a visualização do Pygame
        draw_all_elements(screen, best_individual, sorted_population, cities_locations, generation, N_GENERATIONS)
        
        # Próxima Geração
        next_population = [list(best_individual)]
        while len(next_population) < POPULATION_SIZE:
            parent1 = random.choice(sorted_population[:POPULATION_SIZE//2])
            parent2 = random.choice(sorted_population[:POPULATION_SIZE//2])
            
            if random.random() < CROSSOVER_PROBABILITY:
                child = order_crossover(list(parent1), list(parent2))
            else:
                child = list(random.choice([list(parent1), list(parent2)]))
            
            child = swap_mutation(tuple(child), MUTATION_PROBABILITY)
            
            if list(child) not in next_population:
                next_population.append(list(child))
        
        population = [tuple(ind) for ind in next_population]
        clock.tick(60)

    # **NOVO:** Chamada para a função que plota todos os gráficos de uma vez
    update_performance_plots_at_end(best_fitness_history, best_distance_history, avg_distance_history)

    # Loop de espera para manter a janela aberta após a simulação
    running_display = True
    while running_display:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_display = False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_q or event.key == pygame.K_ESCAPE):
                running_display = False
        
        pygame.display.flip()
        
    # Finalização
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    run_simulation()