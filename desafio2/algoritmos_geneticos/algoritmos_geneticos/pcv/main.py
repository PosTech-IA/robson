# genetic_algorithm_tsp.py

import itertools
import numpy as np
import pygame
import random
import sys
import matplotlib.pyplot as plt
from pygame.locals import *

# --- Parâmetros de Visualização e Simulação ---
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 8
FPS = 30
TSP_DISPLAY_OFFSET = 50

N_CITIES = 30
POPULATION_SIZE = 100
N_GENERATIONS = 500
MUTATION_PROBABILITY = 0.3
CROSSOVER_PROBABILITY = 0.8

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# --- Funções Utilitárias ---
def calculate_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_total_distance(path, cities):
    distance = 0
    for i in range(len(path)):
        city1_index = path[i]
        city2_index = path[(i + 1) % len(path)]
        distance += calculate_distance(cities[city1_index], cities[city2_index])
    return distance

def create_initial_population(n_cities, pop_size):
    population = []
    for _ in range(pop_size):
        individual = list(range(n_cities))
        random.shuffle(individual)
        population.append(individual)
    return population

def calculate_fitness(path, cities):
    distance = calculate_total_distance(path, cities)
    return 1 / (distance + 1e-10)

def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    sublist1 = parent1[start:end+1]
    child[start:end+1] = sublist1
    
    current_index = 0
    for gene in parent2:
        if gene not in sublist1:
            while child[current_index] != -1:
                current_index = (current_index + 1) % size
            child[current_index] = gene
            
    return child

def swap_mutation(individual, mutation_prob):
    mutated_individual = list(individual)
    if random.random() < mutation_prob:
        idx1, idx2 = random.sample(range(len(mutated_individual)), 2)
        mutated_individual[idx1], mutated_individual[idx2] = mutated_individual[idx2], mutated_individual[idx1]
    return tuple(mutated_individual)

# --- Funções de Desenho ---
def draw_cities(screen, cities, color, radius):
    for city in cities:
        pygame.draw.circle(screen, color, city, radius)

def draw_paths(screen, path, cities, color, width=1):
    n = len(path)
    for i in range(n):
        city1_index = path[i]
        city2_index = path[(i + 1) % n]
        city1 = cities[city1_index]
        city2 = cities[city2_index]
        pygame.draw.line(screen, color, city1, city2, width)

def draw_text(screen, text, x, y, color):
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

# --- Inicialização Pygame ---
pygame.init()
pygame.font.init() # Inicializa o módulo de fontes
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Otimização de Rotas com Algoritmo Genético")
clock = pygame.time.Clock()

# --- Criação das Cidades ---
cities_locations = [(random.randint(TSP_DISPLAY_OFFSET, WIDTH - TSP_DISPLAY_OFFSET),
                     random.randint(TSP_DISPLAY_OFFSET, HEIGHT - TSP_DISPLAY_OFFSET))
                    for _ in range(N_CITIES)]

# --- Inicialização da População ---
population = create_initial_population(N_CITIES, POPULATION_SIZE)
best_fitness_history = []
generation = 0

# --- Configuração do Matplotlib ---
plt.ion() # Modo interativo para que o gráfico não bloqueie o programa
fig, ax = plt.subplots()
ax.set_title("Evolução da Aptidão por Geração")
ax.set_xlabel("Geração")
ax.set_ylabel("Aptidão (1/Distância)")

# --- Loop Principal ---
running = True
while running:
    # --- Verificação de Eventos ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            if event.key == pygame.K_ESCAPE:
                running = False

    if generation >= N_GENERATIONS:
        # Se as gerações terminarem, o loop continua só para esperar por um evento de saída
        continue

    # --- Lógica do Algoritmo Genético ---
    generation += 1
    population_fitness = [calculate_fitness(ind, cities_locations) for ind in population]
    sorted_population = [population[i] for i in np.argsort(population_fitness)[::-1]]
    best_individual = sorted_population[0]
    best_fitness = calculate_fitness(best_individual, cities_locations)
    best_fitness_history.append(best_fitness)

    # --- Atualizar Gráfico do Matplotlib ---
    ax.clear()
    ax.set_title(f"Evolução da Aptidão (Geração {generation}/{N_GENERATIONS})")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Aptidão (1/Distância)")
    ax.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, color='blue')
    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- Desenhar na Tela do Pygame ---
    screen.fill(WHITE)
    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_individual, cities_locations, BLUE, 2)
    
    # Desenhar algumas outras rotas da população para diversidade
    num_to_draw = min(5, POPULATION_SIZE)
    for i in range(1, num_to_draw):
        if random.random() < 0.3:
            draw_paths(screen, sorted_population[i], cities_locations, GRAY, 1)

    # Exibir a geração atual na tela do Pygame
    draw_text(screen, f"Geração: {generation}/{N_GENERATIONS}", 10, 10, BLACK)

    pygame.display.flip()

    # --- Próxima Geração (Criação de Descendentes) ---
    next_population = [list(best_individual)]
    while len(next_population) < POPULATION_SIZE:
        tournament = random.sample(sorted_population[:20], 5)
        parent1 = max(tournament, key=lambda ind: calculate_fitness(ind, cities_locations))
        tournament = random.sample(sorted_population[:20], 5)
        parent2 = max(tournament, key=lambda ind: calculate_fitness(ind, cities_locations))
        
        if random.random() < CROSSOVER_PROBABILITY:
            child = order_crossover(list(parent1), list(parent2))
        else:
            child = list(random.choice([list(parent1), list(parent2)]))
            
        child = swap_mutation(tuple(child), MUTATION_PROBABILITY)
        
        if list(child) not in next_population:
            next_population.append(list(child))
    
    population = [tuple(ind) for ind in next_population]
    
# --- Finalização ---
pygame.quit()
sys.exit()