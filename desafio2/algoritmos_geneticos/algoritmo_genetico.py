import pygame
import sys
import random
import math
import numpy as np
from typing import List, Tuple

# --- Constantes do Pygame e da Visualização ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PADDING = 50 # Espaçamento das bordas
SCALE_FACTOR = (min(SCREEN_WIDTH, SCREEN_HEIGHT) - 2 * PADDING) / 100.0

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# --- Parâmetros do Algoritmo Genético ---
N_CITIES = 20
POPULATION_SIZE = 100
NUM_GENERATIONS = 500
MUTATION_RATE = 0.02 # Chance de 2% de uma rota sofrer mutação
ELITISM_SIZE = 2 # Quantos dos melhores indivíduos passam para a próxima geração

# --- Funções do Algoritmo Genético (versão completa) ---

Point = Tuple[int, int]
Path = List[Point]

def calculate_distance(p1: Point, p2: Point) -> float:
    """Calcula a distância euclidiana entre dois pontos."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_fitness(path: Path) -> float:
    """Calcula a distância total de uma rota (fitness). Menor é melhor."""
    distance = 0
    for i in range(len(path)):
        distance += calculate_distance(path[i], path[(i + 1) % len(path)])
    return distance

def crossover(parent1: Path, parent2: Path) -> Path:
    """Realiza o crossover ordenado (OX1) para criar um filho."""
    child = [None] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    
    # Copia o segmento do pai 1
    child[start:end] = parent1[start:end]
    
    # Preenche o restante com os genes do pai 2
    parent2_genes = [gene for gene in parent2 if gene not in child]
    
    current_pos = end
    for gene in parent2_genes:
        if current_pos >= len(child):
            current_pos = 0
        if child[current_pos] is None:
            child[current_pos] = gene
            current_pos += 1

    # Lida com o caso de a parte final já estar preenchida
    idx = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = parent2_genes[idx]
            idx += 1
            
    return child


def mutate(path: Path) -> Path:
    """Troca duas cidades de lugar na rota (mutação de troca)."""
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(path)), 2)
        path[idx1], path[idx2] = path[idx2], path[idx1]
    return path

# --- Funções de Desenho do Pygame ---

def draw_info(screen, font, generation, best_distance):
    """Desenha as informações de geração e distância na tela."""
    gen_text = font.render(f"Geração: {generation}", True, WHITE)
    dist_text = font.render(f"Melhor Distância: {best_distance:.2f}", True, WHITE)
    screen.blit(gen_text, (10, 10))
    screen.blit(dist_text, (10, 40))

def draw_route(screen, route, cities):
    """Desenha as cidades e a melhor rota."""
    # Desenha as cidades
    for city in cities:
        x = int(city[0] * SCALE_FACTOR + PADDING)
        y = int(city[1] * SCALE_FACTOR + PADDING)
        pygame.draw.circle(screen, BLUE, (x, y), 5)

    # Desenha a rota
    if route:
        scaled_route = [(int(p[0] * SCALE_FACTOR + PADDING), int(p[1] * SCALE_FACTOR + PADDING)) for p in route]
        pygame.draw.lines(screen, GREEN, True, scaled_route, 2)
        # Destaca a cidade inicial
        pygame.draw.circle(screen, RED, scaled_route[0], 7)


# --- Loop Principal ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Visualizador de Algoritmo Genético - PCV")
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # 1. Inicialização
    cities = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(N_CITIES)]
    population = [random.sample(cities, len(cities)) for _ in range(POPULATION_SIZE)]
    
    best_route_so_far = None
    best_distance_so_far = float('inf')

    # Loop das gerações
    for generation in range(NUM_GENERATIONS):
        # Checa eventos (ex: fechar a janela)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Avaliação (Fitness)
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # 3. Seleção e Evolução
        new_population = []
        
        # Elitismo: os melhores indivíduos passam direto
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
        new_population.extend(sorted_population[:ELITISM_SIZE])

        # Atualiza a melhor rota encontrada até agora
        if fitness_scores[0] < best_distance_so_far:
            best_distance_so_far = fitness_scores[0]
            best_route_so_far = sorted_population[0]

        # Inverte o fitness para a seleção (maior é melhor)
        selection_probs = 1 / np.array(fitness_scores)
        selection_probs /= np.sum(selection_probs) # Normaliza

        # Gera o resto da nova população
        while len(new_population) < POPULATION_SIZE:
            # Seleciona dois pais
            indices = np.random.choice(len(population), 2, p=selection_probs, replace=False)
            parent1, parent2 = population[indices[0]], population[indices[1]]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutação
            child = mutate(child)
            
            new_population.append(child)

        population = new_population

        # 4. Desenho
        screen.fill(BLACK)
        draw_info(screen, font, generation + 1, best_distance_so_far)
        draw_route(screen, best_route_so_far, cities)
        pygame.display.flip()
        
        # Controla a velocidade da visualização
        clock.tick(10) 

    print(f"Finalizado! Melhor distância encontrada: {best_distance_so_far:.2f}")

    # Mantém a janela aberta no final para ver o resultado
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill(BLACK)
        draw_info(screen, font, f"Finalizado na Geração {NUM_GENERATIONS}", best_distance_so_far)
        draw_route(screen, best_route_so_far, cities)
        pygame.display.flip()


if __name__ == '__main__':
    main()