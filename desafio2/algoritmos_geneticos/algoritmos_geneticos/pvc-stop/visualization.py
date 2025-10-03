# visualization.py

import pygame
import matplotlib.pyplot as plt
import random
# --- Cores ---
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def setup_pygame_display(width, height):
    """Inicializa e configura a janela do Pygame."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Otimização de Rotas com Algoritmo Genético")
    clock = pygame.time.Clock()
    return screen, clock

def draw_all_elements(screen, best_individual, sorted_population, cities, generation, n_generations):
    """Desenha todos os elementos na tela do Pygame."""
    screen.fill(WHITE)
    
    draw_paths(screen, best_individual, cities, BLUE, 2)
    
    num_to_draw = min(5, len(sorted_population))
    for i in range(1, num_to_draw):
        if random.random() < 0.3:
            draw_paths(screen, sorted_population[i], cities, GRAY, 1)

    for city in cities:
        pygame.draw.circle(screen, RED, city, 8)
        
    draw_text(screen, f"Geração: {generation}/{n_generations}", 10, 10, BLACK)

    pygame.display.flip()

def draw_paths(screen, path, cities, color, width):
    """Desenha a rota conectando as cidades."""
    n = len(path)
    for i in range(n):
        city1_index = path[i]
        city2_index = path[(i + 1) % n]
        city1 = cities[city1_index]
        city2 = cities[city2_index]
        pygame.draw.line(screen, color, city1, city2, width)

def draw_text(screen, text, x, y, color):
    """Renderiza e exibe texto na tela."""
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def update_performance_plots_at_end(best_fitness_history, best_distance_history, avg_distance_history):
    """Cria e exibe os 3 gráficos de performance do Matplotlib ao final da simulação."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gráfico 1: Aptidão
    axes[0].set_title("Aptidão do Melhor Indivíduo")
    axes[0].set_xlabel("Geração")
    axes[0].set_ylabel("Aptidão (1/Distância)")
    axes[0].plot(range(1, len(best_fitness_history) + 1), best_fitness_history, color='blue')
    
    # Gráfico 2: Distância da Melhor Rota
    axes[1].set_title("Distância da Melhor Rota")
    axes[1].set_xlabel("Geração")
    axes[1].set_ylabel("Distância")
    axes[1].plot(range(1, len(best_distance_history) + 1), best_distance_history, color='green')
    
    # Gráfico 3: Distância Média da População
    axes[2].set_title("Distância Média da População")
    axes[2].set_xlabel("Geração")
    axes[2].set_ylabel("Distância")
    axes[2].plot(range(1, len(avg_distance_history) + 1), avg_distance_history, color='red')
    
    plt.tight_layout()
    plt.show()