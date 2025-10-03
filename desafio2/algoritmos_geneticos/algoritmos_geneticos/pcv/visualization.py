import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

def draw_cities(screen, cities, color, radius, offset=0):
    for city in cities:
        pygame.draw.circle(screen, color, (city[0] + offset, city[1]), radius)

def draw_paths(screen, path, cities, color, width=1, offset=0):
    n = len(path)
    for i in range(n):
        city1_index = path[i]
        city2_index = path[(i + 1) % n]
        city1 = cities[city1_index]
        city2 = cities[city2_index]
        pygame.draw.line(screen, color, (city1[0] + offset, city1[1]), (city2[0] + offset, city2[1]), width)

def draw_plot(screen, x_values, y_values, y_label="Fitness", x_offset=0, plot_width=300, plot_height=200):
    if not x_values or not y_values:
        return

    min_y = min(y_values)
    max_y = max(y_values)
    range_y = max_y - min_y

    plot_start_x = 20 + x_offset
    plot_start_y = screen.get_height() - plot_height - 20

    pygame.draw.rect(screen, BLACK, (plot_start_x - 5, plot_start_y - 20, plot_width + 10, plot_height + 25), 1)

    for i in range(len(x_values) - 1):
        x1 = plot_start_x + (x_values[i] / max(1, len(x_values) - 1)) * plot_width
        y1 = plot_start_y + plot_height - ((y_values[i] - min_y) / range_y if range_y > 0 else 0) * plot_height
        x2 = plot_start_x + (x_values[i + 1] / max(1, len(x_values) - 1)) * plot_width
        y2 = plot_start_y + plot_height - ((y_values[i + 1] - min_y) / range_y if range_y > 0 else 0) * plot_height
        pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2), 2)
        
    # Desenhar rótulo y
    font = pygame.font.Font(None, 24)
    text = font.render(y_label, True, BLACK)
    text_rect = text.get_rect(midleft=(plot_start_x - 50, plot_start_y + plot_height // 2))
    screen.blit(text, text_rect)

    # Desenhar rótulo x
    text_x = font.render("Geração", True, BLACK)
    text_rect_x = text_x.get_rect(midbottom=(plot_start_x + plot_width // 2, screen.get_height() - 5))
    screen.blit(text_x, text_rect_x)