# plot_functions.py

import matplotlib.pyplot as plt
import numpy as np

def plot_all_results(best_performers, all_populations, generations, lower_bound, upper_bound, fitness_function):
    """
    Plots all the results from the genetic algorithm, including:
    - Final population distribution of parameters a, b, c.
    - Parameter values over generations.
    - Fitness over generations.
    - Quadratic function graph showing evolution.
    """
    
    # --- Plot 1: Final Generation Population Solutions ---
    fig_pop, axs_pop = plt.subplots(3, 1, figsize=(12, 18))
    
    final_population = all_populations[-1]
    best_individual = best_performers[-1][0]

    axs_pop[0].scatter(range(len(final_population)), [ind[0] for ind in final_population], color='blue', label='a')
    axs_pop[0].scatter([final_population.index(best_individual)], [best_individual[0]], color='cyan', s=100, label='Best Individual a')
    axs_pop[0].set_ylabel('a', color='blue')
    axs_pop[0].legend(loc='upper left')
    
    axs_pop[1].scatter(range(len(final_population)), [ind[1] for ind in final_population], color='green', label='b')
    axs_pop[1].scatter([final_population.index(best_individual)], [best_individual[1]], color='magenta', s=100, label='Best Individual b')
    axs_pop[1].set_ylabel('b', color='green')
    axs_pop[1].legend(loc='upper left')
    
    axs_pop[2].scatter(range(len(final_population)), [ind[2] for ind in final_population], color='red', label='c')
    axs_pop[2].scatter([final_population.index(best_individual)], [best_individual[2]], color='yellow', s=100, label='Best Individual c')
    axs_pop[2].set_ylabel('c', color='red')
    axs_pop[2].set_xlabel('Individual Index')
    axs_pop[2].legend(loc='upper left')
    
    axs_pop[0].set_title(f'Final Generation ({generations}) Population Solutions')

    # --- Plot 2: Parameter Values Over Generations ---
    fig_params, ax_params = plt.subplots()
    generations_list = range(1, len(best_performers) + 1)
    a_values = [ind[0][0] for ind in best_performers]
    b_values = [ind[0][1] for ind in best_performers]
    c_values = [ind[0][2] for ind in best_performers]
    ax_params.plot(generations_list, a_values, label='a', color='blue')
    ax_params.plot(generations_list, b_values, label='b', color='green')
    ax_params.plot(generations_list, c_values, label='c', color='red')
    ax_params.set_xlabel('Generation')
    ax_params.set_ylabel('Parameter Values')
    ax_params.set_title('Parameter Values Over Generations')
    ax_params.legend()

    # --- Plot 3: Fitness Over Generations ---
    fig_fit, ax_fit = plt.subplots()
    best_fitness_values = [fit[1] for fit in best_performers]
    min_fitness_values = [min([fitness_function(ind) for ind in population]) for population in all_populations]
    max_fitness_values = [max([fitness_function(ind) for ind in population]) for population in all_populations]
    ax_fit.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
    ax_fit.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    ax_fit.set_xlabel('Generation')
    ax_fit.set_ylabel('Fitness')
    ax_fit.set_title('Fitness Over Generations')
    ax_fit.legend()

    # --- Plot 4: Quadratic Function Evolution ---
    fig_quad, ax_quad = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, generations))
    for i, (best_ind, best_fit) in enumerate(best_performers):
        color = colors[i]
        a, b, c = best_ind
        x_range = np.linspace(lower_bound, upper_bound, 400)
        y_values = a * (x_range ** 2) + b * x_range + c
        ax_quad.plot(x_range, y_values, color=color)

    ax_quad.set_xlabel('x')
    ax_quad.set_ylabel('y')
    ax_quad.set_title('Quadratic Function')

    cax = fig_quad.add_axes([0.92, 0.2, 0.02, 0.6])
    norm = plt.cm.colors.Normalize(vmin=0, vmax=generations)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig_quad.colorbar(sm, ax=ax_quad, cax=cax, orientation='vertical', label='Generation')

    plt.show()