import numpy as np
from numba import njit
from models.solution import Solution
from functions.neighborhoods import (
    generate_job_exchange_neighbors_numba,
    generate_swap_neighbors_numba,
    generate_two_opt_neighbors_numba,
    generate_one_block_neighbors_numba,
    generate_insertion_neighbors_numba
)

def variable_neighborhood_descent_numba(initial_solution, obj_x="tool_switches", obj_y="makespan"):
    """
    Aplica VND (Variable Neighborhood Descent) a uma solução para encontrar ótimo local
    
    Args:
        initial_solution (Solution): Solução inicial
        obj_x (str): Nome do primeiro objetivo para dominância
        obj_y (str): Nome do segundo objetivo para dominância
        
    Returns:
        Solution: Solução localmente otimizada
    """

    # Define ordem das vizinhanças a serem exploradas
    neighborhoods = [
        ("job_exchange", generate_job_exchange_neighbors_numba),
        ("swap", generate_swap_neighbors_numba),
        ("two_opt", generate_two_opt_neighbors_numba),
        ("one_block", generate_one_block_neighbors_numba),
        ("insertion", generate_insertion_neighbors_numba)
    ]
    
    current_best_solution = initial_solution
    current_objectives = np.array([
        current_best_solution.objectives["tool_switches"],
        current_best_solution.objectives["makespan"],
        current_best_solution.objectives["flowtime"]
    ])
    
    k = 0
    while k < len(neighborhoods):
        neighborhood_name, neighborhood_generator = neighborhoods[k]
        improvement_found = False
        
        # Determina quais parâmetros passar para o gerador
        if neighborhood_name == "job_exchange":
            neighbors_iter = neighborhood_generator(
                current_best_solution.job_assignment,
                current_best_solution.instance_data["magazines_capacities"],
                current_best_solution.instance_data["tools_per_job"]
            )
        elif neighborhood_name == "swap":
            neighbors_iter = neighborhood_generator(
                current_best_solution.job_assignment
            )
        elif neighborhood_name == "two_opt":
            neighbors_iter = neighborhood_generator(
                current_best_solution.job_assignment
            )
        elif neighborhood_name == "one_block":
            neighbors_iter = neighborhood_generator(
                current_best_solution.job_assignment,
                current_best_solution.instance_data["tools_requirements_matrix"]
            )
        elif neighborhood_name == "insertion":
            neighbors_iter = neighborhood_generator(
                current_best_solution.job_assignment,
                current_best_solution.instance_data["magazines_capacities"],
                current_best_solution.instance_data["tools_per_job"]
            )
        
        # Avalia cada vizinho
        for neighbor_assignment in neighbors_iter:

            # Cria nova solução com o assignment do vizinho
            neighbor_solution = Solution(
                neighbor_assignment,
                current_best_solution.instance_data,
                current_best_solution.solution_id + f"_vnd_{neighborhood_name}"
            )
            
            neighbor_objectives = np.array([
                neighbor_solution.objectives["tool_switches"],
                neighbor_solution.objectives["makespan"], 
                neighbor_solution.objectives["flowtime"]
            ])
            
            # Verifica se há diferença nos objetivos
            objectives_different = not np.array_equal(neighbor_objectives, current_objectives)
            
            # Critério de aceitação: dominância de Pareto nos objetivos especificados
            if objectives_different and neighbor_solution.dominates_on_axes(current_best_solution, obj_x, obj_y):
                current_best_solution = neighbor_solution
                current_objectives = neighbor_objectives
                improvement_found = True
                k = 0  # Reinicia da primeira vizinhança
                break
        
        if not improvement_found:
            k += 1  # Passa para próxima vizinhança
    
    return current_best_solution