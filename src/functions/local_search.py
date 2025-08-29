from models.solution import Solution
from functions.evaluation import calculate_all_metrics, get_total_tool_switches, get_total_flowtime, get_system_makespan
from functions.neighborhoods import (
    generate_job_exchange_neighbors,
    generate_swap_neighbors,
    generate_two_opt_neighbors,
    generate_one_block_neighbors,
    generate_insertion_neighbors
)

def variable_neighborhood_descent(initial_solution, instance, obj_x, obj_y):
    """
    Aplica uma busca local VND a UMA ÚNICA solução para encontrar seu ótimo local.

    Args:
        initial_solution (Solution): A solução de ponto de partida.
        instance (Instance): A instância do problema.

    Returns:
        Solution: A versão localmente otimizada da solução inicial.
    """
    # Define a ordem das vizinhanças a serem exploradas
    neighborhoods = [
        generate_job_exchange_neighbors,
        generate_swap_neighbors,
        generate_two_opt_neighbors,
        generate_one_block_neighbors,
        generate_insertion_neighbors
    ]
    
    current_best_solution = initial_solution
    
    k = 0
    while k < len(neighborhoods):
        neighborhood_generator = neighborhoods[k]
        improvement_found = False
        
        # Gera e avalia os vizinhos da vizinhança 'k'
        # Passa 'instance' para os geradores que precisam dela
        if neighborhood_generator in [generate_job_exchange_neighbors, generate_one_block_neighbors, generate_insertion_neighbors]:
            neighbors_iter = neighborhood_generator(current_best_solution, instance)
        else:
            neighbors_iter = neighborhood_generator(current_best_solution)

        for neighbor_assignment in neighbors_iter:
            # 1. Cria e avalia a solução vizinha
            problem_copy = instance.copy_with_new_assignment(neighbor_assignment)
            calculate_all_metrics(problem_copy)
            
            objectives = {
                "tool_switches": get_total_tool_switches(problem_copy),
                # "flowtime": get_total_flowtime(problem_copy),
                "makespan": get_system_makespan(problem_copy)
            }
            neighbor_solution = Solution(
                instance=problem_copy, 
                solution_id=current_best_solution.solution_id, 
                objectives=objectives
            )

            is_different = (neighbor_solution.objectives != current_best_solution.objectives)
            
            # 2. Critério de Aceitação: Dominância de Pareto
            if is_different and neighbor_solution.dominates_on_axes(current_best_solution, obj_x, obj_y):
                current_best_solution = neighbor_solution
                improvement_found = True
                k = 0  # Reinicia a busca a partir da primeira vizinhança (k=0)
                break # Para de explorar esta vizinhança e reinicia o 'while'

        if not improvement_found:
            # Se nenhuma melhoria foi encontrada nesta vizinhança, passa para a próxima
            k += 1

    return current_best_solution