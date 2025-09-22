
import random
import numpy as np
from models.pareto_wall import ParetoWall
from functions.metaheuristics import construct_initial_solution
from functions.local_search import variable_neighborhood_descent_numba
from functions.neighborhoods import perturbation_insertion_numba
from models.solution import Solution


def iterated_local_search_numba(instance_data, max_iterations=100, initial_pop_size=100, 
                               archive_size=10, perturbation_strength=2, obj_x="tool_switches", obj_y="makespan"):
    """
    Executa a meta-heurística Iterated Local Search (ILS) multiobjetivo otimizada com Numba
    
    Args:
        instance_data (dict): Dados da instância do problema
        max_iterations (int): Número de iterações do loop principal
        initial_pop_size (int): Tamanho da população inicial
        archive_size (int): Tamanho máximo do arquivo de Pareto
        perturbation_strength (int): Força da perturbação aplicada
        obj_x (str): Nome do primeiro objetivo para dominância
        obj_y (str): Nome do segundo objetivo para dominância
        
    Returns:
        ParetoWall: Arquivo final com melhores soluções não-dominadas
    """
    print("Iniciando o Iterated Local Search (ILS) com Numba...")
    
    # --- PASSO 1: INICIALIZAÇÃO ---
    print(f"Gerando população inicial de {initial_pop_size} soluções...")
    pareto_archive = ParetoWall(max_size=archive_size, objectives_keys=["tool_switches", "makespan"])
    
    for i in range(initial_pop_size):
        # Cria solução inicial aleatória
        job_assignment = construct_initial_solution(
            instance_data["num_jobs"],
            instance_data["num_machines"], 
            instance_data["magazines_capacities"],
            instance_data["tools_per_job"],
            instance_data["tools_requirements_matrix"]
        )
        
        # Cria objeto Solution que calcula objetivos automaticamente
        initial_solution = Solution(job_assignment, instance_data, solution_id=f"init_{i+1}")
        
        # Adiciona ao arquivo Pareto
        pareto_archive.add(initial_solution)
    
    print(f"Arquivo inicializado com {len(pareto_archive)} soluções não-dominadas.")
    
    # --- PASSO 2: LOOP PRINCIPAL DO ILS ---
    for iteration in range(max_iterations):
        print(f"Iteração {iteration+1}/{max_iterations}...")
        
        if not pareto_archive.get_solutions():
            print("Arquivo de Pareto ficou vazio. Interrompendo o ILS.")
            break
        
        # 1. Seleção: Escolhe uma solução aleatória do arquivo para melhorar
        solution_to_improve = random.choice(pareto_archive.get_solutions())
        
        # 2. Perturbação: Aplica um "chute" na solução
        perturbed_assignment = perturbation_insertion_numba(
            solution_to_improve.job_assignment,
            instance_data["magazines_capacities"],
            instance_data["tools_per_job"],
            strength=perturbation_strength
        )
        
        # Cria e avalia a solução perturbada
        perturbed_solution = Solution(
            perturbed_assignment, 
            instance_data, 
            solution_id=f"iter_{iteration+1}_perturbed"
        )
        
        # 3. BUSCA LOCAL: Aplica VND na solução perturbada
        locally_optimal_solution = variable_neighborhood_descent_numba(
            perturbed_solution, obj_x, obj_y
        )
        
        # 4. ATUALIZAÇÃO COM CONVERGÊNCIA: Adiciona solução otimizada
        pareto_archive.add(locally_optimal_solution)
        
        # Log de progresso
        if (iteration + 1) % 10 == 0:
            print(f"Iteração ILS {iteration+1}/{max_iterations}... "
                  f"Tamanho do arquivo: {len(pareto_archive)}")
    
    # --- PASSO 3: RETORNO ---
    print("ILS finalizado.")
    print(f"Arquivo final contém {len(pareto_archive)} soluções não-dominadas.")
    
    return pareto_archive


def multi_objective_ils_numba(instance_data, objective_pairs=None, **kwargs):
    """
    Executa ILS para múltiplos pares de objetivos e combina os resultados
    
    Args:
        instance_data (dict): Dados da instância
        objective_pairs (list): Lista de tuplas (obj_x, obj_y) para otimizar
        **kwargs: Parâmetros adicionais para o ILS
        
    Returns:
        ParetoWall: Arquivo combinado com soluções de todos os pares de objetivos
    """
    if objective_pairs is None:
        objective_pairs = [
            ("tool_switches", "makespan"),
            ("tool_switches", "flowtime"),
            ("makespan", "flowtime")
        ]
    
    print(f"Executando ILS multi-objetivo para {len(objective_pairs)} pares de objetivos...")
    
    # Arquivo combinado para todas as soluções
    combined_archive = ParetoWall(
        max_size=kwargs.get('archive_size', 10) * len(objective_pairs),
        objectives_keys=["tool_switches", "makespan", "flowtime"]
    )
    
    for i, (obj_x, obj_y) in enumerate(objective_pairs):
        print(f"\n--- Executando ILS para objetivos: {obj_x} vs {obj_y} ({i+1}/{len(objective_pairs)}) ---")
        
        # Executa ILS para este par de objetivos
        archive = iterated_local_search_numba(
            instance_data, obj_x=obj_x, obj_y=obj_y, **kwargs
        )
        
        # Adiciona todas as soluções ao arquivo combinado
        for solution in archive.get_solutions():
            combined_archive.add(solution)
        
        print(f"Adicionadas {len(archive)} soluções do par {obj_x} vs {obj_y}")
    
    print(f"\nArquivo final combinado contém {len(combined_archive)} soluções não-dominadas.")
    return combined_archive
