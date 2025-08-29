import random
from models.solution import Solution
from models.pareto_wall import ParetoWall
from functions.evaluation import calculate_all_metrics, get_total_tool_switches, get_total_flowtime, get_system_makespan
from functions.local_search import variable_neighborhood_descent
from functions.neighborhoods import perturbation_insertion

def iterated_local_search(instance, max_iterations=1000, initial_pop_size=100, archive_size=10, perturbation_strength=2, obj_x="tool_switches", obj_y="makespan"):
    """
    Executa a meta-heurística Iterated Local Search (ILS) multiobjetivo.

    Args:
        instance (Instance): A instância do problema base.
        max_iterations (int): O número de iterações do loop principal do ILS.
        initial_pop_size (int): O tamanho da população inicial para gerar a primeira fronteira.
        archive_size (int): O tamanho máximo do arquivo de Pareto.
        perturbation_strength (int): O "tamanho" da perturbação a ser aplicada.

    Returns:
        ParetoWall: O arquivo final contendo as melhores soluções não-dominadas encontradas.
    """
    print("Iniciando o Iterated Local Search (ILS)...")
    
    # --- PASSO 1: INICIALIZAÇÃO ---
    print(f"Gerando população inicial de {initial_pop_size} soluções...")
    pareto_archive = ParetoWall(max_size=archive_size, objectives_keys=["tool_switches", "makespan"])

    for i in range(initial_pop_size):
        # Cria uma solução inicial aleatória
        instance.construct_initial_solution()
        
        # Avalia a solução
        calculate_all_metrics(instance)
        objectives = {
            "tool_switches": get_total_tool_switches(instance),
            # "flowtime": get_total_flowtime(instance),
            "makespan": get_system_makespan(instance)
        }
        initial_solution = Solution(instance=instance, solution_id=f"init_{i+1}", objectives=objectives)
        
        # Adiciona ao arquivo, que se gerencia sozinho
        pareto_archive.add(initial_solution)
    
    print(f"Arquivo inicializado com {len(pareto_archive)} soluções não-dominadas.")

    # --- PASSO 2: LOOP PRINCIPAL DO ILS ---
    for i in range(max_iterations):
        print(i)
        if not pareto_archive.get_solutions():
            print("Arquivo de Pareto ficou vazio. Interrompendo o ILS.")
            break
            
        # 1. Seleção: Escolhe uma solução aleatória do arquivo para melhorar
        solution_to_improve = random.choice(pareto_archive.get_solutions())

        # 2. Perturbação: Aplica um "chute" na solução
        perturbed_assignment = perturbation_insertion(solution_to_improve, instance, strength=perturbation_strength)
        
        # Cria e avalia a solução perturbada
        problem_copy_for_vnd = instance.copy_with_new_assignment(perturbed_assignment)
        calculate_all_metrics(problem_copy_for_vnd)
        obj_p = {
            "tool_switches": get_total_tool_switches(problem_copy_for_vnd),
            # "flowtime": get_total_flowtime(problem_copy_for_vnd),
            "makespan": get_system_makespan(problem_copy_for_vnd)
        }
        perturbed_solution = Solution(instance=problem_copy_for_vnd, solution_id=f"iter_{i+1}", objectives=obj_p)

        # 3. ATUALIZAÇÃO COM DIVERSIDADE: Tenta adicionar a solução apenas perturbada
        #    Isso ajuda a espalhar os pontos na fronteira.
        pareto_archive.add(perturbed_solution)

        # 4. BUSCA LOCAL: Agora, aplica o VND na solução perturbada para buscar a convergência.
        locally_optimal_solution = variable_neighborhood_descent(perturbed_solution, instance, obj_x, obj_y)
        
        # 5. ATUALIZAÇÃO COM CONVERGÊNCIA: Tenta adicionar a solução otimizada.
        pareto_archive.add(locally_optimal_solution)
        
        if (i + 1) % 100 == 0:
            print(f"Iteração ILS {i+1}/{max_iterations}... Tamanho do arquivo: {len(pareto_archive)}")

    # --- PASSO 3: RETORNO ---
    print("ILS finalizado.")
    return pareto_archive