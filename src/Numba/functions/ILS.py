

from random import random
import numpy as np
from models.pareto_wall import ParetoWall
from functions.metaheuristics import calculate_flowtime_all_machines, calculate_makespan_all_machines, construct_initial_solution, get_system_makespan, get_total_flowtime
from models.solution import Solution

def iterated_local_search(instance, tools_per_job, max_iterations=100, initial_pop_size=100, archive_size=10, perturbation_strength=2, obj_x="tool_switches", obj_y="makespan"):

    print("Iniciando o Iterated Local Search (ILS)...")

    # --- PASSO 1: INICIALIZAÇÃO ---
    print(f"Gerando população inicial de {initial_pop_size} soluções...")
    pareto_archive = ParetoWall(max_size=archive_size, objectives_keys=[
                                "tool_switches", "makespan"])

    for i in range(initial_pop_size):
        job_assignment = construct_initial_solution(
            instance["num_jobs"],
            instance["num_machines"],
            instance["magazines_capacities"],
            tools_per_job,
            instance["tools_requirements_matrix"]
        )

        initial_solution = Solution(job_assignment, instance,solution_id=f"init_{i+1}")
        pareto_archive.add(initial_solution)

    print(f"Arquivo inicializado com {len(pareto_archive)} soluções não-dominadas.")

    for i in range(max_iterations):
        print(f"Iteração {i+1}/{max_iterations}...")

        if not pareto_archive.get_solutions():
            print("Nenhuma solução não-dominada encontrada. Encerrando...")
            break

        solution_to_improve = random.choice(pareto_archive.get_solutions())
