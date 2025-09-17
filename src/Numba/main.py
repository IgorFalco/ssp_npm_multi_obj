import os
import numpy as np
from functions.ILS import iterated_local_search, multi_start_ils
from functions.input import read_problem_instance

BASE_DIR = os.path.dirname(__file__)
SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-I")
SSP_NPM_II_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-II")

# Teste com instância pequena primeiro
print("=== TESTE COM INSTÂNCIA PEQUENA ===")
small_instance = read_problem_instance(os.path.join(SSP_NPM_I_PATH, "ins1_m=2_j=10_t=10_var=1.csv"))

print(f"Instância: {small_instance['num_machines']} máquinas, {small_instance['num_jobs']} jobs")

# Executa ILS
result_archive = iterated_local_search(
    small_instance,
    max_iterations=20,
    initial_pop_size=10,
    archive_size=5,
    perturbation_strength=2
)

print(f"\nResultados finais:")
print(f"Soluções não-dominadas encontradas: {len(result_archive.get_solutions())}")

for i, solution in enumerate(result_archive.get_solutions()):
    print(f"Solução {i+1}: Tool switches={solution.objectives['tool_switches']}, "
          f"Makespan={solution.objectives['makespan']}, "
          f"Flowtime={solution.objectives['flowtime']}")

print("\n" + "="*50)

# Teste com instância maior
print("=== TESTE COM INSTÂNCIA MAIOR ===")
large_instance = read_problem_instance(os.path.join(SSP_NPM_II_PATH, "ins640_m=6_j=120_t=120_sw=h_dens=d_var=20.csv"))

print(f"Instância: {large_instance['num_machines']} máquinas, {large_instance['num_jobs']} jobs")

# Executa Multi-start ILS
final_archive = multi_start_ils(
    large_instance,
    num_runs=3,
    max_iterations=50,
    initial_pop_size=30,
    archive_size=15,
    perturbation_strength=3
)

print(f"\nResultados Multi-start:")
print(f"Soluções não-dominadas encontradas: {len(final_archive.get_solutions())}")

# Salva resultados
if len(final_archive.get_solutions()) > 0:
    best_tool_switches = min(s.objectives["tool_switches"] for s in final_archive.get_solutions())
    best_makespan = min(s.objectives["makespan"] for s in final_archive.get_solutions())
    best_flowtime = min(s.objectives["flowtime"] for s in final_archive.get_solutions())
    
    print(f"Melhores objetivos encontrados:")
    print(f"  Tool switches: {best_tool_switches}")
    print(f"  Makespan: {best_makespan}")
    print(f"  Flowtime: {best_flowtime}")
