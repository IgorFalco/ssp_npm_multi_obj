import os

# Supondo que suas classes e funções estão organizadas em pastas
from models.solution import Solution
from models.pareto_wall import ParetoWall
from functions.evaluation import (
    calculate_all_metrics, 
    get_total_tool_switches, 
    get_total_flowtime, 
    get_system_makespan
)
from functions.input import (
    read_problem_instance
)

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FILEPATH = os.path.join(BASE_DIR, "instances/SSP-NPM-I")
RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_FILEPATH, exist_ok=True)

pareto_csv_path = os.path.join(RESULTS_FILEPATH, "pareto_wall.csv")
plot_image_path = os.path.join(RESULTS_FILEPATH, "pareto_wall_plot.png")

pareto_wall = ParetoWall(objectives_keys=["makespan", "flowtime"])

for i in range(1000):

    problem = read_problem_instance(INSTANCE_FILEPATH, "ins1_m=2_j=10_t=10_var=1.csv")
    calculate_all_metrics(problem)
    objectives = {
        "tool_switches": get_total_tool_switches(problem),
        "flowtime": get_total_flowtime(problem),
        "makespan": get_system_makespan(problem)
    }

    new_solution = Solution(
        instance=problem, 
        solution_id=i + 1, 
        objectives=objectives
    )

    pareto_wall.add(new_solution)

print(f"\nProcesso finalizado. Fronteira de Pareto final contém {len(pareto_wall)} soluções.")
pareto_wall.save_to_csv(pareto_csv_path)
pareto_wall.plot(save_path=plot_image_path)