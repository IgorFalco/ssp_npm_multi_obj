import os
from models.pareto_wall import ParetoWall
from functions.input import read_problem_instance
from functions.metaheuristics import iterated_local_search # <-- A grande chamada

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FILEPATH = os.path.join(BASE_DIR, "instances/SSP-NPM-II")
RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_FILEPATH, exist_ok=True)

# --- EXECUÇÃO PRINCIPAL ---

# 1. Lê a instância base UMA VEZ
problem_instance = read_problem_instance(INSTANCE_FILEPATH, "ins480_m=4_j=60_t=120_sw=h_dens=d_var=20.csv")

# 2. Roda a meta-heurística completa
final_pareto_front = iterated_local_search(
    instance=problem_instance,
    max_iterations=1000,
    initial_pop_size=50,
    archive_size=10,
    perturbation_strength=2
)

# 3. Salva e plota o resultado final
print(f"\nProcesso finalizado. Fronteira de Pareto final contém {len(final_pareto_front)} soluções.")

pareto_csv_path = os.path.join(RESULTS_FILEPATH, "pareto_wall.csv")
plot_image_path = os.path.join(RESULTS_FILEPATH, "pareto_wall_plot.png")

final_pareto_front.save_to_csv(pareto_csv_path)
final_pareto_front.plot(save_path=plot_image_path)