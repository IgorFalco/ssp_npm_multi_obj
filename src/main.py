import os

from models.pareto_wall import plot_combined_pareto
from functions.input import read_problem_instance
from functions.metaheuristics import iterated_local_search  # <-- A grande chamada

# --- CONFIGURAÇÃO ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_FILEPATH = os.path.join(BASE_DIR, "instances/SSP-NPM-I")
RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_FILEPATH, exist_ok=True)

num_runs = 10
all_final_paretos = []  # Lista para guardar o resultado de cada uma das 10 execuções
instance_filename = "ins160_m=3_j=20_t=20_var=20.csv"

# --- EXECUÇÃO PRINCIPAL ---

# 1. Lê a instância base UMA VEZ
print(f"Lendo dados da instância: {instance_filename}")
problem_instance = read_problem_instance(INSTANCE_FILEPATH, instance_filename)

for run_number in range(1, num_runs + 1):
    print(f"\n--- INICIANDO EXECUÇÃO {run_number}/{num_runs} ---")

    final_pareto_front = []
    # Roda a meta-heurística completa para esta execução
    final_pareto_front = iterated_local_search(
        instance=problem_instance,
        max_iterations=100,
        initial_pop_size=50,
        archive_size=10,
        perturbation_strength=2
    )

    print(
        f"Execução {run_number} finalizada. Fronteira contém {len(final_pareto_front)} soluções.")

    # Adiciona a fronteira final desta execução à lista de resultados
    all_final_paretos.append(final_pareto_front)

    # --- SALVAMENTO E PLOTAGEM INDIVIDUAL ---
    # Cria nomes de arquivo únicos para esta execução
    pareto_csv_path = os.path.join(
        RESULTS_FILEPATH, f"pareto_wall_run_{run_number}.csv")
    plot_image_path = os.path.join(
        RESULTS_FILEPATH, f"pareto_plot_run_{run_number}.png")

    # Salva e plota o resultado individual desta execução
    final_pareto_front.save_to_csv(pareto_csv_path)
    final_pareto_front.plot(save_path=plot_image_path)

# PASSO 3: PLOTAGEM COMBINADA
print("\n--- GERANDO GRÁFICO COMBINADO ---")

combined_plot_path = os.path.join( RESULTS_FILEPATH, "pareto_front_combined.png")

plot_combined_pareto(
    all_final_paretos,
    save_path=combined_plot_path
)
