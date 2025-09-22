import os
import time
from models.pareto_wall import plot_combined_pareto
from functions.ILS import iterated_local_search_numba
from functions.input import read_problem_instance

BASE_DIR = os.path.dirname(__file__)
RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-I")
SSP_NPM_II_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-II")

instance_filename = "ins640_m=6_j=120_t=120_sw=h_dens=d_var=20.csv"
num_runs = 10

start_time = time.time()

print(f"Lendo dados da instância: {instance_filename}")
instance = read_problem_instance(os.path.join(SSP_NPM_II_PATH, instance_filename))
all_final_paretos = []

for i in range(num_runs):
    print(f"\n--- INICIANDO EXECUÇÃO {i+1}/10 ---")

    final_pareto_front = iterated_local_search_numba(
        instance,
        max_iterations=1000,
        initial_pop_size=50,
        archive_size=10,
        perturbation_strength=2
    )

    print(
        f"Execução {i+1} finalizada. Fronteira contém {len(final_pareto_front)} soluções.")
    
    all_final_paretos.append(final_pareto_front)

    pareto_csv_path = os.path.join(
        RESULTS_FILEPATH, f"pareto_wall_run_{i+1}.csv")
    plot_image_path = os.path.join(
        RESULTS_FILEPATH, f"pareto_plot_run_{i+1}.png")

    # Salva e plota o resultado individual desta execução
    final_pareto_front.plot(save_path=plot_image_path)

print("\n--- GERANDO GRÁFICO COMBINADO ---")

combined_plot_path = os.path.join( RESULTS_FILEPATH, "pareto_front_combined.png")

plot_combined_pareto(
    all_final_paretos,
    save_path=combined_plot_path
)

print("Tempo de execução: ", time.time() - start_time)