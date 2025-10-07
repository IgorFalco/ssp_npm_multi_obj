import os
import sys
import time

from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(__file__)
NUMBA_PATH = os.path.join(BASE_DIR, "../../src/Numba")
sys.path.insert(0, NUMBA_PATH)

from models.pareto_wall import plot_combined_pareto
from functions.ILS import iterated_local_search_numba
from functions.input import read_problem_instance

BASE_DIR = os.path.dirname(__file__)
SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../../src/instances/SSP-NPM-I")

files = os.listdir(SSP_NPM_I_PATH)
files = sorted(files, key=lambda x: int(x.split('_')[0][3:]) if x.startswith('ins') else 0)

for file in files:

    print(f"\n{'='*60}")
    print(f"INICIANDO NOVA INSTÂNCIA: {file}")
    print(f"{'='*60}")

    file_base = os.path.splitext(file)[0]
    RESULTS_FILEPATH = os.path.join(BASE_DIR, f"results_vns/{file_base}")
    
    os.makedirs(RESULTS_FILEPATH, exist_ok=True)

    num_runs = 30

    print(f"Lendo dados da instância: {file}")
    instance = read_problem_instance(os.path.join(SSP_NPM_I_PATH, file))
    iterations = 50*instance["num_machines"]*instance["num_jobs"]


    all_final_paretos = []
    instance_start_time = time.time()

    for i in range(num_runs):
        print(f"\n--- INICIANDO EXECUÇÃO {i+1}/{num_runs} ---")

        run_start_time = time.time()

        final_pareto_front = iterated_local_search_numba(
            instance,
            max_iterations=iterations,
            initial_pop_size=50,
            archive_size=10,
            perturbation_strength=2
        )

        run_time = time.time() - run_start_time

        print(f"Execução {i+1} finalizada em {run_time:.2f}s. Fronteira contém {len(final_pareto_front)} soluções.")
        
        all_final_paretos.append(final_pareto_front)

        pareto_csv_path = os.path.join(
            RESULTS_FILEPATH, f"pareto_wall_run_{i+1}.csv")
        plot_image_path = os.path.join(
            RESULTS_FILEPATH, f"pareto_plot_run_{i+1}.png")

        # LIMPA O MATPLOTLIB ANTES DE PLOTAR
        plt.clf()
        plt.cla() 
        plt.close('all')

        # Salva e plota o resultado individual desta execução
        final_pareto_front.plot(save_path=plot_image_path)
        final_pareto_front.save_to_csv(
            instance_name=file, 
            filepath=pareto_csv_path, 
            iterations=iterations, 
            execution_time=run_time
        )

    print("\n--- GERANDO GRÁFICO COMBINADO ---")

    combined_plot_path = os.path.join(RESULTS_FILEPATH, "pareto_front_combined.png")

    plot_combined_pareto(
        all_final_paretos,
        save_path=combined_plot_path
    )

    total_time = time.time() - instance_start_time
    print(f"Tempo total para {file}: {total_time:.2f}s")
    print(f"{'='*60}")