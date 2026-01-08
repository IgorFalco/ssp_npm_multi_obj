import os
import sys
import time
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(__file__)
GUROBI_PATH = os.path.join(BASE_DIR, "../../src/Gurobi")
sys.path.insert(0, GUROBI_PATH)

from functions.input import read_problem_instance
from functions.epsilon_constraint import EpsilonConstraintMethod
from models.solution import GurobiSolution, ParetoFront
from models.pareto_wall import plot_pareto_front_2d

BASE_DIR = os.path.dirname(__file__)
SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../../src/instances/SSP-NPM-I")

# Configurações globais
MAX_TIME_PER_PAIR = 3600  # 1 hora por par de objetivos (em segundos)
START_INSTANCE_ID = 55  # Começa a partir da ins51 (inclusive)
OBJECTIVE_PAIRS = [
    ('TS', 'FMAX'),   # Tool Switches vs Makespan  
    ('TS', 'TFT'),    # Tool Switches vs Total Flow Time
]

def solve_instance_gurobi(instance_data, instance_name, max_time_per_pair=3600):
    """
    Resolve uma instância com Gurobi para todos os pares de objetivos dentro do tempo limite.
    
    Args:
        instance_data: Dados da instância
        instance_name: Nome da instância
        max_time_per_pair: Tempo máximo por par de objetivos (segundos) - padrão 1 hora
    
    Returns:
        dict: Resultados para cada par de objetivos
    """
    results = {}
    
    for pair_idx, (obj1, obj2) in enumerate(OBJECTIVE_PAIRS):
        print(f"\n--- RESOLVENDO PAR {pair_idx+1}/2: {obj1} vs {obj2} ---")
        
        pair_start_time = time.time()
        
        try:
            num_pareto_points = 10  # Número máximo de pontos na fronteira
            # Importante: este é o tempo MÁXIMO por resolução do Gurobi,
            # isto é, por ponto gerado (cada epsilon) e também para os extremos.
            time_per_resolution = int(max_time_per_pair)
            
            print(f"   Tempo por resolução (cada ponto epsilon): {time_per_resolution//60}min ({time_per_resolution}s)")
            print(f"   Pontos estimados: {num_pareto_points}")
            
            # Cria método epsilon constraint
            epsilon_method = EpsilonConstraintMethod(instance_data, time_limit=time_per_resolution)
            
            # Gera fronteira de Pareto
            pareto_solutions_data = epsilon_method.generate_pareto_front_fast(
                num_points=num_pareto_points, 
                selected_objectives=[obj1, obj2]
            )
            
            pair_time = time.time() - pair_start_time
            
            # Converte para objetos GurobiSolution
            pareto_solutions = []
            for i, sol_data in enumerate(pareto_solutions_data):
                # Inclui TODOS os objetivos (mesmo os que não foram otimizados neste par)
                all_objectives = {
                    'FMAX': sol_data.get('FMAX', 0),
                    'TFT': sol_data.get('TFT', 0), 
                    'TS': sol_data.get('TS', 0)
                }
                
                # Valida solução (verifica os objetivos do par atual)
                pair_objectives = {obj: all_objectives[obj] for obj in [obj1, obj2]}
                is_valid = all(val > 0 if obj in ['FMAX', 'TFT'] else val >= 0 
                              for obj, val in pair_objectives.items())
                is_valid &= all(val < 1000000 for val in pair_objectives.values())
                
                if is_valid:
                    # Inclui estatísticas do Gurobi nos objetivos
                    all_objectives.update({
                        'total_iterations': sol_data.get('total_iterations', 0),
                        'total_solutions_found': sol_data.get('total_solutions_found', 0)
                    })
                    
                    solution = GurobiSolution(
                        job_assignment=sol_data['job_assignment'],
                        instance_data=instance_data,
                        gurobi_objectives=all_objectives,  # Todos os 3 objetivos
                        solution_id=f"sol_{len(pareto_solutions)+1:02d}",
                        status=sol_data.get('status', 'OPTIMAL'),
                        gap=sol_data.get('gap', 0.0)
                    )
                    pareto_solutions.append(solution)
            
            # Cria fronteira de Pareto
            pareto_front = ParetoFront(pareto_solutions)
            
            results[f"{obj1}_{obj2}"] = {
                'pareto_front': pareto_front,
                'execution_time': pair_time,
                'num_solutions': len(pareto_solutions),
                'objectives': (obj1, obj2)
            }
            
            print(f"   ✅ Concluído em {pair_time:.1f}s - {len(pareto_solutions)} soluções ótimas")
            
            # Mostra resumo das soluções
            for i, sol in enumerate(pareto_solutions[:5]):  # Mostra apenas as 5 primeiras
                obj_str = " | ".join([f"{obj}={sol.objectives[obj]:6.1f}" for obj in [obj1, obj2]])
                print(f"   Sol {i+1:2d}: {obj_str}")
            if len(pareto_solutions) > 5:
                print(f"   ... e mais {len(pareto_solutions)-5} soluções")
                
        except Exception as e:
            print(f"   ❌ Erro ao resolver par {obj1}-{obj2}: {e}")
            results[f"{obj1}_{obj2}"] = {
                'pareto_front': None,
                'execution_time': time.time() - pair_start_time,
                'num_solutions': 0,
                'objectives': (obj1, obj2),
                'error': str(e)
            }
    
    return results

def save_results(instance_name, results, results_dir):
    """
    Salva os resultados em arquivos CSV e gráficos (formato compatível com VNS).
    """
    print(f"\n--- SALVANDO RESULTADOS ---")
    
    # Coleta todas as soluções de todos os pares para criar fronteira combinada
    all_solutions = []
    total_execution_time = 0
    total_iterations = 0
    
    for pair_key, result in results.items():
        if result['pareto_front'] is None:
            continue
            
        obj1, obj2 = result['objectives']
        total_execution_time += result['execution_time']
        
        # Adiciona soluções à lista geral
        for sol in result['pareto_front'].solutions:
            # Garante que todas as soluções tenham os 3 objetivos (FMAX, TFT, TS)
            full_objectives = {
                'FMAX': sol.objectives.get('FMAX', sol.gurobi_objectives.get('FMAX', 0)),
                'TFT': sol.objectives.get('TFT', sol.gurobi_objectives.get('TFT', 0)), 
                'TS': sol.objectives.get('TS', sol.gurobi_objectives.get('TS', 0))
            }
            
            # Cria nova solução com todos os objetivos
            complete_solution = GurobiSolution(
                job_assignment=sol.job_assignment,
                instance_data=sol.instance_data,
                gurobi_objectives=full_objectives,
                solution_id=sol.solution_id,
                status=sol.status,
                gap=sol.gap
            )
            all_solutions.append(complete_solution)
            
        # Soma iterações totais
        if result['pareto_front'].solutions:
            total_iterations += result['pareto_front'].solutions[0].gurobi_objectives.get('total_iterations', 0)
        
        # Salva gráfico individual por par
        obj_names = {'FMAX': 'Makespan', 'TFT': 'Flow_Time', 'TS': 'Tool_Switches'}
        plot_filename = f"pareto_plot_{obj_names[obj2]}_vs_{obj_names[obj1]}.png"
        plot_path = os.path.join(results_dir, plot_filename)
        
        plot_pareto_front_2d(
            result['pareto_front'], 
            obj1, obj2, 
            save_path=plot_path,
            title=f"Fronteira Ótima: {obj_names[obj2].replace('_', ' ')} vs {obj_names[obj1].replace('_', ' ')}"
        )
        
        print(f"   ✅ {pair_key}: Gráfico salvo ({result['num_solutions']} soluções)")
    
    if all_solutions:
        pareto = ParetoFront(all_solutions)
        
        # Salva CSV combinado (mesmo formato que VNS)
        csv_path = os.path.join(results_dir, "pareto_front.csv")
        pareto.save_to_csv(
            instance_name, 
            csv_path, 
            iterations=total_iterations,  # Iterações totais do Gurobi
            execution_time=total_execution_time  # Tempo total combinado
        )
        
        print(f"   ✅ CSV combinado salvo: {len(all_solutions)} soluções totais")
        print(f"   📊 Formato compatível com VNS para comparação")

def main():
    """
    Executa testes do Gurobi para obter fronteiras de Pareto ótimas.
    """
    print("🚀 TESTE GUROBI - FRONTEIRAS ÓTIMAS")
    print("="*60)
    print(f"Tempo por par de objetivos: {MAX_TIME_PER_PAIR//60} minutos")
    print(f"Pares de objetivos: {len(OBJECTIVE_PAIRS)}")
    print(f"Tempo máximo por instância: {(MAX_TIME_PER_PAIR * len(OBJECTIVE_PAIRS))//60} minutos")
    print("="*60)
    
    # Lista arquivos de instâncias
    files = os.listdir(SSP_NPM_I_PATH)
    files = sorted(files, key=lambda x: int(x.split('_')[0][3:]) if x.startswith('ins') else 0)
    
    # Processa TODAS as instâncias (cada uma terá 1h por par de objetivos)
    test_files = [f for f in files if f.startswith('ins') and f.endswith('.csv')]  # TODAS as instâncias
    if START_INSTANCE_ID is not None:
        def _get_instance_id(filename: str) -> int:
            try:
                return int(filename.split('_')[0][3:])
            except Exception:
                return -1

        test_files = [f for f in test_files if _get_instance_id(f) >= START_INSTANCE_ID]
    
    if START_INSTANCE_ID is not None:
        print(f"📂 Processando {len(test_files)} instâncias (a partir da ins{START_INSTANCE_ID})")
    else:
        print(f"📂 Processando {len(test_files)} instâncias")
    print(f"⏱️  Cada instância: {(MAX_TIME_PER_PAIR * len(OBJECTIVE_PAIRS))//60} minutos")
    
    overall_start_time = time.time()
    
    for file_idx, file in enumerate(test_files):
        
        elapsed_time = time.time() - overall_start_time
        
        print(f"\n{'='*60}")
        print(f"INSTÂNCIA {file_idx+1}/{len(test_files)}: {file}")
        print(f"Tempo decorrido: {elapsed_time//60:.0f}min {elapsed_time%60:.0f}s")
        print(f"{'='*60}")
        
        file_base = os.path.splitext(file)[0]
        RESULTS_FILEPATH = os.path.join(BASE_DIR, f"results_gurobi/{file_base}")
        os.makedirs(RESULTS_FILEPATH, exist_ok=True)
        
        # Carrega instância
        print(f"📖 Carregando instância: {file}")
        instance_data = read_problem_instance(os.path.join(SSP_NPM_I_PATH, file))
        
        print(f"   ✅ {instance_data['num_machines']}M, {instance_data['num_jobs']}J, {instance_data['num_tools']}T")
        
        # Cada par de objetivos terá 1 hora completa
        max_time_per_pair = MAX_TIME_PER_PAIR
        max_time_per_instance = max_time_per_pair * len(OBJECTIVE_PAIRS)
        
        print(f"   ⏱️  Tempo por par: {max_time_per_pair//60}min | Total instância: {max_time_per_instance//60}min")
        
        # Resolve com Gurobi
        instance_start_time = time.time()
        
        results = solve_instance_gurobi(
            instance_data, 
            file, 
            max_time_per_pair=max_time_per_pair
        )
        
        instance_time = time.time() - instance_start_time
        
        # Salva resultados
        save_results(file, results, RESULTS_FILEPATH)
        
        # Resumo da instância
        total_solutions = sum(r['num_solutions'] for r in results.values())
        print(f"\n📊 RESUMO INSTÂNCIA:")
        print(f"   ⏱️  Tempo: {instance_time:.1f}s")
        print(f"   🎯 Soluções ótimas encontradas: {total_solutions}")
        
        for pair_key, result in results.items():
            if result['pareto_front'] is not None:
                obj1, obj2 = result['objectives']
                print(f"   📈 {obj1}-{obj2}: {result['num_solutions']} soluções em {result['execution_time']:.1f}s")
            else:
                print(f"   ❌ {pair_key}: Falhou")
    
    total_time = time.time() - overall_start_time
    
    print(f"\n{'='*60}")
    print(f"🎉 TESTE GUROBI FINALIZADO!")
    print(f"⏱️  Tempo total: {total_time//60:.0f}min {total_time%60:.0f}s")
    print(f"📂 Instâncias processadas: {file_idx+1}/{len(test_files)}")
    print(f"💾 Resultados salvos em: tests/exp_1/results_gurobi/")
    print(f"🎯 Objetivo: Fronteiras de Pareto ÓTIMAS com 1h por par")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print("\n⚠️  Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
