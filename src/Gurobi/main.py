import os
import time
import argparse
from functions.input import read_problem_instance
from functions.epsilon_constraint import EpsilonConstraintMethod
from models.solution import GurobiSolution, ParetoFront
from models.pareto_wall import plot_pareto_front_2d

# Objetivos disponíveis
AVAILABLE_OBJECTIVES = {
    '1': ('TS', 'FMAX'),   # Tool Switches vs Makespan
    '2': ('TS', 'TFT'),  # Tool Switches vs Total Flow Time
}

def parse_arguments():
    """
    Processa argumentos da linha de comando.
    Uso: python main.py <arquivo.csv> <par_objetivos>
    """
    parser = argparse.ArgumentParser(description='Executa Gurobi multiobjetivo para SSP-NPM')
    parser.add_argument('instance', nargs='?', default='ins1_m=2_j=10_t=10_var=1.csv',
                       help='Nome do arquivo da instância')
    parser.add_argument('objectives', nargs='?', default='1',
                       help='Par de objetivos: 1=TS,FMAX  2=TS,TFT (padrão: 1)')
    
    args = parser.parse_args()
    
    if args.objectives not in AVAILABLE_OBJECTIVES:
        print(f"⚠️  Par de objetivos inválido '{args.objectives}'. Usando padrão: 1")
        args.objectives = '1'
    
    return args.instance, args.objectives

def main():
    """
    Otimização multiobjetivo com Gurobi usando epsilon constraint.
    """
    # Processa argumentos da linha de comando
    instance_filename, objective_choice = parse_arguments()
    selected_objectives = AVAILABLE_OBJECTIVES[objective_choice]
    
    print("🚀 GUROBI MULTIOBJETIVO - MODULAR")
    print("="*60)
    print("Método: Epsilon Restrito")
    print(f"Objetivos: {selected_objectives[0]} vs {selected_objectives[1]}")
    print("="*60)
    
    # Configurações
    BASE_DIR = os.path.dirname(__file__)
    RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
    SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-I")
    
    # Cria diretório de resultados se não existir
    if not os.path.exists(RESULTS_FILEPATH):
        os.makedirs(RESULTS_FILEPATH)
    
    # Configurações
    time_limit = 45
    num_pareto_points = 10
    
    print(f"\n📋 CONFIGURAÇÕES:")
    print(f"   Instância: {instance_filename}")
    print(f"   Objetivos: {selected_objectives[0]} e {selected_objectives[1]}")
    print(f"   Tempo limite: {time_limit}s | Pontos: {num_pareto_points}")
    
    # Lê a instância
    print(f"\n📖 Carregando instância...")
    instance_path = os.path.join(SSP_NPM_I_PATH, instance_filename)
    instance_data = read_problem_instance(instance_path)
    
    print(f"   ✅ {instance_data['num_machines']}M, {instance_data['num_jobs']}J, {instance_data['num_tools']}T")
    
    # Executa otimização
    print(f"\n🎯 INICIANDO OTIMIZAÇÃO MULTIOBJETIVO...")
    start_time = time.time()
    
    epsilon_method = EpsilonConstraintMethod(instance_data, time_limit=time_limit)
    pareto_solutions_data = epsilon_method.generate_pareto_front_fast(
        num_points=num_pareto_points, 
        selected_objectives=list(selected_objectives)
    )
    
    execution_time = time.time() - start_time
    
    print(f"\n✅ OTIMIZAÇÃO CONCLUÍDA!")
    print(f"   ⏱️  Tempo total: {execution_time:.1f} segundos")
    print(f"   📊 Soluções encontradas: {len(pareto_solutions_data)}")
    
    if not pareto_solutions_data:
        print("❌ Nenhuma solução encontrada!")
        return None, None
    
    # Converte para objetos GurobiSolution
    print(f"\n🔄 Processando soluções...")
    pareto_solutions = []
    for i, sol_data in enumerate(pareto_solutions_data):
        # Filtra objetivos selecionados
        filtered_objectives = {obj: sol_data[obj] for obj in selected_objectives}
        
        # Valida solução
        is_valid = all(val > 0 if obj in ['FMAX', 'TFT'] else val >= 0 
                      for obj, val in filtered_objectives.items())
        is_valid &= all(val < 1000000 for val in filtered_objectives.values())
        
        if is_valid:
            solution = GurobiSolution(
                job_assignment=sol_data['job_assignment'],
                instance_data=instance_data,
                gurobi_objectives=filtered_objectives,
                solution_id=f"sol_{len(pareto_solutions)+1:02d}",
                status=sol_data.get('status', 'UNKNOWN'),
                gap=sol_data.get('gap', 0.0)
            )
            pareto_solutions.append(solution)
    
    # Cria fronteira de Pareto
    pareto_front = ParetoFront(pareto_solutions)
    
    # Mostra resumo das soluções
    print(f"\n📊 RESUMO DA FRONTEIRA DE PARETO:")
    print(f"   Soluções não-dominadas: {pareto_front.size()}")
    
    for i, sol in enumerate(pareto_front.solutions):
        obj_str = " | ".join([f"{obj}={sol.objectives[obj]:6.1f}" for obj in selected_objectives])
        print(f"   Sol {i+1:2d}: {obj_str}")
    
    # Preparar diretório de resultados
    print(f"\n💾 SALVANDO RESULTADOS...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    instance_name = instance_filename.replace('.csv', '')
    results_dir = os.path.join(RESULTS_FILEPATH, f"{instance_name}_{timestamp}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 1. Salvar CSV da fronteira de Pareto
    csv_path = os.path.join(results_dir, "pareto_front.csv")
    pareto_front.save_to_csv(instance_filename, csv_path, num_pareto_points, execution_time)
    print(f"   ✅ CSV salvo: pareto_front.csv")
    
    # 2. Gerar gráfico para os objetivos selecionados
    print(f"   🎨 Gerando gráfico...")
    
    obj1, obj2 = selected_objectives
    obj_names = {'FMAX': 'Makespan', 'TFT': 'Flow_Time', 'TS': 'Tool_Switches'}
    plot_filename = f"{obj_names[obj2]}_vs_{obj_names[obj1]}.png"
    plot_2d_path = os.path.join(results_dir, plot_filename)
    
    plot_pareto_front_2d(pareto_front, obj1, obj2, save_path=plot_2d_path,
                       title=f"Fronteira de Pareto: {obj_names[obj2].replace('_', ' ')} vs. {obj_names[obj1].replace('_', ' ')}")
    print(f"   ✅ Gráfico salvo: {plot_filename}")
    
    # 3. Salvar estatísticas resumidas
    print(f"   📝 Salvando estatísticas...")
    stats_path = os.path.join(results_dir, "resumo_estatisticas.txt")
    with open(stats_path, 'w') as f:
        f.write("=== RESUMO DA FRONTEIRA DE PARETO ===\n")
        f.write(f"Soluções não-dominadas: {pareto_front.size()}\n\n")
        for i, sol in enumerate(pareto_front.solutions):
            f.write(f"Solução {i+1:2d}: FMAX={sol.objectives['FMAX']:6.1f} | "
                   f"TFT={sol.objectives['TFT']:6.1f} | TS={sol.objectives['TS']:2.0f}\n")
    print(f"   ✅ Resumo salvo: resumo_estatisticas.txt")
    
    # 4. Análise de trade-offs
    print(f"\n🔄 ANÁLISE DE TRADE-OFFS:")
    if len(pareto_solutions) >= 2:
        # Melhor solução por objetivo
        for obj in selected_objectives:
            best_sol = min(pareto_solutions, key=lambda x: x.get_objective_value(obj))
            obj_names = {'FMAX': 'Makespan', 'TFT': 'Flow Time', 'TS': 'Tool Switches'}
            values_str = " | ".join([f"{o}={best_sol.get_objective_value(o):.1f}" for o in selected_objectives])
            print(f"   🎯 Melhor {obj_names[obj]}: {values_str}")
        
        # Trade-off entre objetivos
        obj1_values = [sol.get_objective_value(selected_objectives[0]) for sol in pareto_solutions]
        obj2_values = [sol.get_objective_value(selected_objectives[1]) for sol in pareto_solutions]
        
        if len(set(obj1_values)) > 1 and len(set(obj2_values)) > 1:
            range_obj1 = max(obj1_values) - min(obj1_values)
            range_obj2 = max(obj2_values) - min(obj2_values)
            print(f"   💡 Variação: {selected_objectives[0]}({range_obj1:.1f}) vs {selected_objectives[1]}({range_obj2:.1f})")
    
    print(f"\n🎉 FINALIZADO! {pareto_front.size()} soluções em {execution_time:.1f}s")
    print(f"� Resultados: {results_dir}")
    
    # Salvar CSV de experimentos consolidado
    save_experiment_results(instance_filename, pareto_front, execution_time, "Gurobi-Epsilon")
    
    return pareto_front, results_dir


def save_experiment_results(instance_name, pareto_front, execution_time, method):
    """
    Salva os resultados do experimento em CSV consolidado.
    """
    import pandas as pd
    
    # Arquivo CSV consolidado na pasta results
    BASE_DIR = os.path.dirname(__file__)
    csv_file = os.path.join(BASE_DIR, "results", "experimentos_gurobi.csv")
    
    # Dados do experimento atual
    experiment_data = []
    
    for i, sol in enumerate(pareto_front.solutions):
        experiment_data.append({
            'instancia': instance_name.replace('.csv', ''),
            'metodo': method,
            'solucao_id': i+1,
            'FMAX': sol.objectives['FMAX'],
            'TFT': sol.objectives['TFT'], 
            'TS': sol.objectives['TS'],
            'tempo_execucao': execution_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Cria DataFrame
    df_new = pd.DataFrame(experiment_data)
    
    # Se arquivo existe, anexa; senão, cria novo
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Salva arquivo consolidado
    df_combined.to_csv(csv_file, index=False)
    print(f"   📈 Experimento salvo em: experimentos_gurobi.csv")


if __name__ == "__main__":
    try:
        # Execução da versão final
        pareto_front, results_dir = main()
        
        print(f"\n📂 Arquivos: CSV, gráfico e estatísticas salvos")
        
    except KeyboardInterrupt:
        print("\n⚠️  Execução interrompida")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()