import gurobipy as gp
from gurobipy import GRB
import numpy as np
from .input import calculate_solution_objectives


class EpsilonConstraintMethod:
    """
    Implementa o método epsilon restrito para otimização multiobjetivo
    usando o solver Gurobi.
    """
    
    def __init__(self, instance_data, time_limit=3600*24):
        """
        Inicializa o método epsilon restrito.
        
        Args:
            instance_data (dict): Dados da instância do problema
            time_limit (int): Limite de tempo em segundos para cada resolução
        """
        self.instance_data = instance_data
        self.time_limit = time_limit
        self.pareto_solutions = []
        self._base_model = None  # Cache do modelo base
        self._variables_template = None  # Cache das variáveis
        
        # Extrai dados da instância
        self.num_machines = instance_data["num_machines"]
        self.num_jobs = instance_data["num_jobs"]
        self.num_tools = instance_data["num_tools"]
        self.magazines_capacities = instance_data["magazines_capacities"]
        self.tool_change_costs = instance_data["tool_change_costs"]
        self.job_cost_per_machine = instance_data["job_cost_per_machine"]
        self.tools_requirements_matrix = instance_data["tools_requirements_matrix"]
        
        # Define conjuntos
        self.J = list(range(1, self.num_jobs + 1))
        self.M = list(range(1, self.num_machines + 1))
        self.T = list(range(1, self.num_tools + 1))
        self.R = self.J[:]  # posições
        
        # Prepara dados para o modelo
        self.p = {(j, m): self.job_cost_per_machine[m-1, j-1] 
                  for m in self.M for j in self.J}
        self.sw = {m: self.tool_change_costs[m-1] for m in self.M}
        self.C = {m: self.magazines_capacities[m-1] for m in self.M}
        self.Tj = {j: [t for t in self.T if self.tools_requirements_matrix[t-1, j-1] == 1] 
                   for j in self.J}
        
        # helper: jobs que usam cada ferramenta
        self.J_t = {t: [j for j in self.J if t in self.Tj.get(j, [])] for t in self.T}
        
        # Constante grande G
        max_p_per_job = {j: max((self.p.get((j, m), 0) for m in self.M), default=0) 
                         for j in self.J}
        G1 = sum(max_p_per_job.values())
        max_swC = max((self.sw[m] * self.C[m] for m in self.M), default=0)
        self.G = int(G1 + (len(self.J) - 1) * max_swC + 1e6)

    def build_base_model(self):
        """
        Constrói o modelo base do Gurobi com todas as restrições,
        mas sem função objetivo definida.
        
        Returns:
            tuple: (model, variables_dict)
        """
        model = gp.Model("SSP-NPM-MultiObjective")
        model.Params.TimeLimit = self.time_limit
        model.Params.OutputFlag = 0  # Reduz output verboso
        model.Params.MIPGap = 0.001
        model.Params.Threads = 0  # Usa todos os cores disponíveis
        model.Params.Presolve = 2  # Presolve agressivo
        model.Params.Cuts = 2  # Cortes agressivos
        model.Params.Heuristics = 0.1  # Reduz tempo em heurísticas
        
        # Variáveis
        x = model.addVars(self.J, self.R, self.M, vtype=GRB.BINARY, name="x")
        v = model.addVars(self.T, self.R, self.M, vtype=GRB.BINARY, name="v")
        w = model.addVars(self.T, self.R, self.M, vtype=GRB.BINARY, name="w")
        f = model.addVars(self.J, self.R, self.M, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
        FMAX = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="FMAX")
        
        # (1) cada job uma vez
        model.addConstrs(
            (gp.quicksum(x[j, r, m] for r in self.R for m in self.M) == 1 for j in self.J),
            name="assign_once"
        )
        
        # (2) uma posição tem no máx um job
        model.addConstrs(
            (gp.quicksum(x[j, r, m] for j in self.J) <= 1 for r in self.R for m in self.M),
            name="pos_at_most_one"  
        )
        
        # (3) precedência
        for r in self.R:
            if r == min(self.R): continue
            for m in self.M:
                model.addConstr(
                    gp.quicksum(x[j, r, m] for j in self.J) <= 
                    gp.quicksum(x[j, r-1, m] for j in self.J),
                    name=f"precedence_r{r}_m{m}"
                )
        
        # (4) ferramentas requeridas
        for t in self.T:
            for r in self.R:
                for m in self.M:
                    jobs_requiring_t = self.J_t[t]
                    if jobs_requiring_t:
                        model.addConstr(
                            gp.quicksum(x[j, r, m] for j in jobs_requiring_t) <= v[t, r, m],
                            name=f"tool_req_t{t}_r{r}_m{m}"
                        )
        
        # (5) capacidade
        model.addConstrs(
            (gp.quicksum(v[t, r, m] for t in self.T) <= self.C[m] 
             for r in self.R for m in self.M),
            name="capacity"
        )
        
        # (6) inserções
        for t in self.T:
            for r in self.R:
                if r == min(self.R): continue
                for m in self.M:
                    model.addConstr(
                        v[t, r, m] - v[t, r-1, m] <= w[t, r, m],
                        name=f"insert_t{t}_r{r}_m{m}"
                    )
        
        # (7) tempo para primeira posição
        r_first = min(self.R)
        for j in self.J:
            for m in self.M:
                pjm = self.p.get((j, m), 0)
                model.addConstr(f[j, r_first, m] == pjm * x[j, r_first, m], 
                               name=f"first_pos_j{j}_m{m}")

        # (8) tempos subsequentes
        for r in self.R:
            if r == r_first: continue
            for j in self.J:
                for m in self.M:
                    sum_prev = gp.quicksum(f[i, r-1, m] for i in self.J if i != j)
                    pjm = self.p.get((j, m), 0)
                    setup_time = self.sw[m] * gp.quicksum(w[t, r, m] for t in self.T)
                    model.addConstr(
                        f[j, r, m] >= sum_prev + setup_time + pjm * x[j, r, m] - 
                        self.G * (1 - x[j, r, m]),
                        name=f"f_rec_j{j}_r{r}_m{m}"
                    )
        
        # (13) FMAX >= f_jrm
        for j in self.J:
            for r in self.R:
                for m in self.M:
                    model.addConstr(FMAX >= f[j, r, m])
        
        # Expressões dos objetivos
        TFT_expr = gp.quicksum(f[j, r, m] for j in self.J for r in self.R for m in self.M)
        TS_expr = gp.quicksum(w[t, r, m] for t in self.T for r in self.R for m in self.M)
        
        variables = {
            'x': x, 'v': v, 'w': w, 'f': f, 'FMAX': FMAX,
            'TFT_expr': TFT_expr, 'TS_expr': TS_expr
        }
        
        return model, variables

    def solve_single_objective(self, objective_name):
        """
        Resolve o problema para um único objetivo.
        
        Args:
            objective_name (str): Nome do objetivo ('FMAX', 'TFT', 'TS')
            
        Returns:
            dict: Dicionário com os valores dos objetivos ou None se inviável
        """
        model, variables = self.build_base_model()
        
        # Define objetivo
        if objective_name.upper() == 'FMAX':
            model.setObjective(variables['FMAX'], GRB.MINIMIZE)
        elif objective_name.upper() == 'TFT':
            model.setObjective(variables['TFT_expr'], GRB.MINIMIZE)
        elif objective_name.upper() == 'TS':
            model.setObjective(variables['TS_expr'], GRB.MINIMIZE)
        else:
            raise ValueError("Objective must be 'FMAX', 'TFT', or 'TS'")
        
        model.optimize()
        
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            try:
                return {
                    'FMAX': variables['FMAX'].X,
                    'TFT': variables['TFT_expr'].getValue(),
                    'TS': variables['TS_expr'].getValue(),
                    'status': model.Status,
                    'iterations': model.IterCount if hasattr(model, 'IterCount') else 0, # Somar
                    'iterations_node': model.NodeCount if hasattr(model, 'NodeCount') else 0, # Somar
                    'number_of_solutions': model.SolCount if hasattr(model, 'SolCount') else 0,
                    'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0
                }
            except Exception as e:
                print(f"    Erro ao extrair valores dos objetivos: {e}")
                return None
        
        return None

    def get_objective_ranges(self, selected_objectives=None):
        """
        Calcula os ranges dos objetivos resolvendo cada um individualmente.
        
        Args:
            selected_objectives (list): Lista dos objetivos selecionados. Se None, usa todos.
        
        Returns:
            dict: Dicionário com ranges mínimos e máximos de cada objetivo
        """
        objectives = selected_objectives or ['FMAX', 'TFT', 'TS']
        print(f"Calculando ranges dos objetivos: {objectives}...")
        
        ranges = {}
        for obj in objectives:
            print(f"  Resolvendo para objetivo {obj}...")
            result = self.solve_single_objective(obj)
            if result is None:
                raise RuntimeError(f"Não foi possível resolver para objetivo {obj}")
            
            ranges[obj] = {
                'min': result[obj],
                'others': {k: v for k, v in result.items() if k != obj and k not in ['status', 'gap']}
            }
            print(f"    {obj} mínimo: {result[obj]:.2f}")
        
        return ranges

    def solve_epsilon_constraint(self, primary_objective, epsilon_values):
        """
        Resolve o problema usando o método epsilon restrito.
        
        Args:
            primary_objective (str): Objetivo a ser minimizado ('FMAX', 'TFT', 'TS')
            epsilon_values (dict): Valores epsilon para os outros objetivos
                                  Ex: {'TFT': 1000, 'TS': 50}
        
        Returns:
            dict: Solução encontrada ou None se inviável
        """
        model, variables = self.build_base_model()
        
        # Define objetivo principal
        if primary_objective.upper() == 'FMAX':
            model.setObjective(variables['FMAX'], GRB.MINIMIZE)
        elif primary_objective.upper() == 'TFT':
            model.setObjective(variables['TFT_expr'], GRB.MINIMIZE)
        elif primary_objective.upper() == 'TS':
            model.setObjective(variables['TS_expr'], GRB.MINIMIZE)
        else:
            raise ValueError("Primary objective must be 'FMAX', 'TFT', or 'TS'")
        
        # Adiciona restrições epsilon
        for obj_name, epsilon_val in epsilon_values.items():
            if obj_name.upper() == 'FMAX':
                model.addConstr(variables['FMAX'] <= epsilon_val, 
                               name=f"epsilon_{obj_name}")
            elif obj_name.upper() == 'TFT':
                model.addConstr(variables['TFT_expr'] <= epsilon_val, 
                               name=f"epsilon_{obj_name}")
            elif obj_name.upper() == 'TS':
                model.addConstr(variables['TS_expr'] <= epsilon_val, 
                               name=f"epsilon_{obj_name}")
        
        model.optimize()
        
        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            try:
                # Extrai solução
                job_assignment = np.zeros((self.num_machines, self.num_jobs), dtype=np.int32)
                
                for j in self.J:
                    for r in self.R:
                        for m in self.M:
                            if variables['x'][j, r, m].X > 0.5:
                                job_assignment[m-1, j-1] = 1
                
                # Valores dos objetivos do Gurobi
                gurobi_fmax = variables['FMAX'].X
                gurobi_tft = variables['TFT_expr'].getValue()
                gurobi_ts = variables['TS_expr'].getValue()
                
                # Calcula objetivos usando Numba para verificação
                makespan, tft, ts = calculate_solution_objectives(
                    job_assignment,
                    self.tools_requirements_matrix,
                    self.magazines_capacities,
                    self.tool_change_costs,
                    self.job_cost_per_machine
                )

                print(f" Numba: FMAX={makespan:.2f}, TFT={tft:.2f}, TS={ts:.2f} | Gurobi: FMAX={gurobi_fmax:.2f}, TFT={gurobi_tft:.2f}, TS={gurobi_ts:.2f}")
                
                # Usa os valores do Gurobi que são mais confiáveis
                return {
                    'FMAX': gurobi_fmax,
                    'TFT': gurobi_tft,
                    'TS': gurobi_ts,
                    'job_assignment': job_assignment,
                    'status': model.Status,
                    'gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0,
                    'verified_objectives': {'FMAX': makespan, 'TFT': tft, 'TS': ts},
                    'iterations': model.IterCount if hasattr(model, 'IterCount') else 0,
                    'iterations_node': model.NodeCount if hasattr(model, 'NodeCount') else 0,
                    'number_of_solutions': model.SolCount if hasattr(model, 'SolCount') else 0
                }
            except Exception as e:
                print(f"    Erro ao extrair solução: {e}")
                return None
        
        return None

    def generate_pareto_front_fast(self, num_points=8, selected_objectives=None):
        """
        Versão otimizada para gerar a fronteira de Pareto rapidamente.
        
        Args:
            num_points (int): Número aproximado de pontos na fronteira
            selected_objectives (list): Lista dos objetivos selecionados
        
        Returns:
            list: Lista de soluções da fronteira de Pareto
        """
        objectives = selected_objectives or ['FMAX', 'TFT', 'TS']
        print(f"Iniciando geração OTIMIZADA da fronteira de Pareto para {objectives}...")
        
        # Calcula ranges dos objetivos com tempo completo (24h)
        ranges = self.get_objective_ranges(objectives)
        
        pareto_solutions = []
        
        # Adiciona as soluções extremas apenas para os objetivos selecionados
        for obj in objectives:
            result = self.solve_single_objective(obj)
            if result is not None and self._is_valid_solution(result):
                # Cria job_assignment vazio para compatibilidade
                result['job_assignment'] = np.zeros((self.num_machines, self.num_jobs), dtype=np.int32)
                pareto_solutions.append(result)
                obj_values = " | ".join([f"{o}={result[o]:.2f}" for o in objectives])
                print(f"  Solução extrema {obj}: {obj_values}")
            elif result is not None:
                obj_values = " | ".join([f"{o}={result[o]:.2f}" for o in objectives])
                print(f"  ✗ Solução extrema {obj} inválida descartada: {obj_values}")
        
        # Estratégia adaptativa: foca nas regiões mais promissoras
        print(f"\nGerando pontos intermediários...")
        
        # Para 2 objetivos, usa estratégia específica
        if len(objectives) == 2:
            obj1, obj2 = objectives
            obj1_min = min(sol[obj1] for sol in pareto_solutions)
            obj1_max = max(sol[obj1] for sol in pareto_solutions)
            obj2_min = min(sol[obj2] for sol in pareto_solutions)
            obj2_max = max(sol[obj2] for sol in pareto_solutions)
        else:
            # Fallback para 3 objetivos (compatibilidade)
            tft_min = min(sol['TFT'] for sol in pareto_solutions)
            tft_max = max(sol['TFT'] for sol in pareto_solutions)
            ts_min = min(sol['TS'] for sol in pareto_solutions)
            ts_max = max(sol['TS'] for sol in pareto_solutions)
        
        # Gera pontos intermediários dinamicamente baseado nos objetivos
        successful_solutions = 0
        
        if len(objectives) == 2:
            # Para 2 objetivos, otimiza o primeiro usando o segundo como constraint
            obj1, obj2 = objectives
            step_size = max(2, int(np.sqrt(num_points)))
            obj2_values = np.linspace(obj2_min, obj2_max, step_size)
            
            for i, obj2_eps in enumerate(obj2_values):
                # Pula se já temos soluções muito próximas
                too_close = any(
                    abs(sol[obj2] - obj2_eps) < 0.1 * (obj2_max - obj2_min)
                    for sol in pareto_solutions
                )
                
                if too_close:
                    continue
                
                print(f"  Combinação {i+1}/{len(obj2_values)}: {obj2}≤{obj2_eps:.1f}")
                
                epsilon_values = {obj2: obj2_eps}
                solution = self.solve_epsilon_constraint(obj1, epsilon_values)
                successful_solutions = self._process_solution(solution, pareto_solutions, successful_solutions)
        
        else:
            # Para 3 objetivos (compatibilidade)
            step_size = max(2, int(np.sqrt(num_points)))
            tft_values = np.linspace(tft_min, tft_max, step_size)
            ts_values = np.linspace(ts_min, ts_max, step_size)
            
            combination_count = 0
            for tft_eps in tft_values:
                for ts_eps in ts_values:
                    combination_count += 1
                    
                    # Pula se já temos soluções muito próximas
                    too_close = any(
                        abs(sol['TFT'] - tft_eps) < 0.1 * (tft_max - tft_min) and
                        abs(sol['TS'] - ts_eps) < 0.1 * (ts_max - ts_min)
                        for sol in pareto_solutions
                    )
                    
                    if too_close:
                        continue
                    
                    print(f"  Combinação {combination_count}: TFT≤{tft_eps:.1f}, TS≤{ts_eps:.1f}")
                    
                    epsilon_values = {'TFT': tft_eps, 'TS': ts_eps}
                    solution = self.solve_epsilon_constraint('FMAX', epsilon_values)
                    successful_solutions = self._process_solution(solution, pareto_solutions, successful_solutions)
        
        # Soma estatísticas totais
        total_iterations = sum(sol.get('iterations', 0) for sol in pareto_solutions)
        total_iterations_node = sum(sol.get('iterations_node', 0) for sol in pareto_solutions)
        total_solutions_found = sum(sol.get('number_of_solutions', 0) for sol in pareto_solutions)
        
        # Adiciona estatísticas agregadas a cada solução
        for sol in pareto_solutions:
            sol['total_iterations'] = total_iterations + total_iterations_node
            sol['total_solutions_found'] = total_solutions_found
        
        print(f"\nFronteira de Pareto OTIMIZADA gerada com {len(pareto_solutions)} soluções.")
        print(f"Estatísticas: {total_iterations} iter + {total_iterations_node} nodes = {total_iterations + total_iterations_node} total")
        self.pareto_solutions = pareto_solutions
        return pareto_solutions

    def _process_solution(self, solution, pareto_solutions, successful_solutions):
        """Processa uma solução verificando se é válida e não-dominada"""
        if solution is not None and self._is_valid_solution(solution):
            is_non_dominated = self._is_non_dominated(solution, pareto_solutions)
            if is_non_dominated:
                old_count = len(pareto_solutions)
                pareto_solutions[:] = [
                    sol for sol in pareto_solutions
                    if not self._dominates(solution, sol)
                ]
                removed_count = old_count - len(pareto_solutions)
                
                pareto_solutions.append(solution)
                successful_solutions += 1
                obj_str = " | ".join([f"{obj}={solution[obj]:.2f}" for obj in ['FMAX', 'TFT', 'TS'] if obj in solution])
                print(f"    ✓ Solução {successful_solutions}: {obj_str}")
                if removed_count > 0:
                    print(f"      (removeu {removed_count} soluções dominadas)")
            else:
                obj_str = " | ".join([f"{obj}={solution[obj]:.2f}" for obj in ['FMAX', 'TFT', 'TS'] if obj in solution])
                print(f"    ○ Solução dominada: {obj_str}")
        elif solution is not None:
            obj_str = " | ".join([f"{obj}={solution[obj]:.2f}" for obj in ['FMAX', 'TFT', 'TS'] if obj in solution])
            print(f"    ✗ Solução inválida descartada: {obj_str}")
        else:
            print(f"    ✗ Nenhuma solução encontrada")
        
        return successful_solutions

    def _is_valid_solution(self, solution):
        """
        Verifica se uma solução é válida (não tem valores zero ou inválidos).
        
        Args:
            solution (dict): Solução a ser validada
            
        Returns:
            bool: True se a solução é válida, False caso contrário
        """
        if solution is None:
            return False
            
        # Verifica se todos os objetivos têm valores positivos e realistas
        fmax = solution.get('FMAX', 0)
        tft = solution.get('TFT', 0)
        ts = solution.get('TS', 0)
        
        # FMAX deve ser > 0 (sempre há algum tempo de processamento)
        if fmax <= 0:
            return False
            
        # TFT deve ser > 0 (sempre há algum tempo total de fluxo)
        if tft <= 0:
            return False
            
        # TS pode ser 0 ou positivo (pode não haver trocas de ferramentas)
        if ts < 0:
            return False
            
        # Verifica se os valores são realistas (não são infinitos ou muito grandes)
        max_reasonable_time = 1000000  # 1 milhão como limite superior razoável
        if fmax > max_reasonable_time or tft > max_reasonable_time:
            return False
            
        return True



    def _dominates(self, sol1, sol2):
        """Verifica se sol1 domina sol2"""
        return (sol1['FMAX'] <= sol2['FMAX'] and
                sol1['TFT'] <= sol2['TFT'] and
                sol1['TS'] <= sol2['TS'] and
                (sol1['FMAX'] < sol2['FMAX'] or
                 sol1['TFT'] < sol2['TFT'] or
                 sol1['TS'] < sol2['TS']))

    def _is_non_dominated(self, solution, pareto_solutions):
        """Verifica se uma solução é não-dominada"""
        return not any(self._dominates(existing_sol, solution) 
                      for existing_sol in pareto_solutions)

