import numpy as np


class GurobiSolution:
    """
    Representa uma solução do problema SSP-NPM obtida via Gurobi.
    Compatível com a estrutura multiobjetivo.
    """
    
    def __init__(self, job_assignment, instance_data, gurobi_objectives=None, 
                 solution_id="", status="", gap=0.0):
        """
        Inicializa uma solução.
        
        Args:
            job_assignment: numpy array (num_machines, num_jobs) com 0 ou 1
            instance_data: dict com dados da instância
            gurobi_objectives: dict com objetivos calculados pelo Gurobi
            solution_id: string identificadora da solução
            status: status do solver Gurobi
            gap: gap de otimização do Gurobi
        """
        self.solution_id = solution_id
        self.job_assignment = job_assignment.copy()
        self.instance_data = instance_data
        self.gurobi_objectives = gurobi_objectives or {}
        self.status = status
        self.gap = gap
        self.objectives = {}
        self.crowding_distance = 0.0
        
        # Usa APENAS os objetivos do Gurobi - são mais precisos e confiáveis
        self.objectives = {
            'FMAX': self.gurobi_objectives.get('FMAX', 0.0),
            'TFT': self.gurobi_objectives.get('TFT', 0.0),
            'TS': self.gurobi_objectives.get('TS', 0.0)
        }
    

    
    def get_objective_value(self, objective_name):
        """
        Retorna o valor de um objetivo específico.
        
        Args:
            objective_name (str): Nome do objetivo ('FMAX', 'TFT', 'TS')
            
        Returns:
            float: Valor do objetivo
        """
        return self.objectives.get(objective_name.upper(), 0.0)
    
    def dominates(self, other_solution):
        """
        Verifica se esta solução domina outra no sentido de Pareto.
        
        Args:
            other_solution (GurobiSolution): Outra solução para comparar
            
        Returns:
            bool: True se esta solução domina a outra
        """
        # Uma solução domina outra se é melhor ou igual em todos os objetivos
        # e estritamente melhor em pelo menos um
        better_in_at_least_one = False
        
        for obj in ['FMAX', 'TFT', 'TS']:
            my_value = self.get_objective_value(obj)
            other_value = other_solution.get_objective_value(obj)
            
            if my_value > other_value:  # Pior em algum objetivo
                return False
            elif my_value < other_value:  # Melhor em algum objetivo
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def is_feasible(self):
        """
        Verifica se a solução é viável.
        
        Returns:
            bool: True se a solução é viável
        """
        # Verifica se cada job está atribuído a exatamente uma máquina
        jobs_assigned = np.sum(self.job_assignment, axis=0)
        if not np.all(jobs_assigned == 1):
            return False
        
        # Verifica restrições de capacidade dos magazines
        num_machines, num_jobs = self.job_assignment.shape
        num_tools = self.instance_data["num_tools"]
        
        for m in range(num_machines):
            assigned_jobs = np.where(self.job_assignment[m, :] == 1)[0]
            if len(assigned_jobs) == 0:
                continue
                
            # Ferramentas necessárias para todos os jobs desta máquina
            required_tools = np.zeros(num_tools, dtype=bool)
            for job_idx in assigned_jobs:
                job_tools = self.instance_data["tools_requirements_matrix"][:, job_idx]
                required_tools |= (job_tools == 1)
            
            # Verifica se não excede capacidade do magazine
            if np.sum(required_tools) > self.instance_data["magazines_capacities"][m]:
                return False
        
        return True
    
    def get_job_assignment_sequence(self):
        """
        Retorna a sequência de jobs para cada máquina.
        
        Returns:
            dict: Dicionário com máquina -> lista de jobs atribuídos
        """
        num_machines, num_jobs = self.job_assignment.shape
        sequences = {}
        
        for m in range(num_machines):
            assigned_jobs = []
            for j in range(num_jobs):
                if self.job_assignment[m, j] == 1:
                    assigned_jobs.append(j + 1)  # Jobs começam em 1
            sequences[m + 1] = assigned_jobs  # Máquinas começam em 1
        
        return sequences
    

    



class ParetoFront:
    """
    Representa uma fronteira de Pareto de soluções.
    """
    
    def __init__(self, solutions=None):
        """
        Inicializa a fronteira de Pareto.
        
        Args:
            solutions (list): Lista de soluções GurobiSolution
        """
        self.solutions = solutions or []
        self._update_crowding_distances()
    
    def add_solution(self, solution):
        """
        Adiciona uma solução à fronteira, mantendo apenas soluções não-dominadas.
        
        Args:
            solution (GurobiSolution): Solução a ser adicionada
        """
        # Remove soluções dominadas pela nova solução
        self.solutions = [
            sol for sol in self.solutions 
            if not solution.dominates(sol)
        ]
        
        # Adiciona a nova solução se ela não for dominada
        is_dominated = any(sol.dominates(solution) for sol in self.solutions)
        if not is_dominated:
            self.solutions.append(solution)
            self._update_crowding_distances()
    
    def _update_crowding_distances(self):
        """Atualiza as distâncias de crowding das soluções"""
        if len(self.solutions) <= 2:
            for sol in self.solutions:
                sol.crowding_distance = float('inf')
            return
        
        # Normaliza objetivos
        objectives = ['FMAX', 'TFT', 'TS']
        obj_values = {obj: [sol.get_objective_value(obj) for sol in self.solutions] 
                      for obj in objectives}
        
        # Calcula ranges
        obj_ranges = {}
        for obj in objectives:
            values = obj_values[obj]
            min_val, max_val = min(values), max(values)
            obj_ranges[obj] = max_val - min_val if max_val > min_val else 1.0
        
        # Inicializa distâncias
        for sol in self.solutions:
            sol.crowding_distance = 0.0
        
        # Calcula crowding distance para cada objetivo
        for obj in objectives:
            # Ordena soluções por este objetivo
            sorted_solutions = sorted(self.solutions, 
                                    key=lambda x: x.get_objective_value(obj))
            
            # Soluções extremas têm distância infinita
            sorted_solutions[0].crowding_distance = float('inf')
            sorted_solutions[-1].crowding_distance = float('inf')
            
            # Calcula distância para soluções intermediárias
            for i in range(1, len(sorted_solutions) - 1):
                if sorted_solutions[i].crowding_distance != float('inf'):
                    distance = (sorted_solutions[i+1].get_objective_value(obj) - 
                               sorted_solutions[i-1].get_objective_value(obj))
                    sorted_solutions[i].crowding_distance += distance / obj_ranges[obj]
    
    def get_best_solutions(self, n=None):
        """
        Retorna as n melhores soluções ordenadas por crowding distance.
        
        Args:
            n (int): Número de soluções a retornar (None para todas)
            
        Returns:
            list: Lista das melhores soluções
        """
        sorted_solutions = sorted(self.solutions, 
                                key=lambda x: x.crowding_distance, reverse=True)
        return sorted_solutions[:n] if n else sorted_solutions
    
    def size(self):
        """Retorna o número de soluções na fronteira"""
        return len(self.solutions)
    

    
    def save_to_csv(self, instance_name, filepath, iterations, execution_time):
        """
        Salva a fronteira de Pareto final em um arquivo CSV.
        O cabeçalho e as colunas são gerados dinamicamente.
        """
        import csv
        
        if not self.solutions:
            print("Nenhuma solução para salvar.")
            return
        
        # Obtém as chaves dos objetivos da primeira solução
        objectives_keys = list(self.solutions[0].objectives.keys())
        
        # Cria o cabeçalho dinamicamente
        header = ["instance", "solution_id", "iterations", "execution_time"] + objectives_keys

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Ordena as soluções pelo primeiro objetivo (eixo x) antes de salvar
            sorted_solutions = sorted(
                self.solutions, key=lambda s: s.objectives[objectives_keys[0]])

            for solution in sorted_solutions:
                # Cria a linha de dados dinamicamente
                row = [instance_name, solution.solution_id, iterations, execution_time]
                for key in objectives_keys:
                    row.append(solution.objectives.get(key))
                writer.writerow(row)

        print(f"Fronteira de Pareto com {len(self.solutions)} soluções salva em: {filepath}")