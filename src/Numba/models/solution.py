import numpy as np
from numba import njit
import sys
import os

# Adiciona o caminho das functions ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from functions.metaheuristics import (
    calculate_tool_switches_all_machines,
    calculate_makespan_all_machines, 
    calculate_flowtime_all_machines,
    get_system_makespan,
    get_total_flowtime
)


class Solution:
    
    def __init__(self, job_assignment, instance_data, solution_id=""):
        """
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        instance_data: dict com dados da instância
        solution_id: string identificadora da solução
        """
        self.solution_id = solution_id
        self.job_assignment = job_assignment.copy()
        self.instance_data = instance_data
        self.objectives = {}
        self.crowding_distance = 0.0
        
        # Calcula objetivos automaticamente
        self._calculate_objectives()
    
    def _calculate_objectives(self):
        """Calcula todos os objetivos da solução"""
        # Tool switches
        tool_switches = calculate_tool_switches_all_machines(
            self.job_assignment,
            self.instance_data["tools_requirements_matrix"],
            self.instance_data["magazines_capacities"]
        )
        
        # Makespans
        makespans = calculate_makespan_all_machines(
            self.job_assignment,
            self.instance_data["tools_requirements_matrix"],
            self.instance_data["magazines_capacities"],
            self.instance_data["tool_change_costs"],
            self.instance_data["job_cost_per_machine"]
        )
        
        # Flowtimes
        flowtimes = calculate_flowtime_all_machines(
            self.job_assignment,
            self.instance_data["tools_requirements_matrix"],
            self.instance_data["magazines_capacities"],
            self.instance_data["tool_change_costs"],
            self.instance_data["job_cost_per_machine"]
        )
        
        # Atualiza objetivos
        self.objectives = {
            "tool_switches": int(np.sum(tool_switches)),
            "makespan": int(get_system_makespan(makespans)),
            "flowtime": int(get_total_flowtime(flowtimes))
        }
    
    def get_assignment_dict(self):
        """Retorna assignment no formato de dicionário (compatível com POO)"""
        assignment = {}
        for machine_id in range(self.job_assignment.shape[0]):
            assigned_jobs = np.where(self.job_assignment[machine_id] != -1)[0]
            assignment[machine_id] = assigned_jobs.tolist()
        return assignment
    
    def dominates(self, other):
        """Verifica se esta solução domina outra (Pareto dominance)"""
        better_or_equal = all(
            self.objectives[obj] <= other.objectives[obj] 
            for obj in self.objectives
        )
        strictly_better = any(
            self.objectives[obj] < other.objectives[obj] 
            for obj in self.objectives
        )
        return better_or_equal and strictly_better
    
    def dominates_on_axes(self, other, obj_x, obj_y):
        """Verifica dominância apenas em dois objetivos específicos"""
        axes = [obj_x, obj_y]
        better_or_equal = all(
            self.objectives[obj] <= other.objectives[obj] for obj in axes
        )
        strictly_better = any(
            self.objectives[obj] < other.objectives[obj] for obj in axes
        )
        return better_or_equal and strictly_better
    
    def copy(self):
        """Cria uma cópia da solução"""
        return Solution(
            self.job_assignment.copy(),
            self.instance_data,
            self.solution_id + "_copy"
        )
    
    def print_summary(self):
        """Imprime resumo da solução"""
        print("**************************************************")
        print(f"RESUMO DA SOLUÇÃO (ID: {self.solution_id})")
        print("--------------------------------------------------")
        print("Objetivos:")
        for key, value in self.objectives.items():
            print(f"  - {key.capitalize()}: {value}")
        print("--------------------------------------------------")
        print("Atribuição de Tarefas:")
        assignment = self.get_assignment_dict()
        for machine_id, job_ids in assignment.items():
            print(f"  - Máquina {machine_id}: {job_ids}")
        print("**************************************************")
    
    def __repr__(self):
        return (f"Solution(ID: {self.solution_id}, "
                f"Objectives: {self.objectives})")
    
    def __eq__(self, other):
        """Duas soluções são iguais se têm os mesmos objetivos"""
        if not isinstance(other, Solution):
            return False
        return self.objectives == other.objectives
    
    def __hash__(self):
        return hash(tuple(sorted(self.objectives.items())))