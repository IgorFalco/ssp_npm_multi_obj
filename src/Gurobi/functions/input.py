import numpy as np
import pandas as pd
from numba import njit


def read_problem_instance(file_path):
    """
    Lê uma instância do problema a partir de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV da instância
        
    Returns:
        dict: Dicionário contendo os dados da instância
    """
    df = pd.read_csv(file_path, header=None, sep=';')

    num_machines, num_jobs, num_tools = df.iloc[0, :3].dropna().astype(
        int).to_list()

    magazines_capacities = np.array(
        df.iloc[1, :num_machines].dropna().astype(int).to_list(), dtype=np.int64)

    tool_change_costs = np.array(
        df.iloc[2, :num_machines].dropna().astype(int).to_list(), dtype=np.int64)

    job_cost_per_machine = np.empty((num_machines, num_jobs), dtype=np.int64)

    for i in range(num_machines):
        job_cost_per_machine[i, :] = df.iloc[3 + i,
                                             :num_jobs].dropna().astype(int).to_list()

    tools_requirements_matrix = np.array(
        df.iloc[3 + num_machines:,
                :num_jobs].dropna(axis=1).astype(int).values,
        dtype=np.int64)

    tools_per_job = tools_requirements_matrix.sum(axis=0)

    return {
        "num_machines": num_machines,
        "num_jobs": num_jobs,
        "num_tools": num_tools,
        "magazines_capacities": magazines_capacities,
        "tool_change_costs": tool_change_costs,
        "job_cost_per_machine": job_cost_per_machine,
        "tools_requirements_matrix": tools_requirements_matrix,
        "tools_per_job": tools_per_job,
    }


@njit
def calculate_solution_objectives(job_assignment, tools_requirements_matrix, 
                                magazines_capacities, tool_change_costs, 
                                job_cost_per_machine):
    """
    Calcula os objetivos de uma solução usando Numba para otimização.
    
    Args:
        job_assignment: Array numpy com a atribuição de jobs
        tools_requirements_matrix: Matriz de requisitos de ferramentas
        magazines_capacities: Capacidades dos magazines
        tool_change_costs: Custos de troca de ferramentas
        job_cost_per_machine: Custos de jobs por máquina
        
    Returns:
        tuple: (makespan, total_flowtime, total_tool_switches)
    """
    num_machines, num_jobs = job_assignment.shape
    num_tools = tools_requirements_matrix.shape[0]
    
    makespans = np.zeros(num_machines)
    total_flowtime = 0.0
    total_tool_switches = 0
    
    for m in range(num_machines):
        # Jobs atribuídos a esta máquina
        assigned_jobs = []
        for j in range(num_jobs):
            if job_assignment[m, j] == 1:
                assigned_jobs.append(j)
        
        if len(assigned_jobs) == 0:
            continue
            
        # Calcula makespan e flowtime desta máquina
        machine_time = 0.0
        current_tools = np.zeros(num_tools, dtype=np.int32)
        
        for job_idx in assigned_jobs:
            # Ferramentas necessárias para este job
            required_tools = tools_requirements_matrix[:, job_idx]
            
            # Calcula trocas de ferramentas necessárias
            tools_to_add = 0
            for t in range(num_tools):
                if required_tools[t] == 1 and current_tools[t] == 0:
                    tools_to_add += 1
            
            # Se exceder capacidade, precisamos trocar
            current_tool_count = np.sum(current_tools)
            if current_tool_count + tools_to_add > magazines_capacities[m]:
                # Remove ferramentas desnecessárias
                tools_to_remove = current_tool_count + tools_to_add - magazines_capacities[m]
                removed = 0
                for t in range(num_tools):
                    if removed >= tools_to_remove:
                        break
                    if current_tools[t] == 1 and required_tools[t] == 0:
                        current_tools[t] = 0
                        removed += 1
                        total_tool_switches += 1
            
            # Adiciona ferramentas necessárias
            for t in range(num_tools):
                if required_tools[t] == 1 and current_tools[t] == 0:
                    current_tools[t] = 1
                    total_tool_switches += 1
            
            # Tempo de setup (trocas de ferramentas)
            setup_time = tools_to_add * tool_change_costs[m]
            
            # Tempo de processamento do job
            processing_time = job_cost_per_machine[m, job_idx]
            
            # Atualiza tempo da máquina
            machine_time += setup_time + processing_time
            total_flowtime += machine_time
        
        makespans[m] = machine_time
    
    makespan = np.max(makespans)
    
    return makespan, total_flowtime, total_tool_switches