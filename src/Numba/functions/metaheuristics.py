import numpy as np
from numba import njit


@njit
def check_machine_eligibility(job_id, machine_id, magazines_capacities, tools_per_job):
    return tools_per_job[job_id] <= magazines_capacities[machine_id]


def calculate_similarity_matrix(tools_requirements_matrix):
    matrix_T = tools_requirements_matrix.T
    shared_ones = np.dot(matrix_T, tools_requirements_matrix)
    inverted_matrix = 1 - tools_requirements_matrix
    shared_zeros = np.dot(inverted_matrix.T, inverted_matrix)
    return shared_ones + shared_zeros


@njit
def calculate_tool_switches_for_machine(machine_jobs, tools_requirements_matrix, magazine_capacity):
    if len(machine_jobs) == 0:
        return 0

    num_tools = tools_requirements_matrix.shape[0]
    magazine = np.zeros(num_tools, dtype=np.int64)
    switches = 0

    for job_id in machine_jobs:
        job_tools = tools_requirements_matrix[:, job_id]

        # Ferramentas que precisariam estar no magazine
        needed_tools = np.logical_or(magazine, job_tools).astype(np.int64)
        tools_count = np.sum(needed_tools)

        if tools_count > magazine_capacity:
            # Precisa remover ferramentas
            excess = tools_count - magazine_capacity
            switches += excess

            # Mantém ferramentas do job atual e remove outras quando possível
            # Prioridade: 1) ferramentas do job atual, 2) ferramentas já no magazine
            priority_tools = job_tools.copy()

            # Adiciona ferramentas já no magazine que não conflitam
            remaining_capacity = magazine_capacity - np.sum(job_tools)
            if remaining_capacity > 0:
                # ferramentas antigas não conflitantes
                available_old_tools = magazine & ~job_tools
                old_tools_indices = np.where(available_old_tools)[0]

                # Adiciona ferramentas antigas até o limite
                for i, tool_idx in enumerate(old_tools_indices):
                    if i < remaining_capacity:
                        priority_tools[tool_idx] = 1
                    else:
                        break

            magazine = priority_tools
        else:
            # Todas as ferramentas cabem
            magazine = needed_tools

    return switches


@njit
def calculate_tool_switches_all_machines(job_assignment, tools_requirements_matrix, magazines_capacities):
    tool_switches = np.zeros(len(magazines_capacities), dtype=np.int64)

    for machine_id in range(len(magazines_capacities)):
        machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
        tool_switches[machine_id] = calculate_tool_switches_for_machine(
            machine_jobs, tools_requirements_matrix, magazines_capacities[machine_id]
        )

    return tool_switches


@njit
def find_best_machine_min_tsj(job_assignment, tools_requirements_matrix, magazines_capacities):
    num_machines = magazines_capacities.shape[0]
    ratios = np.full(num_machines, np.inf, dtype=np.float64)

    for machine_id in range(len(magazines_capacities)):
        machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
        num_jobs = len(machine_jobs)

        if num_jobs > 0:
            tool_switches = calculate_tool_switches_for_machine(
                machine_jobs, tools_requirements_matrix, magazines_capacities[machine_id]
            )
            ratios[machine_id] = tool_switches / np.size(machine_jobs)

    order = np.argsort(ratios)

    ranked = order[np.isfinite(ratios[order])]

    return ranked


@njit
def find_most_similar_job(machine_id, job_assignment, jobs_list, similarity_matrix):
    machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
    if len(machine_jobs) == 0:
        return None

    available_jobs = [job for job in jobs_list if job != -1]
    if not available_jobs:
        return None
    
    last_job_id = machine_jobs[-1]
    best_job = available_jobs[0]
    best_similarity = similarity_matrix[last_job_id, best_job]

    for job_id in available_jobs[1:]:
        similarity = similarity_matrix[last_job_id, job_id]
        if similarity > best_similarity:
            best_similarity = similarity
            best_job = job_id

    return best_job


def construct_initial_solution(num_jobs, num_machines, magazines_capacities, tools_per_job, tools_requirements_matrix):
    jobs_list = np.random.permutation(num_jobs)
    job_assignment = np.full((num_machines, num_jobs), -1, dtype=np.int64)

    similarity_matrix = calculate_similarity_matrix(tools_requirements_matrix)

    # Fase 1: Atribuição inicial - uma tarefa por máquina
    for i in range(num_machines):
        for j, job in enumerate(jobs_list):
            if job != -1 and check_machine_eligibility(job, i, magazines_capacities, tools_per_job):
                job_assignment[i, 0] = job
                jobs_list[j] = -1
                break

    # Fase 2: Alocação gulosa das tarefas restantes
    remaining_jobs = [job for job in jobs_list if job != -1]

    while remaining_jobs:
        ranked = find_best_machine_min_tsj(
            job_assignment, tools_requirements_matrix, magazines_capacities)

        for target_machine in ranked:

            most_similar_job_id = find_most_similar_job(
                target_machine, job_assignment, jobs_list, similarity_matrix)

            if most_similar_job_id is None:
                break

            if check_machine_eligibility(most_similar_job_id, target_machine, magazines_capacities, tools_per_job):
                pos = np.where(job_assignment[target_machine] == -1)[0][0]
                job_assignment[target_machine, pos] = most_similar_job_id
                jobs_list[np.where(jobs_list == most_similar_job_id)[
                    0][0]] = -1
                remaining_jobs.remove(most_similar_job_id)
                break

    return job_assignment


@njit
def calculate_makespan_for_machine(machine_jobs, tools_requirements_matrix, magazine_capacity, 
                                 tool_change_cost, job_costs):
    if len(machine_jobs) == 0:
        return 0
    
    num_tools = tools_requirements_matrix.shape[0]
    magazine = np.zeros(num_tools, dtype=np.int64)
    elapsed_time = 0
    
    # Encontra o índice onde começam as trocas de ferramentas
    start_switch_index = len(machine_jobs)
    for i, job_id in enumerate(machine_jobs):
        job_tools = tools_requirements_matrix[:, job_id]
        temp_magazine = np.logical_or(magazine, job_tools).astype(np.int64)
        
        if np.sum(temp_magazine) > magazine_capacity:
            start_switch_index = i
            break
        else:
            magazine = temp_magazine
    
    # Fase 1: Jobs sem custo de troca
    for i in range(start_switch_index):
        job_id = machine_jobs[i]
        proc_time = job_costs[job_id]
        elapsed_time += proc_time
    
    # Fase 2: Jobs com custo de troca
    for i in range(start_switch_index, len(machine_jobs)):
        job_id = machine_jobs[i]
        job_tools = tools_requirements_matrix[:, job_id]
        
        needed_tools = np.logical_or(magazine, job_tools).astype(np.int64)
        tools_count = np.sum(needed_tools)
        
        if tools_count > magazine_capacity:
            excess = tools_count - magazine_capacity
            num_switches = excess
            
            # Atualiza magazine mantendo ferramentas do job atual
            priority_tools = job_tools.copy()
            remaining_capacity = magazine_capacity - np.sum(job_tools)
            
            if remaining_capacity > 0:
                available_old_tools = magazine & ~job_tools
                old_tools_indices = np.where(available_old_tools)[0]
                
                for j, tool_idx in enumerate(old_tools_indices):
                    if j < remaining_capacity:
                        priority_tools[tool_idx] = 1
                    else:
                        break
            
            magazine = priority_tools
        else:
            num_switches = 0
            magazine = needed_tools
        
        proc_time = job_costs[job_id]
        switch_time = num_switches * tool_change_cost
        elapsed_time += switch_time + proc_time
    
    return elapsed_time


@njit
def calculate_flowtime_for_machine(machine_jobs, tools_requirements_matrix, magazine_capacity, 
                                 tool_change_cost, job_costs):
    if len(machine_jobs) == 0:
        return 0
    
    num_tools = tools_requirements_matrix.shape[0]
    magazine = np.zeros(num_tools, dtype=np.int64)
    elapsed_time = 0
    flowtime = 0
    
    # Encontra o índice onde começam as trocas de ferramentas
    start_switch_index = len(machine_jobs)
    for i, job_id in enumerate(machine_jobs):
        job_tools = tools_requirements_matrix[:, job_id]
        temp_magazine = np.logical_or(magazine, job_tools).astype(np.int64)
        
        if np.sum(temp_magazine) > magazine_capacity:
            start_switch_index = i
            break
        else:
            magazine = temp_magazine
    
    # Fase 1: Jobs sem custo de troca
    for i in range(start_switch_index):
        job_id = machine_jobs[i]
        proc_time = job_costs[job_id]
        elapsed_time += proc_time
        flowtime += elapsed_time
    
    # Fase 2: Jobs com custo de troca
    for i in range(start_switch_index, len(machine_jobs)):
        job_id = machine_jobs[i]
        job_tools = tools_requirements_matrix[:, job_id]
        
        needed_tools = np.logical_or(magazine, job_tools).astype(np.int64)
        tools_count = np.sum(needed_tools)
        
        if tools_count > magazine_capacity:
            excess = tools_count - magazine_capacity
            num_switches = excess
            
            # Atualiza magazine mantendo ferramentas do job atual
            priority_tools = job_tools.copy()
            remaining_capacity = magazine_capacity - np.sum(job_tools)
            
            if remaining_capacity > 0:
                available_old_tools = magazine & ~job_tools
                old_tools_indices = np.where(available_old_tools)[0]
                
                for j, tool_idx in enumerate(old_tools_indices):
                    if j < remaining_capacity:
                        priority_tools[tool_idx] = 1
                    else:
                        break
            
            magazine = priority_tools
        else:
            num_switches = 0
            magazine = needed_tools
        
        proc_time = job_costs[job_id]
        switch_time = num_switches * tool_change_cost
        elapsed_time += switch_time + proc_time
        flowtime += elapsed_time
    
    return flowtime


@njit
def calculate_makespan_all_machines(job_assignment, tools_requirements_matrix, magazines_capacities, 
                                  tool_change_costs, job_cost_per_machine):
    makespans = np.zeros(len(magazines_capacities), dtype=np.int64)
    
    for machine_id in range(len(magazines_capacities)):
        machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
        makespans[machine_id] = calculate_makespan_for_machine(
            machine_jobs, 
            tools_requirements_matrix, 
            magazines_capacities[machine_id],
            tool_change_costs[machine_id],
            job_cost_per_machine[machine_id]
        )
    
    return makespans


@njit
def calculate_flowtime_all_machines(job_assignment, tools_requirements_matrix, magazines_capacities, 
                                  tool_change_costs, job_cost_per_machine):
    flowtimes = np.zeros(len(magazines_capacities), dtype=np.int64)
    
    for machine_id in range(len(magazines_capacities)):
        machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
        flowtimes[machine_id] = calculate_flowtime_for_machine(
            machine_jobs, 
            tools_requirements_matrix, 
            magazines_capacities[machine_id],
            tool_change_costs[machine_id],
            job_cost_per_machine[machine_id]
        )
    
    return flowtimes


@njit
def get_system_makespan(makespans):
    return np.max(makespans)


@njit
def get_total_flowtime(flowtimes):
    return np.sum(flowtimes)
