import numpy as np
import sys

INT_INFINITY = sys.maxsize

def GPCA(instance):
    total_system_tool_switches = 0
    for machine in instance.machines:
        print(f"\n--- Analisando Máquina {machine.id} ---")
        machine.tool_switches = 0
        if len(machine.jobs) == 0:
            continue
        tools_distances = fill_tools_distances(machine, instance.num_tools)
        start_switch_index = fill_start_magazine(machine)

        for i in range(start_switch_index, len(machine.jobs)):
            tools_for_current_job = machine.jobs[i]['tools']
            next_magazine_base = tools_for_current_job.copy()
            empty_slots = machine.capacity - len(next_magazine_base)
            removal_candidates = machine.magazine - tools_for_current_job
            dist_tool_pairs = sorted([(tools_distances[tool, i], tool) for tool in removal_candidates])

            num_switches = max(0, len(dist_tool_pairs) - empty_slots)
            machine.tool_switches += num_switches

            # --- ATUALIZAÇÃO DO MAGAZINE REAL DA MÁQUINA ---
            if empty_slots > 0:
                tools_to_keep = {pair[1] for pair in dist_tool_pairs[:empty_slots]}
                next_magazine = next_magazine_base.union(tools_to_keep)
            else:
                next_magazine = next_magazine_base

            # O atributo da máquina é atualizado diretamente
            machine.magazine = next_magazine

            cleaned_magazine = {int(tool) for tool in machine.magazine}

            print(f"Magazine: {cleaned_magazine}")
            print(f"Tool Switches: {machine.tool_switches}")

        total_system_tool_switches += machine.tool_switches

    return total_system_tool_switches

def calculate_flowtime(instance):
    total_system_flowtime = 0
    for machine in instance.machines:
        print(f"\n--- Analisando Máquina {machine.id} ---")
        machine.flowtime = 0
        machine.tool_switches = 0
        elapsed_time = 0
        if len(machine.jobs) == 0:
            continue
        tools_distances = fill_tools_distances(machine, instance.num_tools)
        start_switch_index = fill_start_magazine(machine)
        
        if start_switch_index == INT_INFINITY:
            start_switch_index = len(machine.jobs)

            # --- TAREFAS SEM CUSTO DE TROCA ---
        for i in range(start_switch_index):
            job_id = machine.jobs[i]['id']
            proc_time = machine.tasks_cost[job_id] # Pega o tempo de processamento
            
            # O relógio avança com o tempo de processamento
            elapsed_time += proc_time
            # O tempo de conclusão da tarefa (o relógio) é somado ao flowtime total
            machine.flowtime += elapsed_time

        for i in range(start_switch_index, len(machine.jobs)):
            tools_for_current_job = machine.jobs[i]['tools']
            next_magazine_base = tools_for_current_job.copy()
            empty_slots = machine.capacity - len(next_magazine_base)
            removal_candidates = machine.magazine - tools_for_current_job
            dist_tool_pairs = sorted([(tools_distances[tool, i], tool) for tool in removal_candidates])
            num_switches = max(0, len(dist_tool_pairs) - empty_slots)

            # --- LÓGICA DE CÁLCULO DE TEMPO ---
            job_id = machine.jobs[i]['id']
            proc_time = machine.tasks_cost[job_id]

            # Calcula o tempo total gasto nesta etapa
            time_for_this_step = (num_switches * machine.tool_change_cost) + proc_time
            
            # O relógio avança com o tempo total desta etapa
            elapsed_time += time_for_this_step
            # O novo tempo de conclusão é somado ao flowtime total da máquina
            machine.flowtime += elapsed_time

            print(f"Tarefa {job_id}: ProcTime={proc_time}, Switches={num_switches}, CustoEtapa={time_for_this_step}, Relogio={elapsed_time}, FlowtimeParcial={machine.flowtime}")

            # A lógica de atualização do magazine é idêntica à do GPCA
            tools_to_keep = {pair[1] for pair in dist_tool_pairs[:empty_slots]}
            machine.magazine = next_magazine_base.union(tools_to_keep)
            
        total_system_flowtime += machine.flowtime

    return total_system_flowtime

def fill_tools_distances(machine, num_tools):

    num_jobs = len(machine.jobs)

    if num_jobs == 0:
        return np.array([[]])
    
    distances = np.full((num_tools, num_jobs), INT_INFINITY)
    last_job = machine.jobs[-1]
    last_job_id = num_jobs - 1

    for tool in range(num_tools):
        if tool in last_job['tools']:
            distances[tool][last_job_id] = last_job_id

    for i in range(num_jobs - 2, -1, -1):
        current_job = machine.jobs[i]
        for tool in range(num_tools):
            if tool in current_job['tools']:
                distances[tool][i] = i
            else:
                distances[tool][i] = distances[tool][i + 1]

    return distances

def fill_start_magazine(machine):

    machine.magazine = set()

    for i, job in enumerate(machine.jobs):
        temp_magazine = machine.magazine.union(job['tools'])
        if (len(temp_magazine) > machine.capacity):
            return i
        else:
            machine.magazine.update(job['tools'])

    return INT_INFINITY

def find_best_machine_min_tsj(machines, instance):
    GPCA(instance)

    best_machine = min(machines, 
                       key=lambda m: (m.tool_switches / len(m.jobs)) if len(m.jobs) > 0 else 0)

    return best_machine

def find_most_similar_job(machine, job_list, instance):
    if not machine.jobs or not job_list:
        return None
    last_job_id = machine.jobs[-1]['id']

    most_similar_job_id = max(job_list, 
                              key=lambda job_id: instance.similarity_matrix[last_job_id, job_id])

    return most_similar_job_id
