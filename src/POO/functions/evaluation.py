import numpy as np
import sys

INT_INFINITY = sys.maxsize

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

def find_best_machine_min_tsj(machines, instance):
    calculate_all_metrics(instance)

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

def calculate_all_metrics(instance):
    for machine in instance.machines:
        machine._calculate_metrics(instance)

def get_total_tool_switches(instance):
    return sum(m.tool_switches for m in instance.machines)

def get_total_flowtime(instance):
    return sum(m.flowtime for m in instance.machines)

def get_system_makespan(instance):
    return max(m.makespan for m in instance.machines)