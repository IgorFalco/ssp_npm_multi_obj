import pandas as pd
import os

from models.instance import Instance
from models.machine import Machine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(BASE_DIR, "instances/SSP-NPM-I")
os.makedirs(filepath, exist_ok=True)


def read_problem_instance(filepath, filename):

    df = pd.read_csv(os.path.join(filepath, filename), header=None, sep=';')

    num_machines, num_jobs, num_tools = df.iloc[0, :3].dropna().astype(int).to_list()
    magazine_capacities = df.iloc[1, :num_machines].dropna().astype(int).to_list()
    tool_change_costs = df.iloc[2, :num_machines].dropna().astype(int).to_list()

    machines = []

    for i in range(num_machines):
        tasks_cost = df.iloc[3 + i, :num_jobs].dropna().astype(int).to_list()
        machine = Machine(
            capacity=magazine_capacities[i],
            tool_change_cost=tool_change_costs[i],
            tasks_cost=tasks_cost
        )
        machines.append(machine)

    tools_requirements_matrix = df.iloc[3 + num_machines:, :num_jobs].dropna(axis=1).astype(int).values

    problem = Instance(
        machines=machines,
        num_jobs=num_jobs,
        num_tools=num_tools,
        tools_requirements_matrix=tools_requirements_matrix,
    )

    return problem


instance = read_problem_instance(filepath, "ins1_m=2_j=10_t=10_var=1.csv")

print(f"Objeto do problema: {instance}")
print(f"Número de Jobs: {instance.num_jobs}")
print(f"Capacidade da máquina 1: {instance.machines[0].capacity}")
print(f"Custo da tarefa 0 na máquina 2: {instance.machines[1].tasks_cost[0]}")
print(f"Matriz de uso de ferramentas (primeiras 5 linhas):\n{instance.tools_requirements_matrix[:5,:]}")