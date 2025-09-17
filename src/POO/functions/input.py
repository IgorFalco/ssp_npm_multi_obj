import os
import pandas as pd

from models.instance import Instance
from models.machine import Machine

def read_problem_instance(filepath, filename):

    df = pd.read_csv(os.path.join(filepath, filename), header=None, sep=';')

    num_machines, num_jobs, num_tools = df.iloc[0, :3].dropna().astype(int).to_list()
    magazine_capacities = df.iloc[1, :num_machines].dropna().astype(int).to_list()
    tool_change_costs = df.iloc[2, :num_machines].dropna().astype(int).to_list()

    machines = []

    for i in range(num_machines):
        tasks_cost = df.iloc[3 + i, :num_jobs].dropna().astype(int).to_list()
        machine = Machine(
            id = i,
            capacity=magazine_capacities[i],
            tool_change_cost=tool_change_costs[i],
            tasks_cost=tasks_cost
        )
        machines.append(machine)

    tools_requirements_matrix = df.iloc[3 + num_machines:, :num_jobs].dropna(axis=1).astype(int).values

    problem = Instance(
        name=filename,
        machines=machines,
        num_jobs=num_jobs,
        num_tools=num_tools,
        tools_requirements_matrix=tools_requirements_matrix,
    )
    return problem