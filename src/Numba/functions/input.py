import numpy as np
import pandas as pd


def read_problem_instance(file_path):

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

    return {
        "num_machines": num_machines,
        "num_jobs": num_jobs,
        "num_tools": num_tools,
        "magazines_capacities": magazines_capacities,
        "tool_change_costs": tool_change_costs,
        "job_cost_per_machine": job_cost_per_machine,
        "tools_requirements_matrix": tools_requirements_matrix,
    }
