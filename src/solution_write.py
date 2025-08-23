import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


def save_pareto_wall(solution, filepath):

    header = ["instance", "solution_id",
              "makespan", "flowtime", "tool_switchs"]

    filename = os.path.join(filepath, "pareto_wall.csv")
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        row = [
            solution.instance,
            solution.solution_id,
            solution.objectives.get("makespan"),
            solution.objectives.get("flowtime"),
            solution.objectives.get("tool_switchs")
        ]
        writer.writerow(row)


def save_solutions(solution, filepath):
    """
    Salva as soluções no formato longo (uma linha por máquina).
    """
    header = ["instance", "solution_id", "machine_id",
              "jobs", "makespan", "flowtime", "tool_switchs"]

    filename = os.path.join(filepath, "solutions.csv")

    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        for machine_id, jobs in solution.machines.items():
            row = [
                solution.instance,
                solution.solution_id,
                machine_id,
                jobs,
                solution.objectives.get("makespan"),
                solution.objectives.get("flowtime"),
                solution.objectives.get("tool_switchs")
            ]
            writer.writerow(row)

def plot_pareto_wall(filepath):

    df = pd.read_csv(os.path.join(filepath, "pareto_wall.csv"))
    plt.plot(df['makespan'], df['flowtime'], '--*')
    plt.xlabel("Makespan")
    plt.ylabel("Flowtime")
    plt.title("Pareto Wall")
    plt.grid(True)
    plt.show()