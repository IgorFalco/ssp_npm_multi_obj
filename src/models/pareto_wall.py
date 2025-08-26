import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

class ParetoWall:

    def __init__(self, objectives_keys=["makespan", "flowtime"]):
        self._solutions = set()
        self.objectives_keys = objectives_keys
    
    def __len__(self):
        return len(self._solutions)

    def __iter__(self):
        return iter(self._solutions)
    
    def plot(self, save_path=None):

        x_axis, y_axis = self.objectives_keys[0], self.objectives_keys[1]

        if not self._solutions:
            print("Nenhuma solução no arquivo para plotar.")
            return
    
        data_for_df = [s.objectives for s in self._solutions]

        df = pd.DataFrame(data_for_df)
        df_sorted = df.sort_values(by=x_axis)
        plt.plot(df_sorted[x_axis], df_sorted[y_axis], 'o-')
        plt.xlabel(x_axis.replace('_', ' ').capitalize())
        plt.ylabel(y_axis.replace('_', ' ').capitalize())
        plt.title(f"Fronteira de Pareto: {x_axis.capitalize()} vs. {y_axis.capitalize()}")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Gráfico salvo em: {save_path}")
        
        plt.show()

    
    def add(self, new_solution):
        obj_x, obj_y = self.objectives_keys[0], self.objectives_keys[1]

        if any(existing.dominates_on_axes(new_solution, obj_x, obj_y) for existing in self._solutions):
            return False
        
        dominated_by_new = {s for s in self._solutions if new_solution.dominates_on_axes(s, obj_x, obj_y)}
        self._solutions -= dominated_by_new
        self._solutions.add(new_solution)
        return True

    def get_solutions(self):
        return self._solutions

    def save_to_csv(self, filepath):
        header = ["instance", "solution_id", "makespan", "flowtime", "tool_switches"]

        file_exists = os.path.isfile(filepath)

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(header)
            
            for solution in self._solutions:    
                row = [
                    solution.instance_name,
                    solution.solution_id,
                    solution.objectives.get("makespan"),
                    solution.objectives.get("flowtime"),
                    solution.objectives.get("tool_switches")
                ]
                writer.writerow(row)
        print(f"Fronteira de Pareto com {len(self)} soluções salva em: {filepath}")

