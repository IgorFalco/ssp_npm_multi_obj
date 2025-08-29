import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

class ParetoWall:

    def __init__(self, max_size=10, objectives_keys=["makespan", "flowtime"]):
        self.max_size = max_size
        self._solutions = []
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

        if any(existing.dominates_on_axes(new_solution, *self.objectives_keys) for existing in self._solutions):
            return
        
        self._solutions = [s for s in self._solutions if not new_solution.dominates_on_axes(s, *self.objectives_keys)]
        if new_solution not in self._solutions:
            self._solutions.append(new_solution)
        else:
            return False
        if len(self._solutions) > self.max_size:
            self._trim_by_crowding_distance()

        return True

    def _trim_by_crowding_distance(self):
        """
        Calcula a distância de aglomeração e remove a solução com a menor distância
        (a que está na região mais congestionada da fronteira).
        """
        if len(self._solutions) < 3:
            return # Não faz sentido calcular para 2 ou menos pontos

        # Reseta a distância de cada solução
        for s in self._solutions:
            s.crowding_distance = 0

        # Calcula a distância para cada objetivo
        for key in self.objectives_keys:
            # Ordena as soluções pelo objetivo atual
            self._solutions.sort(key=lambda s: s.objectives[key])
            
            min_val = self._solutions[0].objectives[key]
            max_val = self._solutions[-1].objectives[key]
            val_range = max_val - min_val
            if val_range == 0: continue

            # Os extremos da fronteira são os mais importantes, damos distância infinita
            self._solutions[0].crowding_distance = float('inf')
            self._solutions[-1].crowding_distance = float('inf')
            
            # Calcula a distância para os pontos intermediários
            for i in range(1, len(self._solutions) - 1):
                distance = self._solutions[i+1].objectives[key] - self._solutions[i-1].objectives[key]
                self._solutions[i].crowding_distance += distance / val_range
        
        # Ordena a lista pela distância (menor primeiro)
        self._solutions.sort(key=lambda s: s.crowding_distance)
        
        # Remove o primeiro elemento, que é o que tem a menor distância (o mais congestionado)
        self._solutions.pop(0)

    def get_solutions(self):
        return self._solutions

    def save_to_csv(self, filepath):
        header = ["instance", "solution_id", "makespan", "flowtime", "tool_switches"]

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
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

