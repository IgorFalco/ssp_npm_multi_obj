import numpy as np
import csv
from numba import njit


@njit
def dominates_on_objectives(obj1, obj2, obj_indices):
    """Verifica se obj1 domina obj2 nos objetivos especificados"""
    better_or_equal = True
    strictly_better = False
    
    for i in obj_indices:
        if obj1[i] > obj2[i]:
            better_or_equal = False
            break
        elif obj1[i] < obj2[i]:
            strictly_better = True
    
    return better_or_equal and strictly_better


@njit
def calculate_crowding_distances(objectives_matrix):
    """Calcula distâncias de crowding para um conjunto de soluções"""
    n_solutions, n_objectives = objectives_matrix.shape
    distances = np.zeros(n_solutions)
    
    for obj_idx in range(n_objectives):
        # Ordena índices pelos valores do objetivo
        sorted_indices = np.argsort(objectives_matrix[:, obj_idx])
        
        obj_values = objectives_matrix[:, obj_idx]
        min_val = obj_values[sorted_indices[0]]
        max_val = obj_values[sorted_indices[-1]]
        
        if max_val - min_val == 0:
            continue
            
        # Extremos recebem distância infinita
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        # Calcula distância para pontos intermediários
        for i in range(1, n_solutions - 1):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i-1]
            next_idx = sorted_indices[i+1]
            
            distance = (obj_values[next_idx] - obj_values[prev_idx]) / (max_val - min_val)
            distances[idx] += distance
    
    return distances


class ParetoWall:
    
    def __init__(self, max_size=10, objectives_keys=["tool_switches", "makespan"]):
        self.max_size = max_size
        self.objectives_keys = objectives_keys
        self._solutions = []
        self._objectives_cache = np.empty((0, len(objectives_keys)))
        self._obj_indices = np.arange(len(objectives_keys))
    
    def __len__(self):
        return len(self._solutions)
    
    def __iter__(self):
        return iter(self._solutions)
    
    def _update_cache(self):
        """Atualiza cache de objetivos para operações rápidas"""
        if not self._solutions:
            self._objectives_cache = np.empty((0, len(self.objectives_keys)))
            return
            
        n_solutions = len(self._solutions)
        self._objectives_cache = np.zeros((n_solutions, len(self.objectives_keys)))
        
        for i, solution in enumerate(self._solutions):
            for j, key in enumerate(self.objectives_keys):
                self._objectives_cache[i, j] = solution.objectives[key]
    
    def add(self, new_solution):
        """Adiciona nova solução mantendo dominância de Pareto"""
        if not self._solutions:
            self._solutions.append(new_solution)
            self._update_cache()
            return True
        
        new_objectives = np.array([new_solution.objectives[key] for key in self.objectives_keys])
        
        # Verifica se nova solução é dominada
        for i, existing_obj in enumerate(self._objectives_cache):
            if dominates_on_objectives(existing_obj, new_objectives, self._obj_indices):
                return False
        
        # Remove soluções dominadas pela nova
        non_dominated = []
        for i, solution in enumerate(self._solutions):
            if not dominates_on_objectives(new_objectives, self._objectives_cache[i], self._obj_indices):
                non_dominated.append(solution)
        
        # Adiciona nova solução se não existir
        if new_solution not in non_dominated:
            non_dominated.append(new_solution)
            self._solutions = non_dominated
            self._update_cache()
            
            # Aplica crowding distance se necessário
            if len(self._solutions) > self.max_size:
                self._trim_by_crowding_distance()
            
            return True
        
        return False
    
    def _trim_by_crowding_distance(self):
        """Remove soluções usando crowding distance"""
        if len(self._solutions) < 3:
            return
        
        distances = calculate_crowding_distances(self._objectives_cache)
        
        # Encontra índice com menor distância (excluindo infinitos)
        finite_distances = distances[np.isfinite(distances)]
        if len(finite_distances) == 0:
            return
        
        min_distance = np.min(finite_distances)
        candidates = np.where(distances == min_distance)[0]
        
        # Remove primeiro candidato com menor distância
        remove_idx = candidates[0]
        self._solutions.pop(remove_idx)
        self._update_cache()
    
    def get_solutions(self):
        """Retorna lista de soluções"""
        return self._solutions
    
    def get_best_by_objective(self, objective_key):
        """Retorna melhor solução para um objetivo específico"""
        if not self._solutions:
            return None
        
        if objective_key not in self.objectives_keys:
            return None
        
        return min(self._solutions, key=lambda s: s.objectives[objective_key])
    
    def get_objectives_matrix(self):
        """Retorna matriz de objetivos para análises externas"""
        return self._objectives_cache.copy()
    
    def save_to_csv(self, filepath):
        """Salva fronteira em CSV"""
        if not self._solutions:
            return
        
        header = ["solution_id"] + self.objectives_keys
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            # Ordena pelo primeiro objetivo
            obj_idx = 0
            sorted_solutions = sorted(self._solutions, 
                                    key=lambda s: s.objectives[self.objectives_keys[obj_idx]])
            
            for solution in sorted_solutions:
                row = [solution.solution_id]
                for key in self.objectives_keys:
                    row.append(solution.objectives[key])
                writer.writerow(row)
    
    def clear(self):
        """Limpa todas as soluções"""
        self._solutions.clear()
        self._update_cache()
    
    def merge(self, other_wall):
        """Mescla com outra ParetoWall"""
        for solution in other_wall.get_solutions():
            self.add(solution)


def create_combined_pareto_wall(pareto_walls, max_size=50):
    """Combina múltiplas ParetoWalls em uma única"""
    if not pareto_walls:
        return None
    
    combined = ParetoWall(max_size=max_size, 
                         objectives_keys=pareto_walls[0].objectives_keys)
    
    for wall in pareto_walls:
        combined.merge(wall)
    
    return combined


def plot_combined_pareto(pareto_walls, save_path=None, title="Combined Pareto Fronts"):
    """
    Plota múltiplas fronteiras Pareto combinadas
    
    Args:
        pareto_walls: Lista de ParetoWall objects
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not pareto_walls:
        print("Nenhuma fronteira Pareto fornecida para plotagem")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 
              'pink', 'gray', 'olive', 'cyan']
    
    # Combina todas as fronteiras
    all_objectives = []
    
    for i, wall in enumerate(pareto_walls):
        if len(wall) == 0:
            continue
            
        objectives_matrix = wall.get_objectives_matrix()
        
        if objectives_matrix.size > 0:
            color = colors[i % len(colors)]
            
            # Plota pontos individuais
            plt.scatter(objectives_matrix[:, 0], objectives_matrix[:, 1], 
                       c=color, alpha=0.6, s=50, 
                       label=f'Run {i+1} ({len(wall)} soluções)')
            
            all_objectives.append(objectives_matrix)
    
    # Combina todas as soluções para fronteira global
    if all_objectives:
        combined_obj = np.vstack(all_objectives)
        
        # Encontra fronteira global
        combined_wall = ParetoWall(max_size=100, 
                                  objectives_keys=pareto_walls[0].objectives_keys)
        
        # Adiciona todas as soluções (simulando)
        for wall in pareto_walls:
            combined_wall.merge(wall)
        
        if len(combined_wall) > 0:
            global_objectives = combined_wall.get_objectives_matrix()
            
            # Ordena para conectar pontos da fronteira
            sorted_indices = np.argsort(global_objectives[:, 0])
            sorted_obj = global_objectives[sorted_indices]
            
            plt.plot(sorted_obj[:, 0], sorted_obj[:, 1], 
                    'k-', linewidth=2, alpha=0.8, 
                    label=f'Fronteira Global ({len(combined_wall)} soluções)')
    
    # Configurações do gráfico
    objectives_keys = pareto_walls[0].objectives_keys
    plt.xlabel(objectives_keys[0].replace('_', ' ').title())
    plt.ylabel(objectives_keys[1].replace('_', ' ').title())
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico combinado salvo em: {save_path}")
    
    plt.show()
