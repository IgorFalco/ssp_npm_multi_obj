class Solution:

    def __init__(self, instance, solution_id, objectives):
        self.instance_name = instance.name
        self.solution_id = solution_id
        self.assignment = {
            m.id: [job['id'] for job in m.jobs] for m in instance.machines
        }
        self.objectives = objectives
        self.crowding_distance = 0

    def __repr__(self):
        return (f"Solution(ID: {self.solution_id}, "
                f"Objectives: {self.objectives})")
    
    def __eq__(self, other):
        """Define que duas soluções são 'iguais' se seus objetivos forem idênticos."""
        if not isinstance(other, Solution):
            return NotImplemented
        return self.objectives == other.objectives
    
    def __hash__(self):
        return hash(tuple(sorted(self.objectives.items())))
    
    def print_summary(self):
        print("**************************************************")
        print(f"RESUMO DA SOLUÇÃO (ID: {self.solution_id})")
        print(f"Instância: {self.instance_name}")
        print("--------------------------------------------------")
        print("Objetivos:")
        for key, value in self.objectives.items():
            print(f"  - {key.capitalize()}: {value}")
        print("--------------------------------------------------")
        print("Atribuição de Tarefas:")
        for machine_id, job_ids in self.assignment.items():
            print(f"  - Máquina {machine_id}: {job_ids}")
        print("**************************************************")

    def dominates(self, other):
        better_or_equal = all(
            self.objectives[obj] <= other.objectives[obj] for obj in self.objectives)
        strictly_better = any(
            self.objectives[obj] < other.objectives[obj] for obj in self.objectives)
        return better_or_equal and strictly_better

    def dominates_on_axes(self, other, obj_x, obj_y):
        """
        Verifica se esta solução domina outra considerando apenas dois objetivos
        específicos (eixo x e eixo y).
        """
        axes_to_check = [obj_x, obj_y]

        # Verifica se os valores são melhores ou iguais nos dois eixos
        better_or_equal = all(
            self.objectives[obj] <= other.objectives[obj] for obj in axes_to_check
        )
        
        # Verifica se o valor é estritamente melhor em pelo menos um dos eixos
        strictly_better = any(
            self.objectives[obj] < other.objectives[obj] for obj in axes_to_check
        )
        
        return better_or_equal and strictly_better