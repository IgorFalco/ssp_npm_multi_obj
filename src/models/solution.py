class Solution:

    def __init__(self, instance, solution_id, machines, objectives):
        self.instance = instance
        self.solution_id = solution_id
        self.machines = machines
        self.objectives = objectives

    def dominates(self, other):
        better_or_equal = all(
            self.objectives[obj] <= other.objectives[obj] for obj in self.objectives)
        strictly_better = any(
            self.objectives[obj] < other.objectives[obj] for obj in self.objectives)
        return better_or_equal and strictly_better
