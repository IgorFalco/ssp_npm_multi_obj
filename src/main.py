from models.solution import Solution
import solution_write
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(BASE_DIR, "results")
os.makedirs(filepath, exist_ok=True)

sol1 = Solution("inst1", 1, {"M1": [2, 5, 1], "M2": [3, 4], "M3": [6]}, {"makespan": 120, "flowtime": 250, "tool_switchs": 12})
sol2 = Solution("inst2", 2, {"M1": [3, 4], "M2": [2, 5, 1], "M3": [6]}, {"makespan": 130, "flowtime": 260, "tool_switchs": 10})
sol3 = Solution("inst3", 3, {"M1": [1, 2], "M2": [3, 4, 5]}, {"makespan": 110, "flowtime": 240, "tool_switchs": 12})

print(sol3.dominates(sol1))  # Should return True or False based on the objectives

solution_write.save_pareto_wall(sol1, filepath)
solution_write.save_solutions(sol1, filepath)
