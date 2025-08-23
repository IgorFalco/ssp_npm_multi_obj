class Instance:

    def __init__(self, machines, num_jobs, num_tools, tools_requirements_matrix):
        self.num_jobs = num_jobs
        self.num_tools = num_tools
        self.machines = machines
        self.tools_requirements_matrix = tools_requirements_matrix  # Matriz tools x jobs

    def __repr__(self):
        return (f"ProblemInstance(Machines: {len(self.machines)}, Jobs: {self.num_jobs}, "
                f"Tools: {self.num_tools})")
