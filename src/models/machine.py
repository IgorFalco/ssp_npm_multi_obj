class Machine:

    id = 0

    def __init__(self, capacity, tool_change_cost, tasks_cost):
        Machine.id += 1
        self.id = Machine.id
        self.capacity = capacity
        self.tool_change_cost = tool_change_cost
        self.tasks_cost = tasks_cost
        self.tool_switches = 0
        self.flow_time = 0
        self.makespan = 0
        self.jobs = []
        self.magazine = set()

    def __repr__(self):
        return f"Machine(ID: {self.id}, Capacity: {self.capacity}, Cost: {self.tool_change_cost}, Task costs: {self.tasks_cost})"

    def check_eligibility(self, job):
        return len(job['tools']) <= self.capacity

    def add_job(self, job):
        self.jobs.append(job)