class Machine:

    id = 0

    def __init__(self, capacity, tool_change_cost, tasks_cost):
        Machine.id += 1
        self.id = Machine.id
        self.capacity = capacity
        self.tool_change_cost = tool_change_cost
        self.tasks_cost = tasks_cost
        self.jobs = []

    def __repr__(self):
        return f"Machine(ID: {self.id}, Capacity: {self.capacity}, Cost: {self.tool_change_cost}, Task costs: {self.tasks_cost})"
