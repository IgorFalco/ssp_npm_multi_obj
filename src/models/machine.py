import sys

from functions.evaluation import fill_tools_distances

INT_INFINITY = sys.maxsize

class Machine:

    def __init__(self, capacity, tool_change_cost, tasks_cost, id):
        self.id = id
        self.capacity = capacity
        self.tool_change_cost = tool_change_cost
        self.tasks_cost = tasks_cost
        self.jobs = []
        self.magazine = set()

        self.tool_switches = 0
        self.flow_time = 0
        self.makespan = 0

    def __repr__(self):
        return f"Machine(ID: {self.id}, Capacity: {self.capacity}, Cost: {self.tool_change_cost}, Task costs: {self.tasks_cost})"

    def check_eligibility(self, job):
        return len(job['tools']) <= self.capacity

    def add_job(self, job):
        self.jobs.append(job)
        
    def _initialize_magazine_and_find_start_index(self):
        """
        Método interno para preparar o magazine inicial e encontrar o ponto de troca.
        """
        self.magazine = set()
        for i, job in enumerate(self.jobs):
            temp_magazine = self.magazine.union(job['tools'])
            if len(temp_magazine) > self.capacity:
                return i
            else:
                self.magazine = temp_magazine
        return INT_INFINITY

    def _calculate_metrics(self, instance):
        """
        Método central que executa a simulação para esta máquina UMA VEZ
        e calcula TODAS as métricas (trocas, flowtime, makespan).
        """
        self.tool_switches = 0
        self.flowtime = 0
        self.makespan = 0
        elapsed_time = 0
        num_jobs = len(self.jobs)

        if num_jobs == 0:
            return

        # --- PRÉ-CÁLCULOS ---
        tools_distances = fill_tools_distances(self, instance.num_tools)
        start_switch_index = self._initialize_magazine_and_find_start_index()
        
        if start_switch_index == INT_INFINITY:
            start_switch_index = num_jobs

        # --- FASE 1: TAREFAS SEM CUSTO DE TROCA ---
        for i in range(start_switch_index):
            proc_time = self.tasks_cost[self.jobs[i]['id']]
            elapsed_time += proc_time
            self.flowtime += elapsed_time

        # --- FASE 2: TAREFAS COM CUSTO DE TROCA ---
        for i in range(start_switch_index, num_jobs):
            tools_for_current_job = self.jobs[i]['tools']
            next_magazine_base = tools_for_current_job.copy()
            empty_slots = self.capacity - len(next_magazine_base)
            removal_candidates = self.magazine - tools_for_current_job
            dist_tool_pairs = sorted([(tools_distances[tool, i], tool) for tool in removal_candidates])
            
            num_switches = max(0, len(dist_tool_pairs) - empty_slots)
            self.tool_switches += num_switches # Atualiza contador de trocas

            proc_time = self.tasks_cost[self.jobs[i]['id']]
            time_for_this_step = (num_switches * self.tool_change_cost) + proc_time
            
            elapsed_time += time_for_this_step
            self.flowtime += elapsed_time # Atualiza flowtime

            tools_to_keep = {pair[1] for pair in dist_tool_pairs[:empty_slots]}
            self.magazine = next_magazine_base.union(tools_to_keep)
        
        # O makespan da máquina é o tempo total decorrido no final
        self.makespan = elapsed_time