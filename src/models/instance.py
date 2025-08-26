import numpy as np
import random
from functions.evaluation import find_best_machine_min_tsj, find_most_similar_job

class Instance:

    def __init__(self, machines, num_jobs, num_tools, tools_requirements_matrix, params):
        self.num_jobs = num_jobs
        self.num_tools = num_tools
        self.machines = machines
        self.tools_requirements_matrix = tools_requirements_matrix  # Matriz tools x jobs

        # Atributos que serão criados por métodos
        self.jobs = []
        self.similarity_matrix = None

        # Chamadas dos métodos internos
        self._initialize_jobs()
        self._construct_similarity_matrix(params["similarity_percentage"])
        self._construct_initial_solution()


    def __repr__(self):
        return (f"ProblemInstance(Machines: {len(self.machines)}, Jobs: {self.num_jobs}, "
                f"Tools: {self.num_tools})")

    def _initialize_jobs(self):
        self.jobs = [
            {
                'id': job_idx, 
                'tools': set(np.where(self.tools_requirements_matrix[:, job_idx] == 1)[0])
            }
            for job_idx in range(self.num_jobs)
        ]

    def _construct_similarity_matrix(self, similarity_percentage):
        # Cria a matriz de similiradidade multiplicando a matriz de requisitos pela sua 
        matrix_T = self.tools_requirements_matrix.T

        # Passo 1: Conta o número de requisitos "1" compartilhados entre todos os pares
        # (Ex: se ambos os jobs precisam da ferramenta, a soma é 1)
        shared_ones = np.dot(matrix_T, self.tools_requirements_matrix)

        # Passo 2: Inverte a matriz (0s viram 1s) para contar os requisitos "0" compartilhados
        inverted_matrix = 1 - self.tools_requirements_matrix
        shared_zeros = np.dot(inverted_matrix.T, inverted_matrix)

        # Passo 3: Soma os dois para obter o total de correspondências (1s e 0s)
        self.similarity_matrix = shared_ones + shared_zeros

    def _construct_initial_solution(self):
        jobs_list = list(range(self.num_jobs))
        random.shuffle(jobs_list)

        # --- FASE 1: ATRIBUIÇÃO INICIAL  ---
        for machine in self.machines:
            for job_id in jobs_list[:]:
                if machine.check_eligibility(self.jobs[job_id]):
                    machine.add_job(self.jobs[job_id])
                    jobs_list.remove(job_id)
                    break 
        
        # --- FASE 2: ALOCAÇÃO GULOSA (TAREFAS RESTANTES) ---
        while jobs_list:
            target_machine = find_best_machine_min_tsj(self.machines, self)
            most_similar_job_id = find_most_similar_job(target_machine, jobs_list, self)

            if most_similar_job_id is None:
                print("Aviso: Não foi possível alocar todas as tarefas restantes.")
                break

            job_to_add = self.jobs[most_similar_job_id]
            target_machine.add_job(job_to_add)
            jobs_list.remove(most_similar_job_id)
