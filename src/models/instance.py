import numpy as np
import random
from functions.evaluation import find_best_machine_min_tsj, find_most_similar_job

class Instance:

    def __init__(self, name, machines, num_jobs, num_tools, tools_requirements_matrix):
        self.name = name
        self.num_jobs = num_jobs
        self.num_tools = num_tools
        self.machines = machines
        self.tools_requirements_matrix = tools_requirements_matrix  # Matriz tools x jobs

        # Atributos que serão criados por métodos
        self.jobs = []
        self.similarity_matrix = None

        # Chamadas dos métodos internos
        self._initialize_jobs()
        self._construct_similarity_matrix()


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

    def _construct_similarity_matrix(self):
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

    def construct_initial_solution(self):
        jobs_list = list(range(self.num_jobs))
        random.shuffle(jobs_list)

        # --- FASE 1: ATRIBUIÇÃO INICIAL  ---
        for machine in self.machines:
            machine.jobs = []
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

    def copy_with_new_assignment(self, new_assignment):
        """
        Cria uma cópia da instância, mas com uma nova atribuição de tarefas.
        Muito mais rápido do que ler o arquivo novamente.
        """
        import copy
        new_instance = copy.copy(self) # Cópia superficial da instância
        new_instance.machines = copy.deepcopy(self.machines) # Cópia profunda das máquinas

        # Atualiza as listas de tarefas das máquinas na nova instância
        for m in new_instance.machines:
            m.jobs = [self.jobs[job_id] for job_id in new_assignment[m.id]]
        
        return new_instance
    
    def get_job_by_id(self, job_id):
        """
        Retorna o dicionário do objeto job correspondente a um ID.
        
        Args:
            job_id (int): O ID da tarefa a ser encontrada.
            
        Returns:
            dict: O dicionário da tarefa com 'id' e 'tools'.
        """
        # Graças à forma como a lista self.jobs é construída, o ID da tarefa
        # corresponde diretamente ao seu índice na lista.
        # Adicionamos uma verificação para garantir que o ID é válido.
        if 0 <= job_id < self.num_jobs:
            return self.jobs[job_id]
        else:
            # Retorna None ou lança um erro se o ID for inválido.
            raise ValueError(f"ID de tarefa inválido: {job_id}. Deve estar entre 0 e {self.num_jobs - 1}.")

    def get_machine_by_id(self, machine_id):
        """
        Retorna o objeto Machine correspondente a um ID.
        """
        # Assumindo que os IDs da máquina (1, 2, ...) correspondem aos índices (0, 1, ...)
        # Se os IDs começam em 1, a busca seria: self.machines[machine_id - 1]
        # Pelo seu código de leitura, o ID é o próprio índice, então a busca é direta.
        if 0 <= machine_id < len(self.machines):
            return self.machines[machine_id]
        else:
            raise ValueError(f"ID de máquina inválido: {machine_id}.")