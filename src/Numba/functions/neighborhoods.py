import numpy as np
from numba import njit
from numba.typed import List
from functions.metaheuristics import check_machine_eligibility


@njit
def find_tool_blocks_numba(job_assignment, machine_id, tool_id, tools_requirements_matrix):
    """
    Encontra blocos contíguos de tarefas que usam uma ferramenta específica
    Versão compatível com Numba
    
    Returns:
        numpy array: Array 2D onde cada linha é um bloco [start_pos, end_pos]
                    Se não há blocos, retorna array vazio (0, 2)
    """
    machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
    
    if len(machine_jobs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    
    # Array temporário para armazenar blocos (máximo possível = len(machine_jobs))
    temp_blocks = np.empty((len(machine_jobs), 2), dtype=np.int32)
    num_blocks = 0
    
    i = 0
    while i < len(machine_jobs):
        job_id = machine_jobs[i]
        if tools_requirements_matrix[tool_id, job_id] == 1:
            block_start = i
            while (i < len(machine_jobs) and 
                   tools_requirements_matrix[tool_id, machine_jobs[i]] == 1):
                i += 1
            temp_blocks[num_blocks, 0] = block_start
            temp_blocks[num_blocks, 1] = i - 1
            num_blocks += 1
        else:
            i += 1
    
    # Retorna apenas os blocos válidos
    return temp_blocks[:num_blocks]


def generate_job_exchange_neighbors_numba(job_assignment, magazines_capacities, tools_per_job):
    """
    Gera vizinhos trocando duas tarefas ENTRE máquinas diferentes
    Versão compatível com Numba (retorna lista em vez de yield)
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        magazines_capacities: numpy array com capacidades das máquinas
        tools_per_job: numpy array com número de ferramentas por tarefa
        
    Returns:
        list: Lista de numpy arrays representando vizinhos
    """
    neighbors = List()
    num_machines = job_assignment.shape[0]
    
    # Itera sobre todos os pares de máquinas
    for m1 in range(num_machines):
        for m2 in range(m1 + 1, num_machines):
            # Encontra tarefas atribuídas a cada máquina
            len_m1 = np.sum(job_assignment[m1] != -1)
            len_m2 = np.sum(job_assignment[m2] != -1)
            
            # Tenta trocar cada tarefa da máquina 1 com cada tarefa da máquina 2
            for pos1 in range(len_m1):
                for pos2 in range(len_m2):

                    job_id1 = job_assignment[m1, pos1]
                    job_id2 = job_assignment[m2, pos2]

                    # Verifica elegibilidade da troca
                    if (check_machine_eligibility(job_id2, m1, magazines_capacities, tools_per_job) and
                        check_machine_eligibility(job_id1, m2, magazines_capacities, tools_per_job)):

                        # Cria vizinho com a troca
                        neighbor = job_assignment.copy()
                        neighbor[m1, pos1], neighbor[m2, pos2] = job_id2, job_id1
                        neighbors.append(neighbor)

    return neighbors


@njit
def generate_swap_neighbors_numba(job_assignment):
    """
    Gera vizinhos trocando duas tarefas DENTRO da mesma máquina
    Versão compatível com Numba (retorna lista em vez de yield)
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        
    Returns:
        list: Lista de numpy arrays representando vizinhos
    """
    neighbors = List()
    num_machines = job_assignment.shape[0]
    
    for machine_id in range(num_machines):
        # Encontra tarefas atribuídas à máquina
        lem_m = np.sum(job_assignment[machine_id] != -1)
        
        if lem_m < 2:
            continue
            
        # Troca todos os pares possíveis
        for i in range(lem_m):
            for j in range(i + 1, lem_m):
                
                # Cria vizinho com a troca
                neighbor = job_assignment.copy()
                neighbor[machine_id, i], neighbor[machine_id, j] = (
                    neighbor[machine_id, j],
                    neighbor[machine_id, i],
                )
                
                neighbors.append(neighbor)
    
    return neighbors


@njit
def generate_two_opt_neighbors_numba(job_assignment):
    """
    Gera vizinhos aplicando 2-opt na sequência de tarefas de cada máquina
    Versão compatível com Numba (retorna lista em vez de yield)
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        
    Returns:
        list: Lista de numpy arrays representando vizinhos
    """
    neighbors = List()
    num_machines = job_assignment.shape[0]

    
    for machine_id in range(num_machines):
        # Encontra tarefas atribuídas à máquina
        len_m = np.sum(job_assignment[machine_id] != -1)

        jobs_seq = job_assignment[machine_id, :len_m]
        if len_m < 2:
            continue
            
        # Aplica 2-opt em diferentes segmentos
        for i in range(len_m):
            for j in range(i + 2, len_m + 1):
                # Copia a sequência e inverte o segmento
                new_seq = jobs_seq.copy()
                new_seq[i:j] = new_seq[i:j][::-1]
                
                # Constrói vizinho
                neighbor = job_assignment.copy()
                neighbor[machine_id, :len_m] = new_seq
                neighbor[machine_id, len_m:] = -1
                
                neighbors.append(neighbor)
    
    return neighbors


def generate_one_block_neighbors_numba(job_assignment, tools_requirements_matrix):
    """
    Gera vizinhos movendo o início de blocos de tarefas que compartilham ferramentas
    Versão compatível com Numba (retorna lista em vez de yield)
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        tools_requirements_matrix: numpy array (num_tools, num_jobs)
        
    Returns:
        list: Lista de numpy arrays representando vizinhos
    """
    neighbors = List()
    num_machines = job_assignment.shape[0]
    num_tools = tools_requirements_matrix.shape[0]
    
    for machine_id in range(num_machines):
        len_valid = np.sum(job_assignment[machine_id] != -1)
        assigned_jobs = job_assignment[machine_id, :len_valid]

        
        if len_valid == 0:
            continue
            
        # Para cada ferramenta, encontra blocos
        for tool_id in range(num_tools):
            blocks = find_tool_blocks_numba(job_assignment, machine_id, tool_id, tools_requirements_matrix)
            
            if len(blocks) < 2:
                continue
                
            # Tenta mover o primeiro job de um bloco para o início de outro
            for remove_block_idx in range(len(blocks)):
                for insert_block_idx in range(len(blocks)):
                    if remove_block_idx == insert_block_idx:
                        continue
                        
                    remove_start = blocks[remove_block_idx, 0]
                    insert_start = blocks[insert_block_idx, 0]
                    
                    if remove_start >= len(assigned_jobs) or insert_start >= len(assigned_jobs):
                        continue
                        
                    # Cria vizinho movendo o job
                    neighbor = job_assignment.copy()
                    
                    # Remove todas as tarefas da máquina
                    neighbor[machine_id, :] = -1
                    
                    # Reconstrói a sequência com o movimento
                    new_sequence = assigned_jobs.copy()
                    
                    # Move o job da posição remove_start para insert_start
                    job_to_move = new_sequence[remove_start]
                    
                    # Remove o job da posição original
                    if remove_start < len(new_sequence) - 1:
                        new_sequence[remove_start:-1] = new_sequence[remove_start+1:]
                    new_sequence = new_sequence[:-1]  # Remove último elemento
                    
                    # Ajusta posição de inserção se necessário
                    actual_insert_pos = insert_start if remove_start > insert_start else insert_start - 1
                    actual_insert_pos = max(0, min(actual_insert_pos, len(new_sequence)))
                    
                    # Insere o job na nova posição
                    temp_sequence = np.empty(len(new_sequence) + 1, dtype=new_sequence.dtype)
                    temp_sequence[:actual_insert_pos] = new_sequence[:actual_insert_pos]
                    temp_sequence[actual_insert_pos] = job_to_move
                    temp_sequence[actual_insert_pos+1:] = new_sequence[actual_insert_pos:]
                    new_sequence = temp_sequence
                    
                    neighbor[machine_id, :len(new_sequence)] = new_sequence
                    neighbor[machine_id, len(new_sequence):] = -1
                    
                    neighbors.append(neighbor)
    
    return neighbors


@njit  
def generate_insertion_neighbors_numba(job_assignment, magazines_capacities, tools_per_job):
    """
    Gera vizinhos movendo uma tarefa para diferentes posições/máquinas
    Versão compatível com Numba (retorna lista em vez de yield)
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        magazines_capacities: numpy array com capacidades das máquinas
        tools_per_job: numpy array com número de ferramentas por tarefa
        
    Returns:
        list: Lista de numpy arrays representando vizinhos
    """
    neighbors = List()
    num_machines = job_assignment.shape[0]
    
    for source_machine in range(num_machines):
        # pega os job_ids válidos da origem
        len_valid_src = np.sum(job_assignment[source_machine] != -1)
        if len_valid_src == 0:
            continue
        source_jobs = job_assignment[source_machine, :len_valid_src]
        
        for pos_in_src in range(len_valid_src):
            job_to_move = source_jobs[pos_in_src]
            
            for target_machine in range(num_machines):
                # Verifica elegibilidade
                if not check_machine_eligibility(job_to_move, target_machine, magazines_capacities, tools_per_job):
                    continue
                
                len_valid_tgt = np.sum(job_assignment[target_machine] != -1)
                
                # tenta inserir em todas as posições possíveis
                for insert_pos in range(len_valid_tgt):
                    # evita movimento trivial (mesma máquina, mesma posição)
                    if source_machine == target_machine:
                        if insert_pos == pos_in_src or insert_pos == pos_in_src + 1:
                            continue
                    
                    neighbor = job_assignment.copy()
                    
                    # === remove o job da origem ===
                    src_seq = neighbor[source_machine][neighbor[source_machine] != -1]
                    new_src_seq = np.empty(len(src_seq) - 1, dtype=src_seq.dtype)
                    k = 0
                    for x in src_seq:
                        if x != job_to_move:
                            new_src_seq[k] = x
                            k += 1
                    neighbor[source_machine, :] = -1
                    neighbor[source_machine, :len(new_src_seq)] = new_src_seq
                    
                    # === insere o job no destino ===
                    tgt_seq = neighbor[target_machine][neighbor[target_machine] != -1]
                    new_tgt_seq = np.empty(len(tgt_seq) + 1, dtype=tgt_seq.dtype)
                    
                    new_tgt_seq[:insert_pos] = tgt_seq[:insert_pos]
                    new_tgt_seq[insert_pos] = job_to_move
                    new_tgt_seq[insert_pos+1:] = tgt_seq[insert_pos:]
                    
                    neighbor[target_machine, :] = -1
                    neighbor[target_machine, :len(new_tgt_seq)] = new_tgt_seq
                    
                    neighbors.append(neighbor)
    
    return neighbors


@njit
def perturbation_insertion_numba(job_assignment, magazines_capacities, tools_per_job, strength=1):
    """
    Perturba uma solução movendo 'strength' tarefas entre máquinas aleatórias
    Versão compatível com Numba
    
    Args:
        job_assignment: numpy array (num_machines, num_jobs) com -1 ou 1
        magazines_capacities: numpy array com capacidades das máquinas  
        tools_per_job: numpy array com número de ferramentas por tarefa
        strength: número de movimentos a realizar
        
    Returns:
        numpy array: Nova atribuição perturbada
    """
    neighbor = job_assignment.copy()
    num_machines = job_assignment.shape[0]
    
    for _ in range(strength):
        # Escolhe duas máquinas diferentes aleatoriamente
        # Substituindo np.random.choice por alternativa compatível com Numba
        machine_indices = np.arange(num_machines)
        np.random.shuffle(machine_indices)
        source_machine = machine_indices[0]
        target_machine = machine_indices[1]
        
        # Encontra tarefas na máquina origem
        source_jobs = np.where(neighbor[source_machine] != -1)[0]
        
        if len(source_jobs) == 0:
            continue
            
        # Escolhe uma tarefa aleatória para mover
        job_idx = np.random.randint(0, len(source_jobs))
        job_pos = source_jobs[job_idx]
        job_id = neighbor[source_machine, job_pos]
        
        # Verifica elegibilidade
        if check_machine_eligibility(job_id, target_machine, magazines_capacities, tools_per_job):
            # Encontra posição livre na máquina destino
            free_pos = np.where(neighbor[target_machine] == -1)[0]
            if free_pos.size > 0:
                # Move o job_id
                neighbor[source_machine, job_pos:-1] = neighbor[source_machine, job_pos+1:]
                neighbor[source_machine, -1] = -1
                neighbor[target_machine, free_pos[0]] = job_id
    
    return neighbor


def get_all_neighborhood_generators():
    """
    Retorna lista com todos os geradores de vizinhança disponíveis
    
    Returns:
        list: Lista de funções geradoras de vizinhança
    """
    return [
        generate_job_exchange_neighbors_numba,
        generate_swap_neighbors_numba, 
        generate_two_opt_neighbors_numba,
        generate_one_block_neighbors_numba,
        generate_insertion_neighbors_numba
    ]
