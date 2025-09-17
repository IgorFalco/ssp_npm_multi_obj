import random
import copy

# ---------------------------------------------------------------------------
# NOTA: Todas as funções a seguir são "geradoras" (usam 'yield').
# Elas produzem um vizinho de cada vez para economizar memória.
# Elas sempre retornam uma CÓPIA PROFUNDA da atribuição de tarefas,
# garantindo que a solução original nunca seja modificada.
# ---------------------------------------------------------------------------

def generate_job_exchange_neighbors(solution, instance):
    """
    Gera vizinhos trocando duas tarefas (jobs) ENTRE máquinas diferentes.
    Corresponde a 'jobExchangeLocalSearchFull' (Vizinhança 1).
    
    Args:
        solution (Solution): O objeto da solução original.
        instance (Instance): A instância do problema para checar a elegibilidade.
        
    Yields:
        dict: Uma nova atribuição de tarefas (dicionário) representando um vizinho.
    """
    base_assignment = solution.assignment
    machine_ids = list(base_assignment.keys())

    if len(machine_ids) < 2:
        return  # Precisa de pelo menos duas máquinas

    # Itera sobre todos os pares de máquinas
    for i in range(len(machine_ids)):
        for j in range(i + 1, len(machine_ids)):
            m1_id = machine_ids[i]
            m2_id = machine_ids[j]

            # Itera sobre todas as tarefas da máquina 1
            for job1_idx in range(len(base_assignment[m1_id])):
                # Itera sobre todas as tarefas da máquina 2
                for job2_idx in range(len(base_assignment[m2_id])):
                    
                    job1_id = base_assignment[m1_id][job1_idx]
                    job2_id = base_assignment[m2_id][job2_idx]

                    # Checa se a troca é válida (elegibilidade)
                    job1_obj = instance.get_job_by_id(job1_id)
                    job2_obj = instance.get_job_by_id(job2_id)
                    m1_obj = instance.get_machine_by_id(m1_id)
                    m2_obj = instance.get_machine_by_id(m2_id)

                    if m1_obj.check_eligibility(job2_obj) and m2_obj.check_eligibility(job1_obj):
                        neighbor_assignment = copy.deepcopy(base_assignment)
                        # Realiza a troca
                        neighbor_assignment[m1_id][job1_idx] = job2_id
                        neighbor_assignment[m2_id][job2_idx] = job1_id
                        yield neighbor_assignment


def generate_swap_neighbors(solution):
    """
    Gera vizinhos trocando duas tarefas (jobs) DENTRO da mesma máquina.
    Corresponde a 'swapLocalSearch' (Vizinhança 2).

    Args:
        solution (Solution): O objeto da solução original.
        
    Yields:
        dict: Uma nova atribuição de tarefas representando um vizinho.
    """
    base_assignment = solution.assignment
    
    for machine_id, jobs in base_assignment.items():
        if len(jobs) < 2:
            continue
        # Itera sobre todos os pares de tarefas a serem trocados
        for i in range(len(jobs)):
            for j in range(i + 1, len(jobs)):
                neighbor_assignment = copy.deepcopy(base_assignment)
                # Realiza a troca
                neighbor_assignment[machine_id][i], neighbor_assignment[machine_id][j] = \
                    neighbor_assignment[machine_id][j], neighbor_assignment[machine_id][i]
                
                yield neighbor_assignment


def generate_two_opt_neighbors(solution):
    """
    Gera vizinhos aplicando o movimento 2-opt na sequência de tarefas
    dentro de cada máquina.
    Corresponde a 'twoOptLocalSearch' (Vizinhança 3).

    Args:
        solution (Solution): O objeto da solução original.
        
    Yields:
        dict: Uma nova atribuição de tarefas representando um vizinho.
    """
    base_assignment = solution.assignment
    
    for machine_id, jobs in base_assignment.items():
        if len(jobs) < 2:
            continue
            
        # Itera sobre todos os pontos de início (i) e fim (j) da sub-sequência
        for i in range(len(jobs)):
            for j in range(i + 2, len(jobs) + 1):
                neighbor_assignment = copy.deepcopy(base_assignment)
                
                sub_list_to_reverse = neighbor_assignment[machine_id][i:j]
                sub_list_to_reverse.reverse()
                neighbor_assignment[machine_id][i:j] = sub_list_to_reverse
                
                yield neighbor_assignment


def _find_tool_blocks(assignment, machine_id, tool_id, instance):
    """
    Função auxiliar para encontrar blocos contíguos de tarefas que usam uma ferramenta.
    """
    blocks = []
    job_sequence = assignment[machine_id]
    i = 0
    while i < len(job_sequence):
        job_id = job_sequence[i]
        if tool_id in instance.jobs[job_id]['tools']:
            block_start = i
            while i < len(job_sequence) and tool_id in instance.jobs[job_sequence[i]]['tools']:
                i += 1
            blocks.append((block_start, i - 1))
        else:
            i += 1
    return blocks

def generate_one_block_neighbors(solution, instance):
    """
    Gera vizinhos movendo o início de um bloco de tarefas (que compartilham
    uma ferramenta) para o início de outro bloco.
    Corresponde a 'oneBlockLocalSearch' (Vizinhança 4).

    Args:
        solution (Solution): O objeto da solução original.
        instance (Instance): A instância do problema.
        
    Yields:
        dict: Uma nova atribuição de tarefas representando um vizinho.
    """
    base_assignment = solution.assignment
    
    for machine_id in base_assignment.keys():
        # Itera sobre todas as ferramentas para encontrar blocos
        tool_ids = list(range(instance.num_tools))
        # 2. Embaralha a lista para iterar em ordem aleatória
        random.shuffle(tool_ids)
        for tool_id in tool_ids:
            blocks = _find_tool_blocks(base_assignment, machine_id, tool_id, instance)
            
            if len(blocks) < 2:
                continue

            # Itera sobre todos os pares de blocos
            for remove_idx in range(len(blocks)):
                for insert_idx in range(len(blocks)):
                    if remove_idx == insert_idx:
                        continue
                    
                    remove_block_start, _ = blocks[remove_idx]
                    insert_block_start, _ = blocks[insert_idx]
                    
                    neighbor_assignment = copy.deepcopy(base_assignment)
                    
                    # Remove o primeiro job do bloco de origem
                    job_to_move_id = neighbor_assignment[machine_id].pop(remove_block_start)
                    
                    # O índice de inserção pode ter mudado após a remoção
                    # Se o item foi removido de uma posição anterior à de inserção
                    actual_insert_pos = insert_block_start if remove_block_start > insert_block_start else insert_block_start - 1
                    
                    neighbor_assignment[machine_id].insert(actual_insert_pos, job_to_move_id)
                    
                    yield neighbor_assignment

def generate_insertion_neighbors(solution, instance):
    """
    Gera vizinhos movendo uma tarefa de uma posição para outra
    (na mesma máquina ou em outra).
    Corresponde a 'jobInsertionLocalSearch'.
    """
    base_assignment = solution.assignment
    machine_ids = list(base_assignment.keys())

    for source_machine_id in machine_ids:
        if not base_assignment[source_machine_id]:
            continue
        for job_idx_in_source in range(len(base_assignment[source_machine_id])):
            
            for target_machine_id in machine_ids:
                # Cria uma cópia temporária com a tarefa já removida
                temp_assignment = copy.deepcopy(base_assignment)
                job_to_move_id = temp_assignment[source_machine_id].pop(job_idx_in_source)
                job_to_move_obj = instance.get_job_by_id(job_to_move_id)

                target_machine_obj = instance.get_machine_by_id(target_machine_id)
                if not target_machine_obj.check_eligibility(job_to_move_obj):
                    continue

                # Itera sobre todas as posições de inserção possíveis
                for insert_pos in range(len(temp_assignment[target_machine_id]) + 1):
                    neighbor_assignment = copy.deepcopy(temp_assignment)
                    neighbor_assignment[target_machine_id].insert(insert_pos, job_to_move_id)
                    yield neighbor_assignment

def perturbation_insertion(solution, instance, strength=1):
    """
    Perturba uma solução movendo um número 'strength' de tarefas
    entre máquinas aleatórias.
    Corresponde a 'jobInsertionDisturb'.
    """
    # Trabalha com uma cópia para não modificar a solução original
    neighbor_assignment = copy.deepcopy(solution.assignment)
    machine_ids = list(neighbor_assignment.keys())

    if len(machine_ids) < 2:
        return neighbor_assignment # Não é possível perturbar com menos de 2 máquinas

    for _ in range(strength):
        # Escolhe duas máquinas diferentes aleatoriamente
        source_id, target_id = random.sample(machine_ids, 2)

        # Se a máquina de origem estiver vazia, não há o que mover
        if not neighbor_assignment[source_id]:
            continue
            
        # Escolhe uma tarefa aleatória da máquina de origem
        job_idx_in_source = random.randrange(len(neighbor_assignment[source_id]))
        job_id_to_move = neighbor_assignment[source_id][job_idx_in_source]
        job_obj_to_move = instance.get_job_by_id(job_id_to_move)

        # Verifica se a máquina de destino é elegível
        target_machine_obj = instance.get_machine_by_id(target_id)
        if target_machine_obj.check_eligibility(job_obj_to_move):
            
            # Remove a tarefa da origem
            neighbor_assignment[source_id].pop(job_idx_in_source)
            
            # Insere em uma posição aleatória no destino
            insert_pos = random.randint(0, len(neighbor_assignment[target_id]))
            neighbor_assignment[target_id].insert(insert_pos, job_id_to_move)

    return neighbor_assignment
