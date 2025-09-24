import os
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# ----------------------------
# Função para ler instância CSV
# ----------------------------
def read_problem_instance(file_path):
    df = pd.read_csv(file_path, header=None, sep=';')

    num_machines, num_jobs, num_tools = df.iloc[0, :3].dropna().astype(
        int).to_list()

    magazines_capacities = np.array(
        df.iloc[1, :num_machines].dropna().astype(int).to_list(), dtype=np.int64)

    tool_change_costs = np.array(
        df.iloc[2, :num_machines].dropna().astype(int).to_list(), dtype=np.int64)

    job_cost_per_machine = np.empty((num_machines, num_jobs), dtype=np.int64)
    for i in range(num_machines):
        job_cost_per_machine[i, :] = df.iloc[3 + i,
                                             :num_jobs].dropna().astype(int).to_list()

    tools_requirements_matrix = np.array(
        df.iloc[3 + num_machines:, :num_jobs].dropna(axis=1).astype(int).values,
        dtype=np.int64)

    tools_per_job = tools_requirements_matrix.sum(axis=0)

    return {
        "num_machines": num_machines,
        "num_jobs": num_jobs,
        "num_tools": num_tools,
        "magazines_capacities": magazines_capacities,
        "tool_change_costs": tool_change_costs,
        "job_cost_per_machine": job_cost_per_machine,
        "tools_requirements_matrix": tools_requirements_matrix,
        "tools_per_job": tools_per_job,
    }

# ----------------------------
# Função que monta e resolve o modelo
# ----------------------------
def build_and_solve_SSP_NPM(J_set, M_set, T_set, R_set,
                            p, sw, C, Tj,
                            objective='FMAX',
                            time_limit=600):

    J = list(J_set)
    M = list(M_set)
    T = list(T_set)
    R = list(R_set)

    # helper: jobs que usam cada ferramenta
    J_t = {t: [j for j in J if t in Tj.get(j, [])] for t in T}

    # constante grande G
    max_p_per_job = {j: max((p.get((j, m), 0) for m in M), default=0) for j in J}
    G1 = sum(max_p_per_job.values())
    max_swC = max((sw[m] * C[m] for m in M), default=0)
    G = int(G1 + (len(J) - 1) * max_swC + 1e6)

    model = gp.Model("SSP-NPM")
    model.Params.TimeLimit = time_limit

    # variáveis
    x = model.addVars(J, R, M, vtype=GRB.BINARY, name="x")
    v = model.addVars(T, R, M, vtype=GRB.BINARY, name="v")
    w = model.addVars(T, R, M, vtype=GRB.BINARY, name="w")
    f = model.addVars(J, R, M, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
    FMAX = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="FMAX")

    # (1) cada job uma vez
    model.addConstrs(
        (gp.quicksum(x[j, r, m] for r in R for m in M) == 1 for j in J),
        name="assign_once"
    )

    # (2) uma posição tem no máx um job
    model.addConstrs(
        (gp.quicksum(x[j, r, m] for j in J) <= 1 for r in R for m in M),
        name="pos_at_most_one"
    )

    # (3) precedência
    for r in R:
        if r == min(R): continue
        for m in M:
            model.addConstr(
                gp.quicksum(x[j, r, m] for j in J) <= gp.quicksum(x[j, r-1, m] for j in J),
                name=f"precedence_r{r}_m{m}"
            )

    # (4) ferramentas requeridas
    for t in T:
        for r in R:
            for m in M:
                jobs_requiring_t = J_t[t]
                if jobs_requiring_t:
                    model.addConstr(
                        gp.quicksum(x[j, r, m] for j in jobs_requiring_t) <= v[t, r, m],
                        name=f"tool_req_t{t}_r{r}_m{m}"
                    )

    # (5) capacidade
    model.addConstrs(
        (gp.quicksum(v[t, r, m] for t in T) <= C[m] for r in R for m in M),
        name="capacity"
    )

    # (6) inserções
    for t in T:
        for r in R:
            if r == min(R): continue
            for m in M:
                model.addConstr(
                    v[t, r, m] - v[t, r-1, m] <= w[t, r, m],
                    name=f"insert_t{t}_r{r}_m{m}"
                )

    # (7) tempo para primeira posição
    r_first = min(R)
    for j in J:
        for m in M:
            pjm = p.get((j, m), 0)
            model.addConstr(f[j, r_first, m] >= pjm * x[j, r_first, m])
            model.addConstr(f[j, r_first, m] <= pjm * x[j, r_first, m] + G*(1-x[j, r_first, m]))

    # força f=0 se não usado
    for j in J:
        for r in R:
            for m in M:
                model.addConstr(f[j, r, m] <= G * x[j, r, m])

    # (8) tempos subsequentes
    for r in R:
        if r == r_first: continue
        for j in J:
            for m in M:
                sum_prev = gp.quicksum(f[i, r-1, m] for i in J if i != j)
                pjm = p.get((j, m), 0)
                model.addConstr(
                    f[j, r, m] >= sum_prev + sw[m]*gp.quicksum(w[t, r, m] for t in T) + pjm*x[j, r, m] - G*(1-x[j, r, m]),
                    name=f"f_rec_j{j}_r{r}_m{m}"
                )

    # (13) FMAX >= f_jrm
    for j in J:
        for r in R:
            for m in M:
                model.addConstr(FMAX >= f[j, r, m])

    # Objetivos
    TFT_expr = gp.quicksum(f[j, r, m] for j in J for r in R for m in M)
    TS_expr = gp.quicksum(w[t, r, m] for t in T for r in R for m in M)

    if objective.upper() == 'FMAX':
        model.setObjective(FMAX, GRB.MINIMIZE)
    elif objective.upper() == 'TFT':
        model.setObjective(TFT_expr, GRB.MINIMIZE)
    elif objective.upper() == 'TS':
        model.setObjective(TS_expr, GRB.MINIMIZE)
    else:
        raise ValueError("Objective must be 'FMAX','TFT','TS'")

    model.optimize()
    return model, x, f, v, w, FMAX, TFT_expr, TS_expr

# ----------------------------
# Função adaptada da instância
# ----------------------------
def build_and_solve_from_instance(instance, objective='FMAX', time_limit=600):
    J = list(range(1, instance["num_jobs"]+1))
    M = list(range(1, instance["num_machines"]+1))
    T = list(range(1, instance["num_tools"]+1))
    R = J[:]  # posições

    p = {(j, m): instance["job_cost_per_machine"][m-1, j-1] for m in M for j in J}
    sw = {m: instance["tool_change_costs"][m-1] for m in M}
    C = {m: instance["magazines_capacities"][m-1] for m in M}
    Tj = {j: [t for t in T if instance["tools_requirements_matrix"][t-1, j-1] == 1] for j in J}

    return build_and_solve_SSP_NPM(J, M, T, R, p, sw, C, Tj,
                                   objective=objective, time_limit=time_limit)

# ----------------------------
# Execução principal
# ----------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    RESULTS_FILEPATH = os.path.join(BASE_DIR, "results")
    SSP_NPM_I_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-I")
    SSP_NPM_II_PATH = os.path.join(BASE_DIR, "../instances/SSP-NPM-II")
    instance_filename = "ins1_m=2_j=10_t=10_var=1.csv"
    instance = read_problem_instance(os.path.join(SSP_NPM_I_PATH, instance_filename))

    model, x, f, v, w, FMAX, TFT, TS = build_and_solve_from_instance(
        instance, objective="FMAX", time_limit=60
    )

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL, GRB.FEASIBLE):
        print("Status:", model.Status)
        print("Objective value:", model.ObjVal)
        for j in range(1, instance["num_jobs"]+1):
            for r in range(1, instance["num_jobs"]+1):
                for m in range(1, instance["num_machines"]+1):
                    if x[j, r, m].X > 0.5:
                        print(f"Job {j} na posição {r} da máquina {m}, f={f[j,r,m].X:.1f}")