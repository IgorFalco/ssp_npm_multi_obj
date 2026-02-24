# SSP-NPM Multi-Objective Optimization

This repository contains implementations for solving the **Sequence Scheduling Problem on Non-Parallel Machines (SSP-NPM)** as a multi-objective combinatorial optimization problem. Three independent solution approaches are provided: a pure Python object-oriented meta-heuristic (`POO`), a JIT-compiled version using Numba (`Numba`), and an exact method based on Mixed Integer Programming using the Gurobi solver (`Gurobi`).

---

## 1. Problem Description

### 1.1 Setting

The SSP-NPM is a job-scheduling problem defined over:

| Symbol | Meaning |
|--------|---------|
| `M` | Set of machines (`m = 1, …, |M|`) |
| `J` | Set of jobs (`j = 1, …, |J|`) |
| `T` | Set of tools (`t = 1, …, |T|`) |
| `C_m` | Magazine capacity of machine `m` (maximum number of tools that can be loaded simultaneously) |
| `sw_m` | Tool-change cost (time per tool switch) on machine `m` |
| `p_{j,m}` | Processing time of job `j` on machine `m` |
| `A_{t,j}` | Binary tool-requirement matrix: 1 if job `j` requires tool `t`, 0 otherwise |

Each job must be assigned to **exactly one** machine and processed in a deterministic sequence. Before executing a job, the machine must have all required tools loaded in its magazine. Because the magazine has a limited capacity, tools must be swapped in and out as the sequence progresses, incurring a setup cost.

### 1.2 Objectives

All three objectives are **minimized** simultaneously:

1. **TS – Total Tool Switches**: the sum of tool insertions across all machines over the entire schedule.
2. **FMAX – Makespan**: the maximum completion time across all machines (i.e., `max_m C_m`).
3. **TFT – Total Flow Time**: the sum of all job completion times across all machines.

Because these objectives are in conflict (reducing tool switches may increase makespan, etc.), the goal is to approximate the **Pareto-optimal front** — the set of solutions for which no objective can be improved without worsening another.

### 1.3 Instance Format

Instances are stored as semicolon-separated CSV files (`src/instances/SSP-NPM-I/` and `src/instances/SSP-NPM-II/`). The structure is:

```
Row 0:   num_machines ; num_jobs ; num_tools
Row 1:   C_1 ; C_2 ; … ; C_M          (magazine capacities)
Row 2:   sw_1 ; sw_2 ; … ; sw_M       (tool-change costs)
Rows 3…3+M-1:  p_{1,m} ; p_{2,m} ; … ; p_{J,m}  (processing times, one row per machine)
Rows 3+M…end: A_{t,1} ; A_{t,2} ; … ; A_{t,J}  (tool-requirement matrix, one row per tool)
```

Instance filenames encode the size: `ins{id}_m={M}_j={J}_t={T}_var={variant}.csv`.

---

## 2. Solution Approaches

### 2.1 POO — Object-Oriented Iterated Local Search (ILS)

**Entry point:** `src/POO/main.py`

This implementation models the problem using Python classes (`Instance`, `Machine`, `Solution`, `ParetoWall`) and solves it with an **Iterated Local Search (ILS)** meta-heuristic. The algorithm is executed 10 independent times; the Pareto front of each run is saved individually, and a combined plot is produced.

#### 2.1.1 Data Structures

- **`Instance`** (`models/instance.py`): Holds all problem data (machines, jobs, tools, requirements matrix). Pre-computes a **job-similarity matrix** `S[i,j]` = number of positions where jobs `i` and `j` agree (both require, or both do not require, the same tool). This matrix guides greedy construction.
- **`Machine`** (`models/machine.py`): Maintains the ordered job sequence and computes TS, flowtime, and makespan in a single simulation pass via `_calculate_metrics`. The simulation uses a **look-ahead distances array** (`fill_tools_distances`) to implement the KTNS (Keep Tool Needed Soonest) policy for deciding which tools to evict from the magazine.
- **`Solution`** (`models/solution.py`): Immutable snapshot of a job assignment with objective values. Supports Pareto dominance checks (`dominates`, `dominates_on_axes`).
- **`ParetoWall`** (`models/pareto_wall.py`): Bounded archive of non-dominated solutions. When the archive exceeds its maximum size, the solution with the smallest **crowding distance** is removed to maintain diversity.

#### 2.1.2 Initial Solution Construction

A greedy-randomized procedure (`Instance.construct_initial_solution`):

1. **Phase 1 – Seeding**: Jobs are shuffled randomly. Each machine receives the first eligible job from the shuffled list (eligibility: `|tools(j)| ≤ C_m`).
2. **Phase 2 – Greedy Completion**: Remaining jobs are assigned iteratively. The machine with the lowest normalized tool-switch rate (`TS / |jobs_on_machine|`) is selected as the target. The job most similar (by `S`) to the last job on that machine is appended.

#### 2.1.3 ILS Loop

```
Initialize ParetoWall archive (capacity = archive_size)
Generate initial_pop_size random solutions → add non-dominated ones to archive

For i = 1 … max_iterations:
    1. Select a random solution s* from the archive
    2. Perturb s*  →  s_p  (perturbation_insertion, strength k)
    3. Add s_p to archive  (diversity update)
    4. Apply VND to s_p  →  s_opt  (convergence step)
    5. Add s_opt to archive
```

**Perturbation** (`perturbation_insertion`): Moves `k` randomly chosen jobs between randomly chosen pairs of machines (checking eligibility), inserting each at a random position in the destination machine's sequence.

#### 2.1.4 Variable Neighborhood Descent (VND)

Starting from a solution, the VND cycles through five neighborhood structures in order, restarting from the first neighborhood whenever an improving neighbor is found:

| # | Neighborhood | Description |
|---|--------------|-------------|
| 1 | **Job Exchange** | Swap one job from machine `m1` with one from `m2` (inter-machine). Eligibility is checked. |
| 2 | **Swap** | Swap two jobs at different positions within the same machine (intra-machine). |
| 3 | **2-Opt** | Reverse a contiguous sub-sequence within the same machine. |
| 4 | **One-Block** | Move the first job of a tool-sharing block to the start of another block of the same tool (intra-machine). |
| 5 | **Insertion** | Remove one job from any position and re-insert it at any other position on the same or a different machine. |

Acceptance criterion: a neighbor is accepted only if it **Pareto-dominates** (on the two chosen objective axes) the current solution.

#### 2.1.5 Pareto Archive Management

After each addition attempt, dominated solutions are removed. When `|archive| > max_size`, the **crowding distance** of each solution is recomputed per objective (normalized by the objective range) and the most crowded solution (smallest distance) is discarded, preserving boundary solutions (distance = ∞).

---

### 2.2 Numba — JIT-Compiled ILS

**Entry point:** `src/Numba/main.py`

This version implements the same ILS algorithm as the POO variant, but replaces Python objects with NumPy arrays and accelerates the objective evaluation function (`calculate_solution_objectives` in `functions/input.py`) using **Numba's `@njit` decorator**, which compiles the function to native machine code on first call.

Key differences from the POO version:

- The instance is represented as a plain dictionary of NumPy arrays (no class hierarchy).
- Job assignments are encoded as a binary matrix `job_assignment[m, j] ∈ {0,1}` instead of lists.
- Tool magazine simulation uses simple array operations compatible with Numba's nopython mode.
- The `Solution` class (`models/solution.py`) wraps the NumPy assignment and calls the compiled evaluator at construction time.

The ILS loop, VND, and Pareto archive logic mirror the POO implementation.

---

### 2.3 Gurobi — Exact Epsilon-Constraint Method

**Entry point:** `src/Gurobi/main.py`

This approach generates the Pareto front **exactly** using a Mixed Integer Programming (MIP) formulation solved by Gurobi, combined with the **ε-constraint method**.

#### 2.3.1 MIP Formulation

**Decision variables:**

| Variable | Type | Meaning |
|----------|------|---------|
| `x[j,r,m]` | Binary | 1 if job `j` is assigned to position `r` on machine `m` |
| `v[t,r,m]` | Binary | 1 if tool `t` is present in the magazine at position `r` on machine `m` |
| `w[t,r,m]` | Binary | 1 if tool `t` is inserted (switched in) at position `r` on machine `m` |
| `f[j,r,m]` | Continuous ≥ 0 | Completion time of job `j` at position `r` on machine `m` |
| `FMAX` | Continuous ≥ 0 | Makespan (maximum completion time across all machines) |

**Objective expressions:**
- `TS = Σ_{t,r,m} w[t,r,m]`
- `TFT = Σ_{j,r,m} f[j,r,m]`
- `FMAX` (variable)

**Constraints:**
1. **Assignment**: each job assigned to exactly one (machine, position) pair.
2. **Position uniqueness**: at most one job per (position, machine).
3. **Sequence compactness**: position `r` can be used only if position `r-1` is used.
4. **Tool requirement**: if job `j` (requiring tool `t`) occupies position `r` on machine `m`, then `v[t,r,m] = 1`.
5. **Magazine capacity**: `Σ_t v[t,r,m] ≤ C_m` for all `r, m`.
6. **Tool insertion**: `w[t,r,m] ≥ v[t,r,m] - v[t,r-1,m]` (tool inserted when newly loaded).
7. **Completion time – first position**: `f[j,1,m] = p_{j,m} · x[j,1,m]`.
8. **Completion time – subsequent positions** (Big-M linearization): `f[j,r,m] ≥ Σ_{i≠j} f[i,r-1,m] + sw_m · Σ_t w[t,r,m] + p_{j,m} · x[j,r,m] - G·(1 - x[j,r,m])`.
9. **Makespan**: `FMAX ≥ f[j,r,m]` for all `j, r, m`.

The Big-M constant `G` is computed as `Σ_j max_m(p_{j,m}) + (|J|-1) · max_m(sw_m · C_m) + 10^6`.

Gurobi parameters: `MIPGap = 0.001`, `TimeLimit` per solve (default 45 s), aggressive presolve and cuts, all available CPU cores.

#### 2.3.2 ε-Constraint Method

The `EpsilonConstraintMethod.generate_pareto_front_fast` procedure:

1. **Anchor solutions**: solve the MIP once per objective (minimizing TS, FMAX, and TFT individually) to obtain the extreme points of the Pareto front and bound the objective ranges.
2. **Intermediate points**: for a 2-objective pair (e.g., TS vs FMAX), linearly space `√(num_points)` values of the secondary objective between its minimum and maximum. For each value `ε`, add the constraint `secondary_obj ≤ ε` and minimize the primary objective.
3. **Non-dominance filtering**: each new solution is added to the set only if it is not dominated by any existing solution; existing dominated solutions are removed.
4. **Objective verification**: each solution's objectives are independently verified using the Numba evaluator for cross-checking against Gurobi's reported values.

Results are saved as a CSV (Pareto front), a PNG plot, and a text summary. An aggregated experiment log (`experimentos_gurobi.csv`) accumulates results across multiple runs.

---

## 3. Pareto Front Visualization

Both the POO and Numba variants produce:
- Per-run CSV files and 2D scatter/line plots of the Pareto front.
- A **combined plot** overlaying the Pareto fronts of all independent runs (`pareto_front_combined.png`), enabling visual assessment of convergence and spread.

The Gurobi variant produces a 2D plot for the selected objective pair and a per-run CSV with full objective vectors.

---

## 4. Repository Structure

```
src/
├── instances/
│   ├── SSP-NPM-I/     # 160 benchmark instances (2–3 machines, 10–20 jobs/tools)
│   └── SSP-NPM-II/    # Additional benchmark instances
├── POO/
│   ├── main.py                    # ILS entry point (10 runs)
│   ├── models/
│   │   ├── instance.py            # Problem instance (similarity matrix, construction)
│   │   ├── machine.py             # Machine simulation (KTNS, metrics)
│   │   ├── solution.py            # Solution with Pareto dominance
│   │   └── pareto_wall.py         # Bounded Pareto archive + plots
│   └── functions/
│       ├── input.py               # CSV parser → Instance object
│       ├── evaluation.py          # Metric calculation, greedy helpers
│       ├── metaheuristics.py      # ILS orchestration
│       ├── local_search.py        # VND implementation
│       └── neighborhoods.py       # 5 neighborhood generators + perturbation
├── Numba/
│   ├── main.py                    # Numba ILS entry point (10 runs)
│   ├── models/                    # Solution and ParetoWall (NumPy-based)
│   └── functions/
│       ├── input.py               # CSV parser → dict + @njit evaluator
│       ├── ILS.py                 # ILS and multi-objective ILS
│       ├── metaheuristics.py      # Initial solution construction
│       ├── local_search.py        # VND (Numba-compatible)
│       └── neighborhoods.py       # Neighborhood generators (NumPy)
└── Gurobi/
    ├── main.py                    # Epsilon-constraint entry point
    ├── models/
    │   ├── solution.py            # GurobiSolution, ParetoFront
    │   └── pareto_wall.py         # 2D Pareto plot
    └── functions/
        ├── input.py               # CSV parser → dict + @njit evaluator (shared)
        └── epsilon_constraint.py  # MIP model + ε-constraint procedure
```

---

## 5. Dependencies

```
pandas
matplotlib
numba
gurobipy>=12.0,<13.0
pymoo
```

A valid Gurobi license is required to run the `Gurobi` module.

---

## 6. Hyperparameters Summary

| Parameter | POO / Numba | Description |
|-----------|-------------|-------------|
| `num_runs` | 10 | Independent ILS executions |
| `max_iterations` | 100 (POO) / 1000 (Numba) | ILS loop iterations per run |
| `initial_pop_size` | 50 | Random solutions for archive seeding |
| `archive_size` | 10 | Maximum Pareto archive size |
| `perturbation_strength` | 2 | Number of job moves per perturbation |

| Parameter | Gurobi | Description |
|-----------|--------|-------------|
| `time_limit` | 45 s | Per-solve time limit |
| `num_pareto_points` | 10 | Target number of Pareto points |
| `MIPGap` | 0.001 | Relative MIP optimality gap |
