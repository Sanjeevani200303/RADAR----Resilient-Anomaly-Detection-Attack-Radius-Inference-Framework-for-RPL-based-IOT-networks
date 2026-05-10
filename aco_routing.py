"""
aco_routing.py — Full Ant Colony Optimization routing on RPL graph topology.

Improvements over v1:
  - Real adjacency graph built from parent-child hop relationships
    (hop_count n → hop_count n+1 edges), not a fully-connected free-for-all.
  - Path score = MIN trust along path (weakest-link), not mean — one
    compromised node cannot hide behind trusted neighbours.
  - Route output preserves ACO-discovered hop order (not re-sorted by trust).
  - Early-stopping when best path score stops improving.
  - Convergence delta and patience are configurable.
  - Hyperparameters documented with rationale.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_ANTS        = 20     # ant population per iteration
N_ITERATIONS  = 50     # maximum ACO iterations
ALPHA         = 1.5    # pheromone influence weight (τ^α)
BETA          = 2.0    # heuristic influence weight (η^β)
RHO           = 0.3    # pheromone evaporation rate ∈ (0,1)
Q             = 1.0    # pheromone deposit constant
MIN_PHEROMONE = 0.01   # floor — prevents pheromone stagnation
MAX_PATH_LEN  = 6      # max hops per ant path
PATIENCE      = 8      # early-stop: iterations with no improvement
CONV_DELTA    = 1e-4   # min improvement to count as progress


def _build_graph(df):
    """
    Build an adjacency list from RPL hop topology.

    Strategy:
      - Nodes at hop h can forward to nodes at hop h-1 (toward root).
      - We link every node at hop h to every node at hop h-1 as potential
        parents (conservative: real RPL chooses best OF0 parent, but we
        allow ACO to discover the best).
      - Nodes without hop_count get a default hop level of 3.
      - Returns: adj dict {node_idx: [neighbour_idx, ...]}
    """
    adj = defaultdict(list)
    node_ids = df['mote_id'].values

    if 'hop_count' not in df.columns:
        # Fallback: fully connected (original behaviour)
        n = len(node_ids)
        for i in range(n):
            adj[i] = [j for j in range(n) if j != i]
        return adj

    hop_to_nodes = defaultdict(list)
    for i, row in df.iterrows():
        hop_to_nodes[int(row['hop_count'])].append(i)

    for hop, nodes_at_hop in hop_to_nodes.items():
        parents_hop = hop - 1
        if parents_hop in hop_to_nodes:
            for child_idx in nodes_at_hop:
                for parent_idx in hop_to_nodes[parents_hop]:
                    adj[child_idx].append(parent_idx)
                    adj[parent_idx].append(child_idx)  # bidirectional

    # Nodes with no edges get connected to same-hop peers as fallback
    for i in range(len(node_ids)):
        if not adj[i]:
            same_hop_peers = hop_to_nodes.get(
                int(df.iloc[i].get('hop_count', 3)), []
            )
            adj[i] = [j for j in same_hop_peers if j != i] or [
                j for j in range(len(node_ids)) if j != i
            ]

    return adj


def _build_heuristic(node_row):
    """
    Heuristic η for a node = desirability as next hop.
    Higher trust + lower hop cost + safe cluster = better.
    Safe-cluster boost is 1.3× (a conservative, tunable multiplier).
    """
    trust = float(node_row['trust_score'])
    hop   = float(node_row.get('hop_count', 3)) + 1.0
    safe  = 1.3 if node_row.get('is_safe_cluster', False) else 1.0
    return (trust * safe) / hop


def run_aco(df):
    """
    Returns:
      result      — DataFrame of best-path nodes in hop order
      graph_data  — dict with everything needed to draw the network:
                    {
                      'candidates': DataFrame,
                      'adj':        adjacency dict {idx: [idx,...]},
                      'tau':        final pheromone matrix (n×n ndarray),
                      'best_path':  list of node indices in hop order,
                      'best_score': float,
                      'iter_scores': list of per-iteration best scores,
                    }
    """
    print("[INFO] Running ACO-based secure routing on RPL topology graph...")

    candidates = df[
        (df['trust_score'] > 0.1) & (df['anomaly'] == 0)
    ].drop_duplicates(subset=['mote_id']).reset_index(drop=True).copy()

    if len(candidates) < 2:
        print("[WARNING] Not enough trusted nodes for ACO routing.")
        return pd.DataFrame(), {}

    n_nodes = len(candidates)
    adj     = _build_graph(candidates)

    tau = np.full((n_nodes, n_nodes), MIN_PHEROMONE)
    eta = np.array([_build_heuristic(candidates.iloc[i]) for i in range(n_nodes)])

    best_path       = []
    best_path_score = -np.inf
    no_improve      = 0
    iter_scores     = []          # track convergence curve

    for iteration in range(N_ITERATIONS):
        all_paths  = []
        all_scores = []

        for _ in range(N_ANTS):
            start   = np.random.randint(n_nodes)
            path    = [start]
            visited = {start}

            for _ in range(min(MAX_PATH_LEN - 1, n_nodes - 1)):
                current    = path[-1]
                neighbours = [j for j in adj[current] if j not in visited]
                if not neighbours:
                    break

                scores = np.array([
                    (tau[current, j] ** ALPHA) * (eta[j] ** BETA)
                    for j in neighbours
                ])
                total = scores.sum()
                if total == 0:
                    break

                probs     = scores / total
                next_node = neighbours[np.random.choice(len(neighbours), p=probs)]
                path.append(next_node)
                visited.add(next_node)

            path_score = float(np.min([
                candidates.iloc[i]['trust_score'] for i in path
            ]))
            all_paths.append(path)
            all_scores.append(path_score)

            if path_score > best_path_score:
                best_path_score = path_score
                best_path       = list(path)

        iter_scores.append(best_path_score)

        tau *= (1.0 - RHO)
        tau  = np.clip(tau, MIN_PHEROMONE, None)

        for path, score in zip(all_paths, all_scores):
            deposit = Q * score
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                if adj[i] and j in adj[i]:
                    tau[i, j] += deposit
                    tau[j, i] += deposit

        if iteration > 0:
            improvement = best_path_score - getattr(run_aco, '_prev_score', -np.inf)
            no_improve  = no_improve + 1 if improvement < CONV_DELTA else 0
        run_aco._prev_score = best_path_score

        if no_improve >= PATIENCE:
            print(f"[INFO] ACO converged at iteration {iteration + 1}/{N_ITERATIONS}")
            break

    if not best_path:
        print("[WARNING] ACO found no path.")
        return pd.DataFrame(), {}

    route_rows = candidates.iloc[best_path].copy()
    route_rows['aco_rank'] = range(1, len(route_rows) + 1)

    cols = ['mote_id', 'trust_score', 'aco_rank']
    if 'hop_count' in route_rows.columns:
        cols.insert(2, 'hop_count')
    if 'cluster' in route_rows.columns:
        cols.append('cluster')
    result = route_rows[cols].copy()

    graph_data = {
        'candidates':  candidates,
        'adj':         dict(adj),
        'tau':         tau,
        'best_path':   best_path,
        'best_score':  best_path_score,
        'iter_scores': iter_scores,
    }

    print(f"[RESULT] ACO Best Path Score (min-trust): {best_path_score:.4f}")
    print(result.to_string(index=False))
    return result, graph_data
