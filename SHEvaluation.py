import sys
__all__ = [
    'constraint'
]


def mutual_weight(G, u, v, weight=None):
    try:
        a_uv = G[u][v].get(weight, 1)
    except KeyError:
        a_uv = 0
    try:
        a_vu = G[v][u].get(weight, 1)
    except KeyError:
        a_vu = 0
    return a_uv + a_vu


def normalized_mutual_weight(G, u, v, norm=sum, weight=None):
    scale = norm(mutual_weight(G, u, w, weight)
                 for w in G.neighbors(u))
    return 0 if scale == 0 else mutual_weight(G, u, v, weight) / scale

def constraint(G, nodes=None, weight=None):
    if nodes is None:
        nodes = G.nodes
    constraint = {}
    for v in nodes:
        # Constraint is not defined for isolated nodes
        neighbors_of_v = set(G.neighbors(v))
        if len(neighbors_of_v) == 0:
            constraint[v] = float('nan')
            continue
        constraint[v] = sum(local_constraint(G, v, n, weight)
                            for n in neighbors_of_v)
    return constraint


def local_constraint(G, u, v, weight=None):
    nmw = normalized_mutual_weight
    direct = nmw(G, u, v, weight=weight)
    indirect = sum(nmw(G, u, w, weight=weight) * nmw(G, w, v, weight=weight)
                   for w in set(G.neighbors(u)))
    return (direct + indirect) ** 2
