from copy import deepcopy
import random
import networkx as nx


def uniformWeights(G):
    '''
    Every incoming edge of v with degree dv has weight 1/dv.
    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        dv = sum([edata['weight'] for v1, v2, edata in in_edges])
        for v1, v2, _ in in_edges:
            Ew[(v1, v2)] = 1 / dv
    return Ew


def randomWeights(G):
    '''
    Every edge has random weight.
    After weights assigned,
    we normalize weights of all incoming edges so that they sum to 1.
    '''
    Ew = dict()
    for u in G:
        in_edges = G.in_edges([u], data=True)
        ew = [random.random() for e in in_edges]  # random edge weights
        total = 0  # total sum of weights of incoming edges (for normalization)
        for num, (v1, v2, edata) in enumerate(in_edges):
            total += edata['weight'] * ew[num]
        for num, (v1, v2, _) in enumerate(in_edges):
            Ew[(v1, v2)] = ew[num] / total
    return Ew


def runLT(G, S, Ew=None):
    """
     NOTE: multiple k edges between nodes (u,v) are considered as one node with weight k. For this reason
     when u is activated the total weight of (u,v) = Ew[(u,v)]*k
    :param G: networkx directed graph
    :param S: Seed set S  type(S) = list
    :param Ew: Infleunce edge weights type(Ew) = dict
    :return:
    """
    if type(G) == nx.Graph:
        G = G.to_directed()
    edges = G.edges
    for edge in edges:
        G.edges[edge[0], edge[1]]['weight'] = 1
    if Ew is not None:
        assert type(Ew) == dict, 'Infleunce edge weights Ew should be an instance of dict'
    else:
        Ew = uniformWeights(G)  # or Ew = uniformWeights(G)

    T = deepcopy(S)  # targeted set
    lv = dict()  # threshold for nodes
    for u in G:
        lv[u] = random.random()

    W = dict(zip(G.nodes(), [0] * len(G)))
    Sj = deepcopy(S)
    while len(Sj):  # while we have newly activated nodes
        Snew = []
        for u in Sj:
            for v in G[u]:
                if v not in T:
                    W[v] += Ew[(u, v)] * G[u][v]['weight']
                    if W[v] >= lv[v]:
                        Snew.append(v)
                        T.append(v)
        Sj = deepcopy(Snew)
    return T
