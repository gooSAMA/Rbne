from copy import deepcopy
from random import random
import numpy as np
import networkx as nx
import igraph as ig


def convertGraph(networkx_graph):
    edgeList = nx.to_pandas_edgelist(networkx_graph).values
    return ig.Graph(edgeList)

def unWeightIC(g, S, p=0.3):
    """
    无权重的IC模型
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """
    g = convertGraph(g)
    new_active, A = S[:], S[:]
    while new_active:
        # For each newly active node, find its neighbors that become activated
        new_ones = []
        for node in new_active:
            success = np.random.uniform(0, 1, len(g.neighbors(node, mode="out"))) < p
            new_ones += list(np.extract(success, g.neighbors(node, mode="out")))

        new_active = list(set(new_ones) - set(A))
        # Add newly activated nodes to the set of activated nodes
        A += new_active
    return A

def weightIC(G, S, p=.3):
    """
    :param G: networkx graph
    :param S: nodes set
    :param p: propagation probability
    :return: resulted influenced set of vertices (including S)
    """
    edges = G.edges
    for edge in edges:
        G.edges[edge[0], edge[1]]['weight'] = 1

    T = deepcopy(S)
    Acur = deepcopy(S)
    Anext = []
    i = 0
    while Acur:
        for u in Acur:
            for v in G[u]:
                if v not in T:
                    w = G[u][v]['weight']
                    if random() < 1 - (1 - p) ** w:
                        Anext.append((v, u))
        Acur = [edge[0] for edge in Anext]
        i += 1
        T.extend(Acur)
        Anext = []
    return T
