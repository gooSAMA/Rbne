"""
    使用Rbne得到嵌入后的IM后续操作
    Implementation of heuristic based on Connected Components (CC) for undirected graph.
    We first delete an edge from original graph with probability (1-p)**w.
    Then we calculate SCC for this graph.
    Then for all vertices in a component we add to its score the number of nodes in this component.
    Procedure repeats R times to get some average.
    Then k nodes with top scores are selected.

    References:
    Kempe et al. "Maximizing the spread of influence through a social network" Claim 2.3
"""
import numpy as np
import random
from heapq import nlargest, nsmallest
from copy import deepcopy

class RbneIM:
    def __init__(self, embedding, graph, p=0.5, theta=0.5):
        """
        :param embedding: graph embedding
        :param graph: netwrokx graph
        :param ic_p: propagation probability under Independent Cascade
        :param theta: hyper-parameter of connecting edges
        """
        self.embedding = embedding
        self.graph = graph
        self.p = p
        self.theta = theta
        self.weight = None
        self.calcGraphWeight()

    def get_cos_similar_matrix(self, matrix_1):
        num = np.dot(matrix_1, np.array(matrix_1).T)  # 向量点乘
        denom = np.linalg.norm(matrix_1, axis=1).reshape(-1, 1) * np.linalg.norm(matrix_1, axis=1)  # 求模长的乘积
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def calcGraphWeight(self):
        self.weight = self.get_cos_similar_matrix(self.embedding)
        # 丢弃阈值小于theta的相似度的边
        self.weight[self.weight < self.theta] = 0
        # np.savetxt("Weight2.csv", self.weight, delimiter=',')

    def heuristic(self, topk, R=20):
        """
        :param R: number of iterations to estimate scores of nodes (int)
        :return:
        """
        G = self.graph
        k = topk
        p = self.p
        scores = dict(zip(G.nodes(), [0] * len(G)))  # initialize scores
        for it in range(R):
            # remove blocked edges from graph G
            E = deepcopy(G)
            edge_rem = [e for e in E.edges() if random.random() < (1 - p) ** (self.weight[e[0]][e[1]])]
            E.remove_edges_from(edge_rem)

            # initialize CC
            CC = dict()  # each component is reflection os the number of a component to its members
            explored = dict(zip(E.nodes(), [False] * len(E)))
            c = 0
            # perform BFS to discover CC
            for node in E:
                if not explored[node]:
                    c += 1
                    explored[node] = True
                    CC[c] = [node]
                    component = list(dict(E[node]).keys())
                    for neighbor in component:
                        if not explored[neighbor]:
                            explored[neighbor] = True
                            CC[c].append(neighbor)
                            component += list(dict(E[neighbor]).keys())

            # add score only to top components
            topCC = nlargest(k, CC.values(), key=lambda dv: len(dv))

            for component in topCC:
                weighted_score = 1.0 / len(component) ** (.5)
                # weighted_score = 1
                for node in component:
                    if random.random() < weighted_score:
                        scores[node] += weighted_score

        S = nsmallest(k, scores.keys(), key=lambda key: scores[key])  # select k nodes with top scores
        return S

    def train(self, topk):
        return self.heuristic(topk)
