
import numpy as np
import random
import networkx as nx


class SIR(object):
    def __init__(self, G, seed_node_list, beta=0.5, gama=0.5):
        """
        初始化模型参数,初始化易感者、感染者和移除者在总体人数N中的比例
        :param beta: 感染系数
        :param gama: 恢复系数
        """
        self.beta = beta
        self.gama = gama
        k = len(seed_node_list)
        self.s0 = 1 - (k / len(G))
        self.i0 = k / len(G)
        self.r0 = 0
        self.seed_node_list = seed_node_list
        self.G = G

    def sir_model(self, y, beta, gama):
        s, i, r = y
        ds_dt = - beta * s * i
        di_dt = beta * s * i - gama * i
        dr_dt = gama * i
        return np.array([ds_dt, di_dt, dr_dt])

    def run(self):
        # 自定义
        influenced_node = []
        G = self.G
        i_state_node = self.seed_node_list[:] # 列表拷贝
        r_state_node = []
        count = 0
        count_ = []
        resu = []
        # 假设每个节点之间的传染行为是相互独立的
        # 对于每个节点来说，传播的初始s/i/r是累计的
        s0, i0, r0 = self.s0, self.i0, self.r0
        while len(i_state_node) > 0:
            # 取出每个节点，进行SIR传播
            node = i_state_node.pop()
            # 找邻居节点
            node_neighbors = G.neighbors(node)
            # 依次计算每个邻居节点的度
            for u in node_neighbors:
                # 受到邻居感染节点个数的影响
                m = 1
                for v in G.neighbors(u):
                    if v in i_state_node:
                        m += 1

                pro = self.beta * m
                if len(i_state_node) / len(G) < i0 and random.random() < pro:
                    i_state_node.append(u)
                    influenced_node.append(u)

            if len(r_state_node) / len(G) < r0: # 恢复
                r_state_node.append(node)

            if random.random() < self.gama: # 保持I状态
                i_state_node.append(node)

            # upate
            if abs(i0) > pow(10, -8) > 0:
                resus = self.sir_model([s0, i0, r0], self.beta, self.gama)
                # 更新s0/i0/r0
                s0 += resus[0]
                i0 += resus[1]
                r0 += resus[2]
                resu.append([s0, i0, r0])
            else:
                break
            print(s0, i0, r0)

        # r_state_node & influenced_node
        print(len(set(influenced_node)))
        return len(set(r_state_node)) + len(set(i_state_node))


def sir_one(graph, seeds, beta=0.6, mu=0.5):
    """单接触SIR模型
    传染率beta；恢复率mu"""
    I_set = set(seeds)
    S_set = set(graph.nodes()).difference(I_set)
    R_set = set()

    t = 0

    while len(I_set) > 0:
        Ibase = I_set.copy()
        for i in Ibase:
            # I --> R
            if random.random() < mu:
                I_set.remove(i)
                R_set.add(i)

            # S --> I
            if random.random() < beta:
                tmp = list(set(graph.neighbors(i)).intersection(S_set).copy())
                if len(tmp) > 0:
                    s = random.choice(tmp)
                    S_set.remove(s)
                    I_set.add(s)

        t += 1

    per_R = len(R_set) / len(graph.nodes())
    # print(t)
    # print("The percentage of R: " + str(per_R))

    return len(R_set), per_R



def sir_more(graph, seeds, beta=0.5, mu=0.5):
    """全接触SIR模型
    传染率beta；恢复率mu"""
    I_set = set(seeds)
    S_set = set(graph.nodes()).difference(I_set)
    R_set = set()

    t = 0

    while len(I_set) > 0:
        Ibase = I_set.copy()
        for i in Ibase:
            # I --> R
            if random.random() < mu:
                I_set.remove(i)
                R_set.add(i)

            # S --> I
            tmp = list(set(graph.neighbors(i)).intersection(S_set).copy())
            for s in tmp:
                if random.random() < beta:
                    S_set.remove(s)
                    I_set.add(s)

        t += 1

    per_R = len(R_set) / len(graph.nodes())
    print("The percentage of R: " + str(per_R))

    return len(R_set)
