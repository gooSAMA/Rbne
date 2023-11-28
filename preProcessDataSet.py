import pandas as pd
import networkx as nx
import numpy as np
import random
import tqdm

def load_graph(dataset):
    graph = nx.from_edgelist(pd.read_csv("./input/{0}_edges.csv".format(dataset)).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


if __name__ == '__main__':
    dataset="Facebook_Government" # Dolphins_edges.csv | Facebook_Caltech36_edges.csv | Facebook_Government_edges.csv
    g = load_graph(dataset)
    nodes = list(g.nodes)
    maxNodeIndex = np.max(nodes)
    for logic_node_index in tqdm.tqdm(range(maxNodeIndex + 1)):
        if not logic_node_index in nodes:
            print(logic_node_index)
            # 随机加边
            numberEdges = random.randint(1, 6)
            g.add_node(logic_node_index)
            for randomEdge in range(numberEdges):
                g.add_edge(logic_node_index, random.choice(nodes))
    # 重新存储
    edges = [[edge[0], edge[1]] for edge in g.edges]
    edges = np.array(edges)
    df = pd.DataFrame({'id1': edges[:, 0], 'id2': edges[:, 1]})
    df.to_csv("./input/{0}_edges.csv".format(dataset), sep=',', index=False)
