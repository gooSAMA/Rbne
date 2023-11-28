"""Dataset utilities and printing."""

import pandas as pd
import networkx as nx
from texttable import Texttable
from motif_count import MotifCounterMachine
from community_detection import communityPartition
from FineGrainedUserRleDivision import FineGrainedUserRleDivision

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def load_graph(agrs):
    """
    Reading an edge list csv to an NX graph object.
    :param graph_path: Path to the edhe list csv.
    :return graph: NetworkX object.
    """
    # 这里需要实际去测试
    # if 'cora' in agrs.graph_input:
    #     agrs.dimensions = 512
    graph = nx.from_edgelist(pd.read_csv(agrs.graph_input).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def generate_structural_features(args, databaseName, graph):
    if args.features == "degree": # 度中心性
        degrees = {str(node): [str(graph.degree(node))] for node in graph.nodes()}
        save_dict_map_to_csv(degrees, "degree_values.csv")
        return degrees
    elif args.features == "community": # 按照社区结构划分
        partition = communityPartition(graph, resolution=2)
        communities = {str(node): [str(partition[node])] for node in graph.nodes()}
        save_dict_map_to_csv(communities, "community_values.csv")
        return communities
    elif args.features == "fineGrainedUserRoleDiversion": # FineGrainedUserRleDivision
        roleDics = FineGrainedUserRleDivision(graph, databaseName).run()
        return roleDics
    else:  # NMF and motif ==> CoarseGrainedUserRoleDivision
        machine = MotifCounterMachine(graph, args)
        return machine.create_string_labels()


def save_dict_map_to_csv(my_dict, filename = 'my_file.csv'):
    with open('./output/' + filename, 'w') as f:
        [f.write('{0},{1}\n'.format(key, value[0])) for key, value in my_dict.items()]


def read_node_label(filename, skip_head=False, sep=','):
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        if skip_head:
            fin.readline()
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(sep)
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y