"""
    Role based Node Embedding.
"""
from walkers import FirstOrderRandomWalker, SecondOrderRandomWalker
import networkx as nx
from tqdm import tqdm
import random, os
from gensim.models.word2vec import Word2Vec, LineSentence
import numpy as np


class Rbne:
    def __init__(self, graph, args):
        if os.path.exists(args.walks_filename):
            os.remove(args.walks_filename)
        self.model = None
        assert type(graph) == type(nx.karate_club_graph())
        self.role_type_walks = []
        self.graph = graph
        self.args = args
        self.walks = None
        self.sampler = None
        self.type_map = None
        self.feature_map = None
        self.embeddings = None

    def structure_random_walks(self):
        """
        Doing first/second order random walks.
        """
        if self.args.sampling == "second":
            print(self.args.walk_length)
            self.sampler = SecondOrderRandomWalker(self.graph, self.args.P, self.args.Q, self.args.walk_number,
                                                   self.args.walk_length)
        else:
            self.sampler = FirstOrderRandomWalker(self.graph, self.args.walk_number, self.args.walk_length)
        self.walks = self.sampler.walks
        del self.sampler

    def role_type_random_walks(self, node_role_maps):
        """
        {node: type}
        在对应type集合中的随机游走
        :param node_role_maps: 用户角色字典
        :param needProcess: node_role_maps 是否已经处理过了
        :return:
        """
        isFineGrainedUserRleDivision = False
        if self.args.features == "fineGrainedUserRoleDiversion":
            isFineGrainedUserRleDivision = True
        self.classify_node_by_role(node_role_maps, isFineGrainedUserRleDivision)
        self.feature_map = node_role_maps
        nodes = list(self.graph.nodes())
        for iteration in range(self.args.walk_number):
            print("\nRole type random walk. Random walk round: " + str(iteration + 1) + "/" + str(
                self.args.walk_number) + ".\n")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                node = str(node)
                walk_nodes = self.role_type_walk(start_node=node, isFineGrainedUserRleDivision=isFineGrainedUserRleDivision)
                self.role_type_walks.append(walk_nodes)

    def role_type_walk(self, start_node, isFineGrainedUserRleDivision):
        walk = [start_node]
        while len(walk) < self.args.walk_length:
            if isFineGrainedUserRleDivision:
                for node_type in self.feature_map[int(start_node)]:
                    same_type_nodes = self.type_map[node_type]
                    walk.append(random.choice(same_type_nodes))
            else:
                node_type = self.feature_map[start_node]
                same_type_nodes = self.type_map[node_type[0]]
                walk.append(random.choice(same_type_nodes))
        return walk

    def classify_node_by_role(self, feature_map, isFineGrainedUserRleDivision):
        """
        按照role type 进行归类
        :param feature_map:
        :return:
        """
        type_map = {}
        nodes = feature_map.keys()
        for node in nodes:
            if isFineGrainedUserRleDivision:
                for node_type in feature_map[node]:
                    if node_type in type_map.keys():
                        type_map[node_type].append(node)
                    else:
                        type_map[node_type] = [node]
            else:
                node_type = feature_map[node][0]
                if node_type in type_map.keys():
                    type_map[node_type].append(node)
                else:
                    type_map[node_type] = [node]
        self.type_map = type_map

    def get_node_context_documents(self):
        # return self.walks# + self.role_type_walks
        return self.walks + self.role_type_walks

    def create_embedding(self):
        """
        从结构随机游走和用户角色类型随机游走的两个语料库中训练节点的表示
        :return: node emebdding
        """
        print("\nCreating node embedding... \n")
        corpus = self.get_node_context_documents()
        paths = self.args.walks_filename.split("/")
        if not os.path.exists("./" + paths[1]):
            os.makedirs("./" + paths[1])

        with open(self.args.walks_filename, mode="w+", encoding='utf-8') as file:
            for walk in corpus:
                _line = ""
                for e in walk:
                    if len(_line) != 0:
                        _line += " "
                    _line += str(e)
                file.write(_line + "\n")
        sentences = LineSentence(self.args.walks_filename)
        model = Word2Vec(
            sentences,
            size=self.args.dimensions,
            window=1,
            min_count=self.args.min_count,
            sg=1,
            alpha=self.args.alpha,
            min_alpha=self.args.min_alpha,
            sample=self.args.down_sampling,
            iter=self.args.epochs,
            workers=self.args.workers)

        self.model = model
        self.get_graph_node_embeddings()
        print("Successful created embedding. ")

    def get_graph_node_embeddings(self):
        if self.model is None:
            print("\nmodel not train \n")
            return {}
        nodes = [int(node) for node in self.graph.nodes]
        nodes.sort()
        embeddings_ = []
        for node in nodes:
            embeddings_.append(self.model.wv[str(node)])
        self.embeddings = np.array(embeddings_)


    def train_final_embedding(self):
        """
        将上面create_embedding的结果作为网络属性，输入到GCN中，得到最终的嵌入表示
        即：structure + gcn
        这样的效果会不会更好？
        解释：进一步加强了属性或者结构特性
        :return:
        """
        pass
