from IC.IC import unWeightIC, weightIC
from LT.LT import runLT
from OLI import OpinionLeaderInfluence
import utils
from FineGrainedUserRleDivision import FineGrainedUserRleDivision as roleD

if __name__ == "__main__":
    from parameters import parameter_parser

    args = parameter_parser()
    graph = utils.load_graph(args)

    roleDivisionInstance = roleD(graph)
    dict_ = roleDivisionInstance.run()
    print(dict_)
    # oliInstance = OpinionLeaderInfluence(graph)
    # OLI = oliInstance.run()
    # topk = int(len(graph.nodes) * 0.2) + 1
    # # sort by OLI and select topk
    # result = sorted(OLI.items(), key=lambda x: x[1], reverse=True)[:topk]
    # seeds = [key[0] for key in result]
    # msg = "seeds: "
    # for node in seeds:
    #     msg = msg + str(node) + " "
    # print(msg)
    #
    # a = weightIC(graph, seeds)
    # print(a)
    #
    # b = runLT(graph, seeds)
    # print(b)


