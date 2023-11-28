"""
 SHEvaluation.py 的测试案例代码
"""


import SHEvaluation
import utils
from parameters import parameter_parser
import numpy as np

args = parameter_parser()
graph = utils.load_graph(args)

result1 = SHEvaluation.constraint(graph)

print(result1)

results = []
nodes = list(graph.nodes)
nodes.sort()

for node in nodes:
    results.append([node, result1[node] ])

np.savetxt("Sc.csv", np.array(results), delimiter=',')

