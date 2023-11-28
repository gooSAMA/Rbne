
# Set up
import powerlaw
import pylab
from matplotlib.pyplot import savefig, fill_between

pylab.rcParams['xtick.major.pad']='8'
pylab.rcParams['ytick.major.pad']='8'

from matplotlib import rc
import pandas as pd
import networkx as nx
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

def drawPic(words, name = "cora"):
    rc('font', family='Times New Roman')
    rc('font', size=18.0)
    rc('text', usetex=False)

    panel_label_font = FontProperties().copy()
    panel_label_font.set_weight("bold")
    panel_label_font.set_size(24.0)
    panel_label_font.set_family("Times New Roman")

    figCCDF = powerlaw.plot_pdf(words, linear_bins=True, color='purple', linewidth=4, linestyle="", marker="+", markersize=16)
    ####
    figCCDF.set_ylabel(r'Count of users')
    figCCDF.set_xlabel(r"Frequency of being source user")

    figname = name+"_powerlaw"
    savefig('./output/'+figname+'.png', bbox_inches='tight', dpi=300)

def load_graph(name):
    graph = nx.from_edgelist(pd.read_csv(name).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def generateDatasetDegreeValue(name = "cora"):
    filepath = 'input/{0}_edges.csv'.format(name)
    graph = load_graph(filepath)
    nodes = list(graph.nodes)
    nodes.sort()
    results = []
    for node in nodes:
        results.append(len(list(nx.neighbors(graph, node))))
    return np.array(results)



if __name__ == '__main__':
    name = "CA-AstroPh" # acm | cora | netscience | CA-GrQc | CA-AstroPh
    words = generateDatasetDegreeValue(name)
    drawPic(words, name)