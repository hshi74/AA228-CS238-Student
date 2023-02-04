import graphviz
import numpy as np
import networkx as nx
import pandas as pd
import pdb
import shutil
import sys
import time

from scipy.special import loggamma
from tqdm import tqdm


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write(f"{idx2names[edge[0]]}, {idx2names[edge[1]]}\n")


def draw_graph(G, idx2names, filename):
    G = nx.relabel_nodes(G, idx2names)
    nx.drawing.nx_pydot.write_dot(G, filename)
    img_file = graphviz.render('dot', 'png', filename)
    shutil.move(img_file, filename)


def init_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(list(data.columns))
    return G


def get_m_alpha(node, parents, data):
    sizes = data.loc[:, parents + [node]].groupby(parents + [node]).size()
    if len(parents) > 0:
        m = sizes.unstack(fill_value=0).values
    else:
        m = sizes.values[None]
    alpha = np.ones(m.shape)
    m_0 = np.sum(m, axis=1)
    alpha_0 = np.sum(alpha, axis=1)
    return m, m_0, alpha, alpha_0


def bayesian_score_component(G, node, data):
    parents = list(G.pred[node])
    m, m_0, alpha, alpha_0 = get_m_alpha(node, parents, data)
    score = np.sum(loggamma(alpha_0) - loggamma(alpha_0 + m_0)) + np.sum(loggamma(alpha + m) - loggamma(alpha))
    return score


def bayesian_score(G, data):
    return [bayesian_score_component(G, node, data) for node in G]


def k2(G, data):
    for node in G:
        score_cur = bayesian_score_component(G, node, data)
        while True:
            score_best = float('-inf')
            parent_best = None

            for parent in G:
                if parent == node or parent in G.pred[node] or node in G.pred[parent]:
                    continue

                G.add_edge(parent, node)
                score_new = bayesian_score_component(G, node, data)
                if nx.is_directed_acyclic_graph(G) and score_new > score_best:
                    score_best = score_new
                    parent_best = parent

                G.remove_edge(parent, node)

            if parent_best and (score_cur == 0 or score_best > score_cur):
                score_cur = score_best
                G.add_edge(parent_best, node)
            else:
                break


def rand_graph_neighbor(G):
    n = nx.number_of_nodes(G)
    i = np.random.randint(0, n)
    j = (i + np.random.randint(1, n)) % n
    G_ = G.copy()
    if G.has_edge(i, j):
        G_.remove_edge(i, j)
        if np.random.rand() > 0.5:
            G_.add_edge(j, i)
    else:
        G_.add_edge(i, j)
    return G_


def local_search(G, D, k_max=1000):
    y = sum(bayesian_score(G, D))
    for k in tqdm(range(k_max)):
        G_ = rand_graph_neighbor(G)
        if nx.is_directed_acyclic_graph(G_):
            y_ = sum(bayesian_score(G_, D))
        else:
            y_ = float('-inf')

        if y_ > y:
            y, G = y_, G_

    return G


def compute(dataset_size):
    k_max_dict = {'small': 1000, 'medium': 1000, 'large': 10000}
    
    data = pd.read_csv(f'data/{dataset_size}.csv')

    idx2names = {}

    for idx, col in enumerate(data.columns):
        idx2names[idx] = col

    print('idx2names: ', idx2names)
    data.columns = np.arange(len(data.columns))

    G = init_graph(data)

    start = time.time()
    k2(G, data)
    G = local_search(G, data, k_max=k_max_dict[dataset_size])
    score = sum(bayesian_score(G, data))
    end = time.time()
    
    write_gph(G, idx2names, f'data/{dataset_size}.gph')
    draw_graph(G, idx2names, f'data/{dataset_size}.png')

    print(f'time spent: {end - start} | score: {score}')


def main():
    dataset_size = sys.argv[1]
    compute(dataset_size)


if __name__ == '__main__':
    main()