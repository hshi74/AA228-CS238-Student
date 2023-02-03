import graphviz
import itertools
import networkx as nx
import numpy as np
import pandas as pd
import pdb
import shutil
import sys
import time

from scipy.special import gammaln


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write(f"{idx2names[edge[0]]}, {idx2names[edge[1]]}\n")


def compute(infile, outfile):
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING
    # pass

    data = pd.read_csv(infile)
    idx2names = {}

    for idx, col in enumerate(data.columns):
        idx2names[idx] = col

    print('idx2names: ', idx2names)
    data.columns = np.arange(len(data.columns))

    G = init_graph(data)

    start_time = time.time()
    while True:
        score = bayesian_score(G, data)
        k2(G, data, 1)
        edge_direction_optimization(G, data)
        if score == bayesian_score(G, data):
            break

    end_time = time.time()

    write_gph(G, idx2names, outfile)
    draw_graph(G, idx2names, infile.replace('csv', 'png'))

    print(f'time spent : {end_time - start_time}, final score: {score}')


def init_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(list(data.columns))
    return G


def bayesian_score_component(G, node, data, parent_func_name=''):
    score = 0
    ri = max(data[node])
    parents = sorted(list(G.pred[node]))

    for inst in itertools.product(*[data[p].unique() for p in parents]):
        temp = data.copy()
        if parent_func_name == 'k2':
            pass
        for l in range(len(parents)):
            temp = temp[temp[parents[l]] == inst[l]]

        score += gammaln(ri) - gammaln(ri + len(temp))

        for k in range(ri):
            m = len(temp[temp[node]] == k + 1)
            score += gammaln(1 + m)

    return score


def bayesian_score(G, data):
    return sum(bayesian_score_component(G, node, data) for node in G)


def draw_graph(G, idx2names, filename):
    G = nx.relabel_nodes(G, idx2names)
    nx.drawing.nx_pydot.write_dot(G, filename)
    img_file = graphviz.render('dot', 'png', filename)
    shutil.move(img_file, filename)


def k2(G, data, parent_count):
    print('Running k2 algorithm')
    scores = [bayesian_score_component(G, node, data) for node in G]
    # print('scores: ', scores)

    for idx, node in enumerate(G):
        parents = []

        while True:
            curr_score = sum(scores)
            curr_best_local_score = scores[idx]
            curr_best_parent = None

            for new_parent in G:
                Gtemp = G.copy()

                if new_parent == node or new_parent in Gtemp.pred[node] or node in Gtemp.pred[new_parent]:
                    continue

                Gtemp.add_edge(new_parent, node)
                new_local_score = bayesian_score_component(Gtemp, node, data, 'k2')
                if nx.is_directed_acyclic_graph(Gtemp) and new_local_score < curr_best_local_score:
                    curr_best_local_score = new_local_score
                    curr_best_parent = new_parent

            if curr_best_parent:
                G.add_edge(curr_best_parent, node)
                parents.append(curr_best_parent)
                scores[idx] = curr_best_local_score

                print(f'{curr_score} -> {sum(scores)}')

                if len(parents) >= parent_count:
                    break

            else:
                break


def edge_direction_optimization(G, data):
    print('Running edge direction optimization')
    scores = {node: bayesian_score_component(G, node, data) for node in G}

    for edge in list(G.edges):
        curr_score = sum(scores.values())
        Gtemp = G.copy()

        parent, child = edge
        parent_score = scores[parent]
        child_score = scores[child]
        Gtemp.remove_edge(*edge)
        Gtemp.add_edge(*edge[::-1])

        if not nx.is_directed_acyclic_graph(Gtemp):
            continue

        new_parent_score = bayesian_score_component(Gtemp, parent, data)
        new_child_score = bayesian_score_component(Gtemp, child, data)

        if new_parent_score + new_child_score < parent_score + child_score:
            print(f'swap {parent} -> {child}')
            G.remove_edge(*edge)
            G.add_edge(*edge[::-1])
            scores[parent] = new_parent_score
            scores[child] = new_child_score
            print(f'{curr_score} -> {sum(scores.values())}')


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
