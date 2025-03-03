import torch
import numpy as np
from nltk.tree import *
from hierarchies.tree import Tree as T
import pandas as pd
from _collections import OrderedDict

def get_label(node):
    if isinstance(node, Tree):
        return node.label()
    else:
        return node


def get_classes(hierarchy: Tree, output_all_nodes=False):
    """
    Return all classes associated with a hierarchy. The classes are sorted in
    alphabetical order using their label, putting all leaf nodes first and the
    non-leaf nodes afterwards.
    Args:
        hierarhcy: The hierarchy to use.
        all_nodes: Set to true if the non-leaf nodes (excepted the origin) must
            also be included.
    Return:
        A pair (classes, positions) of the array of all classes (sorted) and the
        associated tree positions.
    """

    def get_classes_from_positions(positions):
        classes = [get_label(hierarchy[p]) for p in positions]
        class_order = np.argsort(classes)  # we output classes in alphabetical order
        positions = [positions[i] for i in class_order]
        classes = [classes[i] for i in class_order]
        return classes, positions

    positions = hierarchy.treepositions("leaves")
    classes, positions = get_classes_from_positions(positions)

    if output_all_nodes:
        positions_nl = [p for p in hierarchy.treepositions() if p not in positions]
        classes_nl, positions_nl = get_classes_from_positions(positions_nl)
        classes += classes_nl
        positions += positions_nl

    return classes, positions

def convert_nltk_tree(molemap_tree):
    d1 = []
    d2 = []
    for node in molemap_tree.inodes:
        if node.depth == 2:
            d2.append(Tree(node.name, [molemap_tree.G.nodes[c]['label'] for c in node.succ]))

    for node in molemap_tree.inodes:
        if node.depth == 1:
            d1.append(Tree(node.name, [n for n in d2 if node.name in n.label()]))

    tree = Tree('skin_leison', d1)
    torch.save(tree, 'hierarchies/molemap_nltk_tree.pt')


if __name__ == '__main__':
    # dog = Tree('dog', [])
    dp1 = Tree('dp', [Tree('d', ['are']), Tree('np', ['dog'])])
    dp2 = Tree('dp', [Tree('d', ['the']), Tree('np', ['cat'])])
    vp = Tree('vp', [Tree('v', ['chased']), dp2])
    tree = Tree('s', [dp1, vp])
    tree.pretty_print()

    get_classes(tree)

    # build molemap nltk tree
    csv = pd.read_csv('molemap/img2targets_d4_MIC_PNG.POL.csv')
    graph = 'hierarchies/graphs/molemap/graph-preset-noprune0.json'
    wnid = 'hierarchies/wnids/molemap.txt'
    classes = sorted(set(csv.label))
    molemap_tree = T('molemap', path_graph=graph, path_wnids=wnid, classes=classes)

    convert_nltk_tree(molemap_tree)





