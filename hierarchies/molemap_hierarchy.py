import os
import pandas as pd
import networkx as nx
from _collections import OrderedDict
import matplotlib.pyplot as plt
from hierarchies.thirdparty.nx import write_graph

from hierarchies.graph import (
    build_random_graph,
    prune_single_successor_nodes,
    generate_graph_fname,
    print_graph_stats,
    assert_all_wnids_in_graph,
    augment_graph,
    build_induced_graph)

from hierarchies.utils import Colors


DATASET_CSV_FILES = {
    'molemap': 'datasets/molemap_dataset.csv'
}


def generate_wnids(data_csv, only_leaf=False):
    df = pd.read_csv(data_csv)
    l0, l1, l2 = [], [], []
    for l in df.label:
        parts = l.split(':')
        l0.append(parts[0])
        l1.append(':'.join(parts[0:2]))
        l2.append(l)

    i2l0, i2l1, i2l2 = sorted(set(l0)), sorted(set(l1)), sorted(set(l2))

    all_ids = ["n{:02}".format(i) for i in range(1, len(i2l0) + len(i2l1) + len(i2l2) + 1)]
    leaf_ids = all_ids[len(i2l0) + len(i2l1):]
    assert len(leaf_ids) == len(i2l2)

    if only_leaf:
        return leaf_ids
    else:
        return all_ids, (i2l0, i2l1, i2l2)


def get_graph_path_from_args(
    dataset,
    method,
    seed=0,
    branching_factor=2,
    extra=0,
    no_prune=False,
    fname="",
    path="",
    multi_path=False,
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    **kwargs,
):
    if path:
        return path
    fname = generate_graph_fname(
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        extra=extra,
        no_prune=no_prune,
        fname=fname,
        multi_path=multi_path,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        checkpoint=checkpoint,
        arch=arch,
    )
    directory = f'hierarchies/graphs/{dataset}'

    if os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, f"{fname}.json")
    return path


# step 1: build id for leaf classes
# construct edge between nodes
def build_molemap_graph(data_csv):
    ids, (i2l0, i2l1, i2l2) = generate_wnids(data_csv, only_leaf=False)
    id_labels = OrderedDict()
    id_hypernyms = OrderedDict()

    for id, label in zip(ids, i2l0 + i2l1 + i2l2):
        # print(id, label)
        id_labels[id] = label

    for id in id_labels.keys():
        if len(id_labels[id].split(':')) == 1:
            id_hypernyms[id] = 'n00'
        else:
            hypernym = ':'.join(id_labels[id].split(':')[:-1])
            id_hypernyms[id] = list(id_labels.keys())[list(id_labels.values()).index(hypernym)]
            # print(id, id_hypernyms[id])

    G = nx.DiGraph()
    for id in id_labels.keys():
        G.add_node(id)
        G.add_edge(id_hypernyms[id], id)
        nx.set_node_attributes(G, {id: id_labels[id]}, "label")
        if id_hypernyms[id] == 'n00':
            nx.set_node_attributes(G, {id_hypernyms[id]: 'skin_lesion'}, "label")
        else:
            nx.set_node_attributes(G, {id_hypernyms[id]: id_labels[id_hypernyms[id]]}, 'label')

    return G


def generate_hierarchy(
    dataset,
    method="preset",
    seed=0,
    branching_factor=2,
    extra=0,
    no_prune=False,
    fname="",
    path="",
    single_path=False,
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    model=None,
    **kwargs,
):

    path_wnids = f'hierarchies/wnids/{dataset}.txt'
    if not os.path.exists(path_wnids):
        os.makedirs(os.path.dirname(path_wnids), exist_ok=True)
        print(f'==>generate graph ids for {dataset}...and save it to {path_wnids}')
        wnids = generate_wnids(DATASET_CSV_FILES[dataset], only_leaf=True)
        with open(path_wnids, 'w') as f:
            for line in wnids:
                f.write(line)
                f.write('\n')
    else:
        with open(path_wnids) as f:
            wnids = [wnid.strip() for wnid in f.readlines()]

    if method == "preset":
        G = build_molemap_graph(DATASET_CSV_FILES[dataset])
    elif method == "random":
        G = build_random_graph(wnids, seed=seed, branching_factor=branching_factor)
    elif method == "induced":
        G = build_induced_graph(
            wnids,
            dataset=dataset,
            checkpoint=checkpoint,
            model=arch,
            linkage=induced_linkage,
            affinity=induced_affinity,
            branching_factor=branching_factor,
            state_dict=model.state_dict() if model is not None else None,
        )
    else:
        raise NotImplementedError(f'Method "{method}" not yet handled.')

    print_graph_stats(G, "matched")
    assert_all_wnids_in_graph(G, wnids)

    if not no_prune:
        G = prune_single_successor_nodes(G)
        print_graph_stats(G, "pruned")
        assert_all_wnids_in_graph(G, wnids)

    if extra > 0:
        G, n_extra, n_imaginary = augment_graph(G, extra, True)
        print(f"[extra] \t Extras: {n_extra} \t Imaginary: {n_imaginary}")
        print_graph_stats(G, "extra")
        assert_all_wnids_in_graph(G, wnids)

    path = get_graph_path_from_args(
        dataset=dataset,
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        extra=extra,
        no_prune=no_prune,
        fname=fname,
        path=path,
        single_path=single_path,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        checkpoint=checkpoint,
        arch=arch,
    )

    print(path)
    if not os.path.exists(path):
        write_graph(G, path)
        Colors.green("==> Wrote tree to {}".format(path))
    else:
        print(f'graph already exist in {path}')
    return G


if __name__ == '__main__':
    os.chdir('Hierarchical_skin/') # current project dir
    G = generate_hierarchy(dataset='molemap', method='preset', no_prune=True)
    from hierarchies.thirdparty.nx import get_leaves
    print(G.nodes['n01']['label'])
    # print(list(G.nodes))
    # G.remove_nodes_from(['n01'])
    # print(list(get_leaves(G, 'n69')))
    print(get_leaves(G, root=None))
    # A = nx.nx_agraph.to_agraph(G)
    # A.layout(prog="fdp")    # ["neato", "dot", "twopi", "circo", "fdp", "nop"]
    # A.draw("file.png")
    print(list(G.succ['n00']))
    # print(G.pred['n00'])
