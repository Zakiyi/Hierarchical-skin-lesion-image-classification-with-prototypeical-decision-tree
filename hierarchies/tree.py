"""Tree and node utilities for navigating the NBDT hierarchy"""
# source code from https://github.com/alvinwan/neural-backed-decision-trees

from collections import defaultdict
from hierarchies.thirdparty.wn import get_wnids, FakeSynset, wnid_to_synset, wnid_to_name
from hierarchies.thirdparty.nx import read_graph, get_leaves, get_leaf_to_path
from hierarchies.utils import (
    dataset_to_default_path_graph,
    dataset_to_default_path_wnids,
    hierarchy_to_path_graph,
)
from hierarchies.utils import DATASETS, DATASET_TO_NUM_CLASSES


def dataset_to_dummy_classes(dataset):
    assert dataset in DATASETS
    num_classes = DATASET_TO_NUM_CLASSES[dataset]
    return [FakeSynset.create_from_offset(i).wnid for i in range(num_classes)]


def add_arguments(parser):
    parser.add_argument(
        "--hierarchy",
        help="Hierarchy to use. If supplied, will be used to "
        "generate --path-graph. --path-graph takes precedence.",
    )
    parser.add_argument(
        "--path-graph", help="Path to graph-*.json file."
    )  # WARNING: hard-coded suffix -build in generate_checkpoint_fname
    parser.add_argument("--path-wnids", help="Path to wnids.txt file.")


class Node:
    def __init__(self, tree, wnid, other_class=False):
        self.tree = tree

        self.wnid = wnid
        self.name = self.wnid_to_name(wnid)     # wnid_to_name(wnid)     # wnid --> synet --> class name
        self.synset = wnid_to_synset(wnid)

        self.original_classes = tree.classes
        self.num_original_classes = len(self.tree.wnids_leaves)

        self.has_other = other_class and not (self.is_root() or self.is_leaf())
        self.num_children = len(self.succ)

        self.num_classes = self.num_children + int(self.has_other)

        (
            self.class_index_to_child_index,   # old_to_new
            self.child_index_to_class_index,   # new_to_old
        ) = self.build_class_mappings()
        self.classes = self.build_classes()

        assert len(self.classes) == self.num_classes, (
            f"Number of classes {self.num_classes} does not equal number of "
            f"class names found ({len(self.classes)}): {self.classes}"
        )

        self.leaves = list(self.get_leaves())
        self.num_leaves = len(self.leaves)
        self.preds = None

    def wnid_to_class_index(self, wnid):
        return self.tree.wnids_leaves.index(wnid)

    def wnid_to_child_index(self, wnid):
        return [child.wnid for child in self.children].index(wnid)

    def wnid_to_name(self, wnid):
        return self.tree.G.nodes[wnid]['label']

    @property
    def parent(self):
        if not self.parents:
            return None
        return self.parents[0]

    def predecessors_nodes(self, wnid):
        p = []
        while list(self.tree.G.predecessors(wnid)):
            p.append(list(self.tree.G.predecessors(wnid))[0])
            wnid = list(self.tree.G.predecessors(wnid))[0]
        return p

    @property
    def depth(self):
        return len(self.predecessors_nodes(self.wnid))
    @property
    def pred(self):
        return self.tree.G.pred[self.wnid]

    @property
    def parents(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.pred]

    @property
    def succ(self):
        return self.tree.G.succ[self.wnid]

    @property
    def children(self):
        return [self.tree.wnid_to_node[wnid] for wnid in self.succ]

    def get_leaves(self):
        return get_leaves(self.tree.G, self.wnid)

    def is_leaf(self):
        return len(self.succ) == 0

    def is_root(self):
        return len(self.pred) == 0

    def build_class_mappings(self):
        if self.is_leaf():
            return {}, {}

        old_to_new = defaultdict(lambda: [])
        new_to_old = defaultdict(lambda: [])
        for new_index, child in enumerate(self.succ):   # succ contains child ids
            for leaf in get_leaves(self.tree.G, child):
                old_index = self.wnid_to_class_index(leaf)
                old_to_new[old_index].append(new_index)
                new_to_old[new_index].append(old_index)   # parent ---> child

        if not self.has_other:
            return old_to_new, new_to_old

        new_index = self.num_children
        for old in range(self.num_original_classes):
            if old not in old_to_new:
                old_to_new[old].append(new_index)
                new_to_old[new_index].append(old)
        return old_to_new, new_to_old

    def build_classes(self):
        return [
            ",".join([self.original_classes[old] for old in old_indices])
            for new_index, old_indices in sorted(
                self.child_index_to_class_index.items(), key=lambda t: t[0]
            )
        ]

    @property
    def class_counts(self):
        """Number of old classes in each new class"""
        return [len(old_indices) for old_indices in self.child_index_to_class_index]

    @staticmethod
    def dim(nodes):
        return sum([node.num_classes for node in nodes])


class Tree:
    def __init__(self, dataset, path_graph=None, path_wnids=None, classes=None, hierarchy=None):
        if dataset and hierarchy and not path_graph:
            path_graph = hierarchy_to_path_graph(dataset, hierarchy)
        if dataset and not path_graph:
            path_graph = dataset_to_default_path_graph(dataset)   # hierarchy_to_path_graph(dataset, "induced")
        if dataset and not path_wnids:
            path_wnids = dataset_to_default_path_wnids(dataset)
        if dataset and not classes:
            classes = dataset_to_dummy_classes(dataset)

        self.load_hierarchy(dataset, path_graph, path_wnids, classes)

    def load_hierarchy(self, dataset, path_graph, path_wnids, classes):
        self.dataset = dataset
        self.path_graph = path_graph
        self.path_wnids = path_wnids     # wordnet id
        self.classes = classes
        self.G = read_graph(path_graph)  # graph
        self.wnids_leaves = get_wnids(path_wnids)   # wordnet id of leaf classes
        self.wnid_to_class = {wnid: cls for wnid, cls in zip(self.wnids_leaves, self.classes)}
        self.wnid_to_class_index = {wnid: i for i, wnid in enumerate(self.wnids_leaves)}
        self.wnid_to_node = self.get_wnid_to_node()
        self.nodes = [self.wnid_to_node[wnid] for wnid in sorted(self.wnid_to_node)]
        self.inodes = [node for node in self.nodes if not node.is_leaf()]
        self.leaves = [self.wnid_to_node[wnid] for wnid in self.wnids_leaves]

    def update_from_model(self, model, arch, dataset, classes=None, path_wnids=None, path_graph=None):
        from nbdt.nbdt.hierarchy import generate_hierarchy  # avoid circular import hah
        assert model is not None, "`model` cannot be NoneType"
        path_graph = generate_hierarchy(dataset=dataset, method="induced", arch=arch, model=model, path=path_graph)
        tree = Tree(dataset, path_graph=path_graph, path_wnids=path_wnids, classes=classes, hierarchy="induced")
        self.load_hierarchy(
            dataset=tree.dataset,
            path_graph=tree.path_graph,
            path_wnids=tree.path_wnids,
            classes=tree.classes
        )

    @classmethod
    def create_from_args(cls, args, classes=None):
        return cls(
            args.dataset,
            args.path_graph,
            args.path_wnids,
            classes=classes,
            hierarchy=args.hierarchy,
        )

    @property
    def root(self):
        for node in self.inodes:
            if node.is_root():
                return node
        raise UserWarning("Should not be reachable. Tree should always have root")

    def get_wnid_to_node(self):
        wnid_to_node = {}
        for wnid in self.G:
            wnid_to_node[wnid] = Node(self, wnid)
        return wnid_to_node

    def get_leaf_to_steps(self):
        node = self.inodes[0]
        leaf_to_path = get_leaf_to_path(self.G)
        leaf_to_steps = {}
        for leaf in self.wnids_leaves:
            next_indices = [index for index, _ in leaf_to_path[leaf][1:]] + [-1]
            leaf_to_steps[leaf] = [
                {
                    "node": self.wnid_to_node[wnid],
                    "name": self.wnid_to_node[wnid].name,
                    "next_index": next_index,  # curr node's next child index to traverse
                }
                for next_index, (_, wnid) in zip(next_indices, leaf_to_path[leaf])
            ]
        return leaf_to_steps

    def visualize(self, path_html, G=None, dataset=None, **kwargs):
        """
        :param path_html: Where to write the final generated visualization
        """
        if G is None:
            G = self.G
        from hierarchies.visualization import generate_hierarchy_vis_from  # avoid circular import hah
        generate_hierarchy_vis_from(
            G=G,
            dataset=dataset,
            path_html=path_html,
            **kwargs
        )


if __name__ == '__main__':
    import argparse
    import pandas as pd
    from datasets.molemap import MolemapDataset
    df = pd.read_csv('datasets/molemap_dataset.csv')
    df_train = df[df.is_train == 1].copy()
    trainset = MolemapDataset('datasets/resize512_MIC_POL', df_train, size=320, is_train=True, test_mode=False, debug=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="molemap")
    args = parser.parse_args()
    args.path_graph = 'hierarchies/graphs/molemap/graph-preset-noprune.json'
    args.path_wnids = 'hierarchies/wnids/molemap.txt'
    args.hierarchy = 'preset'
    tree = Tree.create_from_args(args, classes=trainset.classes)
    # print([(nodes.wnid, nodes.name, nodes.depth) for nodes in tree.inodes])
    # print([(node.wnid, node.name, node.depth) for node in tree.leaves])
    print(tree.inodes[2].wnid, list(tree.inodes[2].succ))
    n = []
    for nodes in tree.inodes:
        if nodes.depth == 1:
            n.append(nodes.name)
    print(n)
