import os
import json
from pathlib import Path
from hierarchies.utils import Colors
import base64
from io import BytesIO
from collections import defaultdict
from hierarchies.thirdparty.nx import (
    write_graph,
    get_roots,
    get_root,
    read_graph,
    get_leaves,
    get_depth)
from hierarchies.thirdparty.wn import (
    get_wnids,
    synset_to_wnid,
    wnid_to_name,
    get_wnids_from_dataset,
)
from hierarchies.utils import fwd
from hierarchies.graph import get_graph_path_from_args, generate_graph_fname


def set_dot_notation(node, key, value):
    """
    >>> a = {}
    >>> set_dot_notation(a, 'above.href', 'hello')
    >>> a['above']['href']
    'hello'
    """
    curr = last = node
    key_part = key
    if "." in key:
        for key_part in key.split("."):
            last = curr
            curr[key_part] = node.get(key_part, {})
            curr = curr[key_part]
    last[key_part] = value


def get_class_image_from_dataset(dataset, candidate):
    """Returns image for given class `candidate`. Image is PIL."""
    if isinstance(candidate, int):
        candidate = dataset.classes[candidate]
    for sample, label in dataset:
        intersection = compare_wnids(dataset.classes[label], candidate)
        if label == candidate or intersection:
            return sample
    raise UserWarning(f"No samples with label {candidate} found.")


def compare_wnids(label1, label2):
    from nltk.corpus import wordnet as wn  # entire script should not depend on wordnet

    synsets1 = wn.synsets(label1, pos=wn.NOUN)
    synsets2 = wn.synsets(label2, pos=wn.NOUN)
    wnids1 = set(map(synset_to_wnid, synsets1))
    wnids2 = set(map(synset_to_wnid, synsets2))
    return wnids1.intersection(wnids2)


def image_to_base64_encode(image, format="jpeg"):
    """Converts PIL image to base64 encoding, ready for use as data uri."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue())


def skid_to_name(id, G):
    name = G.nodes[id]['label']
    return name.split(':')[-1]


def build_tree(
    G,
    root,
    parent="null",
    color_info=(),
    force_labels_left=(),
    include_leaf_images=False,
    dataset=None,
    image_resize_factor=1,
    include_fake_sublabels=False,
    include_fake_labels=False,
    node_to_conf={},
):
    """
    :param color_info dict[str, dict]: mapping from node labels or IDs to color
                                       information. This is by default just a
                                       key called 'color'
    """
    children = [
        build_tree(
            G,
            child,
            root,
            color_info=color_info,
            force_labels_left=force_labels_left,
            include_leaf_images=include_leaf_images,
            dataset=dataset,
            image_resize_factor=image_resize_factor,
            include_fake_sublabels=include_fake_sublabels,
            include_fake_labels=include_fake_labels,
            node_to_conf=node_to_conf,
        )
        for child in G.succ[root]
    ]
    _node = G.nodes[root]
    label = _node.get("label", "")
    sublabel = root

    if root.startswith("f") and label.startswith("(") and not include_fake_labels:
        label = ""

    if (
        root.startswith("f") and not include_fake_sublabels
    ):  # WARNING: hacky, ignores fake wnids -- this will have to be changed lol
        sublabel = ""

    node = {
        "sublabel": sublabel,
        "label": label,
        "parent": parent,
        "children": children,
        "alt": _node.get("alt", ", ".join(map(lambda wnid: skid_to_name(wnid, G), get_leaves(G, root=root)))),
        "id": root,
    }
    # print('ssdf ', _node.get("alt", ", ".join(map(lambda wnid: skid_to_name(wnid, G), get_leaves(G, root=root)))))
    if label in color_info:
        node.update(color_info[label])

    if root in color_info:
        node.update(color_info[root])

    if label in force_labels_left:
        node["force_text_on_left"] = True

    is_leaf = len(children) == 0
    if include_leaf_images and is_leaf:
        try:
            image = get_class_image_from_dataset(dataset, label)
        except UserWarning as e:
            print(e)
            return node
        base64_encode = image_to_base64_encode(image, format="jpeg")
        image_href = f"data:image/jpeg;base64,{base64_encode.decode('utf-8')}"
        image_height, image_width = image.size
        node["image"] = {
            "href": image_href,
            "width": image_width * image_resize_factor,
            "height": image_height * image_resize_factor,
        }

    for key, value in node_to_conf[root].items():
        set_dot_notation(node, key, value)
    return node


def build_graph(G):
    return {
        "nodes": [
            {"name": wnid, "label": G.nodes[wnid].get("label", ""), "id": wnid}
            for wnid in G.nodes
        ],
        "links": [{"source": u, "target": v} for u, v in G.edges],
    }


def get_color_info(
    G, color, color_leaves, color_path_to=None, color_nodes=(), theme="regular"
):
    """Mapping from node to color information."""
    nodes = {}

    theme_to_bg = {"minimal": "#EEEEEE", "dark": "#111111"}
    nodes["bg"] = theme_to_bg.get(theme, "#FFFFFF")

    theme_to_text_rect = {
        "minimal": "rgba(0,0,0,0)",
        "dark": "rgba(17,17,17,0.8)",
    }
    nodes["text_rect"] = theme_to_text_rect.get(theme, "rgba(255,255,255,0.8)")

    leaves = list(get_leaves(G))
    if color_leaves:
        for leaf in leaves:
            nodes[leaf] = {"color": color, "highlighted": True, "theme": theme}

    for (id, node) in G.nodes.items():
        if node.get("label", "") in color_nodes or id in color_nodes:
            nodes[id] = {"color": color, "highlighted": True, "theme": theme}
        else:
            nodes[id] = {"color": "black", "theme": theme}

    root = get_root(G)
    target = None
    for leaf in leaves:
        node = G.nodes[leaf]
        if node.get("label", "") == color_path_to or leaf == color_path_to:
            target = leaf
            break

    if target is not None:
        for node in G.nodes:
            nodes[node] = {
                "color": "#5a5c5a",
                "color_incident_edge": True,
                "highlighted": False,
                "theme": theme,
            }

        while target != root:
            nodes[target] = {
                "color": color,
                "color_incident_edge": True,
                "highlighted": True,
                "theme": theme,
            }
            view = G.pred[target]
            target = list(view.keys())[0]
        nodes[root] = {"color": color, "highlighted": False, "theme": theme}
    return nodes


def generate_vis_fname(vis_color_path_to=None, vis_out_fname=None, **kwargs):
    fname = vis_out_fname
    if fname is None:
        fname = generate_graph_fname(**kwargs).replace(
            "graph-", f'{kwargs["dataset"]}-', 1
        )
    if vis_color_path_to is not None:
        fname += "-" + vis_color_path_to
    return fname


def generate_node_conf(node_conf):
    node_to_conf = defaultdict(lambda: {})
    if not node_conf:
        return node_to_conf

    for node, key, value in node_conf:
        if value.isdigit():
            value = int(value)
        node_to_conf[node][key] = value
    return node_to_conf


def generate_vis(
    path_template,
    data,
    path_html,
    zoom=0.6,
    straight_lines=True,
    show_sublabels=False,
    height=750,
    margin_top=20,
    above_dy=325,
    y_node_sep=180,
    hide=[],
    _print=False,
    out_dir=".",
    scale=1,
    colormap="colormap_annotated.png",
    below_dy=475,
    root_y="null",
    width=1000,
    margin_left=250,
    bg="#FFFFFF",
    text_rect="rgba(255,255,255,0.8)",
    stroke_width=0.3,
    verbose=False,
):
    fname = Path(path_html).stem
    out_dir = Path(path_html).parent
    with open(path_template) as f:
        html = (
            f.read()
            .replace("CONFIG_MARGIN_LEFT", str(margin_left))
            .replace("CONFIG_VIS_WIDTH", str(width))
            .replace("CONFIG_SCALE", str(scale))
            .replace("CONFIG_PRINT", str(_print).lower())
            .replace("CONFIG_HIDE", str(hide))
            .replace("CONFIG_Y_NODE_SEP", str(y_node_sep))
            .replace("CONFIG_ABOVE_DY", str(above_dy))
            .replace("CONFIG_BELOW_DY", str(below_dy))
            .replace("CONFIG_TREE_DATA", json.dumps([data]))
            .replace("CONFIG_ZOOM", str(zoom))
            .replace("CONFIG_STRAIGHT_LINES", str(straight_lines).lower())
            .replace("CONFIG_SHOW_SUBLABELS", str(show_sublabels).lower())
            .replace("CONFIG_TITLE", fname)
            .replace("CONFIG_VIS_HEIGHT", str(height))
            .replace("CONFIG_BG_COLOR", bg)
            .replace("CONFIG_TEXT_RECT_COLOR", text_rect)
            .replace("CONFIG_STROKE_WIDTH", str(stroke_width))
            .replace("CONFIG_MARGIN_TOP", str(margin_top))
            .replace("CONFIG_ROOT_Y", str(root_y))
            .replace(
                "CONFIG_COLORMAP",
                f"""<img src="{colormap}" style="
        position: absolute;
        top: 40px;
        left: 80px;
        height: 250px;
        border: 4px solid #ccc;">"""
                if isinstance(colormap, str) and os.path.exists(colormap)
                else "",
            )
        )

    os.makedirs(str(out_dir), exist_ok=True)
    with open(path_html, "w") as f:
        f.write(html)

    if verbose:
        Colors.green("==> Wrote HTML to {}".format(path_html))


def generate_hierarchy_vis(args):
    path_hie = get_graph_path_from_args(**vars(args))
    print("==> Reading from {}".format(path_hie))
    G = read_graph(path_hie)

    path_html = f"./{generate_vis_fname(**vars(args))}.html"
    kwargs = vars(args)

    dataset = None
    if args.dataset and args.vis_leaf_images:
        cls = getattr(data, kwargs.pop('dataset'))
        dataset = cls(root="./data", train=False, download=True)

    kwargs.pop('dataset', '')
    kwargs.pop('fname', '')
    return generate_hierarchy_vis_from(
        G, dataset, path_html, verbose=True, **kwargs
    )


def generate_hierarchy_vis_from(
    G,
    dataset,
    path_html,
    color="blue",
    vis_root=None,
    vis_no_color_leaves=False,
    vis_color_path_to=None,
    vis_color_nodes=(),
    vis_theme="regular",
    vis_force_labels_left=(),
    vis_leaf_images=False,
    vis_image_resize_factor=1,
    vis_fake_sublabels=False,
    vis_zoom=2,
    vis_curved=False,
    vis_sublabels=False,
    vis_height=800,
    vis_width=1000,
    vis_margin_top = 5,
    vis_margin_left=250,
    vis_hide=(),
    vis_above_dy=325,
    vis_below_dy=475,
    vis_scale=1,
    vis_root_y="null",
    vis_colormap="colormap_annotated.png",
    vis_node_conf=(),
    verbose=False,
    **kwargs
):
    """
    :param path_html: Where to write final hierarchy
    """

    roots = list(get_roots(G))
    num_roots = len(roots)
    root = vis_root or next(get_roots(G))

    assert root in G, f"Node {root} is not a valid node. Nodes: {G.nodes}"

    color_info = get_color_info(
        G,
        color,
        color_leaves=not vis_no_color_leaves,
        color_path_to=vis_color_path_to,
        color_nodes=vis_color_nodes or (),
        theme=vis_theme,
    )

    node_to_conf = generate_node_conf(vis_node_conf)

    tree = build_tree(
        G,
        root,
        color_info=color_info,
        force_labels_left=vis_force_labels_left or [],
        dataset=dataset,
        include_leaf_images=vis_leaf_images,
        image_resize_factor=vis_image_resize_factor,
        include_fake_sublabels=vis_fake_sublabels,
        node_to_conf=node_to_conf,
    )
    graph = build_graph(G)

    if num_roots > 1:
        Colors.red(f"Found {num_roots} roots! Should be only 1: {roots}")
    elif verbose:
        print(f"Found just {num_roots} root.")

    parent = Path(fwd()).parent
    generate_vis(
        str(parent / "nbdt/templates/tree-template.html"),
        tree,
        path_html,
        zoom=vis_zoom,
        straight_lines=not vis_curved,
        show_sublabels=vis_sublabels,
        height=vis_height,
        bg=color_info["bg"],
        text_rect=color_info["text_rect"],
        width=vis_width,
        margin_top=vis_margin_top,
        margin_left=vis_margin_left,
        hide=vis_hide or [],
        above_dy=vis_above_dy,
        below_dy=vis_below_dy,
        scale=vis_scale,
        root_y=vis_root_y,
        colormap=vis_colormap,
        verbose=verbose,
    )

# if __name__ == '__main__':
#     --vis - sublabels - -vis - zoom = 1.25 - -dataset = CIFAR10