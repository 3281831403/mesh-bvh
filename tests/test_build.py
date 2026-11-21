import math

import pytest

from bvh import (
    BvhError,
    build_batch,
    build_tree,
    choose_axis,
    compute_aabb,
    refit,
    sah_cost,
    split_node,
    validate_tree,
)

EPS = 1e-6


def tri_at(x, y=0.0, z=0.0):
    """Small asymmetric triangle whose AABB starts at ``(x, y, z)``."""
    x = float(x)
    y = float(y)
    z = float(z)
    return [
        (x, y, z),
        (x + 1.0, y, z),
        (x, y + 1.0, z + 0.5),
    ]


def box(x0, y0, z0, x1, y1, z1):
    return {
        "min": [float(x0), float(y0), float(z0)],
        "max": [float(x1), float(y1), float(z1)],
    }


def _reachable_nodes(tree):
    nodes = tree["nodes"]
    stack = [tree["root"]]
    seen = set()
    ordered = []
    while stack:
        index = stack.pop()
        assert 0 <= index < len(nodes)
        if index in seen:
            continue
        seen.add(index)
        node = nodes[index]
        ordered.append(node)
        if not node["leaf"]:
            stack.append(node["right"])
            stack.append(node["left"])
    return ordered


def _leaves(tree):
    return [node for node in _reachable_nodes(tree) if node["leaf"]]


def _depth(tree):
    nodes = tree["nodes"]

    def walk(index):
        node = nodes[index]
        if node["leaf"]:
            return 1
        return 1 + max(walk(node["left"]), walk(node["right"]))

    return walk(tree["root"])


def _find_leaf(tree, triangle_index):
    for leaf in _leaves(tree):
        if triangle_index in leaf["indices"]:
            return leaf
    raise AssertionError("triangle index is not present in any leaf")


def _assert_encloses(parent_aabb, child_aabb):
    for axis in range(3):
        assert parent_aabb["min"][axis] <= child_aabb["min"][axis] + EPS
        assert parent_aabb["max"][axis] >= child_aabb["max"][axis] - EPS


def test_aabb_uses_vertex_extents():
    triangles = [
        [(-2.5, -1.0, -0.25), (3.0, 2.0, 0.5), (0.0, 4.0, 1.5)],
        [(5.0, -3.0, 2.0), (1.0, 0.0, -2.0), (-4.0, 2.5, 3.5)],
    ]

    bounds = compute_aabb(triangles)

    assert bounds["min"] == pytest.approx([-4.0, -3.0, -2.0], abs=EPS)
    assert bounds["max"] == pytest.approx([5.0, 4.0, 3.5], abs=EPS)


def test_sah_cost_uses_surface_area():
    left = {"aabb": box(0, 0, 0, 1, 1, 1), "count": 1}
    right = {"aabb": box(0, 0, 0, 2, 1, 1), "count": 3}

    # Areas are 6 and 10; the 3x3x3 parent has area 54.
    expected = 1.0 + (6.0 * 1 + 10.0 * 3) / 54.0
    assert sah_cost(left, right, 54.0) == pytest.approx(expected, abs=EPS)
    assert sah_cost(left, right, 0.0) == 0.0


def test_split_node_uses_spatial_bins():
    x_positions = [-4.0, 4.0, -3.0, 3.0, -2.0, 2.0, -1.0, 1.0]
    triangles = [tri_at(10.0 * x) for x in x_positions]

    axis = choose_axis(triangles)
    left, right = split_node(triangles, axis, bins=4)

    assert axis == 0
    assert len(left) == 4
    assert len(right) == 4
    assert all(vertex[0] < 0.0 for triangle in left for vertex in triangle)
    assert all(vertex[0] > 0.0 for triangle in right for vertex in triangle)
    assert sorted(
        triangle[0][0] for triangle in left + right
    ) == sorted(triangle[0][0] for triangle in triangles)


def test_split_node_keeps_both_sides_non_empty():
    triangles = [tri_at(5.0, 5.0, 5.0) for _ in range(6)]

    left, right = split_node(triangles, axis=0, bins=8)

    assert len(left) > 0
    assert len(right) > 0
    assert len(left) + len(right) == len(triangles)


def test_leaf_capacity_is_enforced():
    triangles = [tri_at(10.0 * index) for index in range(5)]

    tree = build_tree(triangles, leaf_size=3)
    leaves = _leaves(tree)

    assert len(leaves) == 2
    assert all(1 <= len(leaf["indices"]) <= 3 for leaf in leaves)
    assert all(len(leaf["triangles"]) == len(leaf["indices"]) for leaf in leaves)
    assert sorted(index for leaf in leaves for index in leaf["indices"]) == list(
        range(5)
    )


def test_refit_updates_descendant_aabbs():
    triangles = [tri_at(10.0 * index) for index in range(4)]
    tree = build_tree(triangles, leaf_size=2)
    moved_triangle = [
        (1000.0, 0.0, 0.0),
        (1001.0, 0.5, 0.25),
        (1000.0, 1.0, 0.5),
    ]

    result = refit(tree, {3: moved_triangle})

    assert result is tree
    leaf = _find_leaf(tree, 3)
    position = leaf["indices"].index(3)
    assert [tuple(vertex) for vertex in leaf["triangles"][position]] == [
        tuple(vertex) for vertex in moved_triangle
    ]
    assert leaf["aabb"]["min"][0] == pytest.approx(1000.0, abs=EPS)
    assert leaf["aabb"]["max"][0] == pytest.approx(1001.0, abs=EPS)

    nodes = tree["nodes"]
    root = nodes[tree["root"]]
    assert root["aabb"]["max"][0] == pytest.approx(1001.0, abs=EPS)
    for node in _reachable_nodes(tree):
        if not node["leaf"]:
            _assert_encloses(node["aabb"], nodes[node["left"]]["aabb"])
            _assert_encloses(node["aabb"], nodes[node["right"]]["aabb"])


def test_tree_depth_is_logarithmic():
    triangles = [
        tri_at(float(index % 8) * 10.0, float(index // 8) * 10.0)
        for index in range(64)
    ]

    tree = build_tree(triangles, leaf_size=2)

    limit = math.ceil(math.log2(len(triangles) / 2.0)) + 1
    assert _depth(tree) <= limit


def test_validate_tree_reports_child_out_of_bounds():
    triangles = [tri_at(0.0), tri_at(10.0), tri_at(20.0)]
    root = {
        "aabb": box(0, 0, 0, 21, 1, 0.5),
        "count": 3,
        "leaf": False,
        "left": 1,
        "right": 9,
        "indices": [],
        "triangles": [],
    }
    overfull_leaf = {
        "aabb": box(0, 0, 0, 21, 1, 0.5),
        "count": 3,
        "leaf": True,
        "left": None,
        "right": None,
        "indices": [0, 1, 2],
        "triangles": triangles,
    }
    empty_leaf = {
        "aabb": box(0, 0, 0, 0, 0, 0),
        "count": 0,
        "leaf": True,
        "left": None,
        "right": None,
        "indices": [],
        "triangles": [],
    }
    broken_tree = {
        "root": 0,
        "leaf_size": 2,
        "nodes": [root, overfull_leaf, empty_leaf],
    }

    problems = validate_tree(broken_tree, triangles)

    assert any(problem.startswith("child out of bounds:") for problem in problems)
    assert any(problem.startswith("leaf over capacity:") for problem in problems)
    assert any(problem.startswith("empty node:") for problem in problems)


def test_batch_isolates_and_rolls_back():
    mesh_a = {"name": "a", "tris": [tri_at(0.0), tri_at(5.0)]}
    mesh_b = {"name": "b", "tris": [tri_at(100.0)]}

    built = build_batch([mesh_a, mesh_b], {"leaf_size": 1})
    entries = built["trees"]

    assert len(entries) == 2
    assert entries[0]["tree"] is not entries[1]["tree"]
    assert entries[0]["tree"]["nodes"] is not entries[1]["tree"]["nodes"]

    tree_a = entries[0]["tree"]
    tree_b = entries[1]["tree"]
    root_a = tree_a["nodes"][tree_a["root"]]
    root_b = tree_b["nodes"][tree_b["root"]]
    assert root_a["aabb"]["max"][0] < 50.0
    assert root_b["aabb"]["min"][0] > 50.0
    assert sorted(index for leaf in _leaves(tree_a) for index in leaf["indices"]) == [
        0,
        1,
    ]
    assert [index for leaf in _leaves(tree_b) for index in leaf["indices"]] == [0]

    with pytest.raises(BvhError):
        build_batch(
            [mesh_a, {"name": "bad", "tris": []}],
            {"leaf_size": 1},
        )

    rebuilt = build_batch([mesh_a, mesh_b], {"leaf_size": 1})
    assert [entry["name"] for entry in rebuilt["trees"]] == ["a", "b"]


def test_zero_leaf_size_is_rejected():
    with pytest.raises(BvhError):
        build_tree([tri_at(0.0)], leaf_size=0)

    with pytest.raises(BvhError):
        build_tree([tri_at(0.0)], leaf_size=-1)


def test_single_triangle_has_one_leaf():
    tree = build_tree([tri_at(2.0, 3.0, 4.0)], leaf_size=8)
    leaves = _leaves(tree)

    assert tree["root"] == 0
    assert len(tree["nodes"]) == 1
    assert len(leaves) == 1
    assert leaves[0]["leaf"] is True
    assert leaves[0]["count"] == 1
    assert leaves[0]["indices"] == [0]
    assert leaves[0]["aabb"]["min"] == pytest.approx([2.0, 3.0, 4.0], abs=EPS)
    assert leaves[0]["aabb"]["max"] == pytest.approx([3.0, 4.0, 4.5], abs=EPS)


def test_large_leaf_size_makes_root_leaf():
    triangles = [tri_at(0.0), tri_at(10.0), tri_at(20.0)]

    tree = build_tree(triangles, leaf_size=99)
    root = tree["nodes"][tree["root"]]

    assert len(tree["nodes"]) == 1
    assert root["leaf"] is True
    assert root["left"] is None
    assert root["right"] is None
    assert root["count"] == 3
    assert tree["leaf_size"] == 99


def test_empty_geometry_raises_bvh_error():
    with pytest.raises(BvhError):
        compute_aabb([])

    with pytest.raises(BvhError):
        build_tree([], leaf_size=4)


def test_leaf_count_is_ceil_n_over_leaf_size():
    triangles = [tri_at(10.0 * index) for index in range(4)]

    tree = build_tree(triangles, leaf_size=2)

    assert len(_leaves(tree)) == math.ceil(len(triangles) / 2)


def test_batch_preserves_mesh_order():
    names = ["zeta", "alpha", "mu"]
    meshes = [
        {"name": names[0], "tris": [tri_at(0.0)]},
        {"name": names[1], "tris": [tri_at(50.0)]},
        {"name": names[2], "tris": [tri_at(100.0)]},
    ]

    built = build_batch(meshes, {"leaf_size": 2})

    assert built["leaf_size"] == 2
    assert [entry["name"] for entry in built["trees"]] == names
    for entry in built["trees"]:
        tree = entry["tree"]
        root = tree["nodes"][tree["root"]]
        assert root["leaf"] is True
        assert root["count"] == 1
