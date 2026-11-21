"""BVH construction and maintenance contracts.

Geometry uses right-handed XYZ coordinates.  A triangle is a list of three
``(x, y, z)`` vertices.  An AABB is ``{"min": [x, y, z], "max": [x, y, z]}``.
Every geometric comparison in the test suite uses an absolute tolerance of
``1e-6``.

The tree is a flat node pool::

    {
        "root": 0,
        "leaf_size": int,
        "nodes": [node, ...],
    }

and every node has this shape::

    {
        "aabb": {"min": [x, y, z], "max": [x, y, z]},
        "count": int,                 # triangles in this subtree
        "leaf": bool,
        "left": int | None,           # child node index for internal nodes
        "right": int | None,          # child node index for internal nodes
        "indices": [int, ...],         # leaf only, indices into original tris
        "triangles": [triangle, ...],  # leaf only, aligned with indices
    }

Internal nodes have empty ``indices``/``triangles`` and two valid child node
indices.  Leaves have no children.  A leaf may hold between 1 and
``leaf_size`` triangles, inclusive.
"""

import math


class BvhError(ValueError):
    """Raised when geometry, a tree, or batch configuration is invalid."""


# Number of spatial bins used by the tree builder when scoring splits.
_BUILD_BINS = 16


# ---------------------------------------------------------------------------
# validation / geometry helpers
# ---------------------------------------------------------------------------

def _checked_triangle(tri):
    """Return ``tri`` normalised to a tuple of three ``(x, y, z)`` float
    vertices, or raise :class:`BvhError` if it is malformed."""
    try:
        verts = list(tri)
    except TypeError:
        raise BvhError(f"malformed triangle: {tri!r} is not a sequence") from None
    if len(verts) != 3:
        raise BvhError(
            f"malformed triangle: expected 3 vertices, got {len(verts)}"
        )
    out = []
    for vertex in verts:
        try:
            coords = list(vertex)
        except TypeError:
            raise BvhError(
                f"malformed triangle: vertex {vertex!r} is not a sequence"
            ) from None
        if len(coords) != 3:
            raise BvhError(
                f"malformed triangle: vertex {vertex!r} does not have 3 coords"
            )
        try:
            out.append(tuple(float(c) for c in coords))
        except (TypeError, ValueError):
            raise BvhError(
                f"malformed triangle: vertex {vertex!r} has non-numeric coords"
            ) from None
    return tuple(out)


def _checked_tris(tris):
    """Return ``tris`` as a list of normalised triangles; raise on empty or
    malformed input."""
    if isinstance(tris, (str, bytes)):
        raise BvhError("tris must be a non-empty sequence of triangles")
    try:
        items = list(tris)
    except TypeError:
        raise BvhError("tris must be a non-empty sequence of triangles") from None
    if not items:
        raise BvhError("tris must be a non-empty sequence of triangles")
    return [_checked_triangle(tri) for tri in items]


def _check_leaf_size(leaf_size):
    if isinstance(leaf_size, bool) or not isinstance(leaf_size, int) or leaf_size < 1:
        raise BvhError(
            f"leaf_size must be a positive integer, got {leaf_size!r}"
        )
    return leaf_size


def _centroid(tri):
    return (
        (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
        (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
        (tri[0][2] + tri[1][2] + tri[2][2]) / 3.0,
    )


def _aabb_of(checked_tris):
    """Exact AABB over every vertex of the normalised triangles."""
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for tri in checked_tris:
        for vertex in tri:
            for axis in range(3):
                if vertex[axis] < mins[axis]:
                    mins[axis] = vertex[axis]
                if vertex[axis] > maxs[axis]:
                    maxs[axis] = vertex[axis]
    return {"min": mins, "max": maxs}


def _union(a, b):
    return {
        "min": [min(a["min"][i], b["min"][i]) for i in range(3)],
        "max": [max(a["max"][i], b["max"][i]) for i in range(3)],
    }


def _surface_area(aabb):
    dx = aabb["max"][0] - aabb["min"][0]
    dy = aabb["max"][1] - aabb["min"][1]
    dz = aabb["max"][2] - aabb["min"][2]
    return 2.0 * (dx * dy + dy * dz + dz * dx)


def _choose_axis_of(checked_tris):
    centroids = [_centroid(tri) for tri in checked_tris]
    best = 0
    best_spread = -1.0
    for axis in range(3):
        values = [c[axis] for c in centroids]
        spread = max(values) - min(values)
        if spread > best_spread:  # strict: ties keep the lower axis index
            best = axis
            best_spread = spread
    return best


def _depth_budget(count, leaf_size):
    """Maximum allowed depth of a (sub)tree holding ``count`` triangles."""
    return math.ceil(math.log2(max(1.0, count / leaf_size))) + 1


# ---------------------------------------------------------------------------
# public contracts
# ---------------------------------------------------------------------------

def compute_aabb(tris):
    """Return the exact axis-aligned bounding box of all triangle vertices.

    ``tris`` is a non-empty sequence of triangles, each being three ``(x, y,
    z)`` vertices.  The result is a dict with ``"min"`` and ``"max"`` lists of
    three floats.  Bounds are the extrema over every vertex; triangle centres
    or box centres must never be used.  Degenerate (flat) boxes are valid, and
    comparisons are expected to use an absolute tolerance of ``1e-6``.

    Raises:
        BvhError: if ``tris`` is empty or contains a malformed triangle.
    """
    return _aabb_of(_checked_tris(tris))


def sah_cost(left, right, parent_area):
    """Return the Surface Area Heuristic cost of a two-way split.

    ``left`` and ``right`` are partition summaries shaped as
    ``{"aabb": AABB, "count": int}`` where ``count`` is the number of
    triangles on that side.  ``parent_area`` is the surface area of the
    parent AABB.  Surface area means the true box area
    ``2 * (dx*dy + dy*dz + dz*dx)``; volume or half-area must not be used.

    The returned cost is::

        1.0 + (area(left) * count(left) + area(right) * count(right))
              / parent_area

    If ``parent_area`` is zero (or negative), the area ratio is zero and the
    function returns ``0.0``.  Results are compared with an absolute tolerance
    of ``1e-6``.
    """
    if parent_area <= 0:
        return 0.0
    return 1.0 + (
        _surface_area(left["aabb"]) * left["count"]
        + _surface_area(right["aabb"]) * right["count"]
    ) / parent_area


def choose_axis(tris):
    """Return the split axis (0, 1, or 2) for ``tris``.

    The axis is the one with the largest spread of triangle centroids
    (``max centroid - min centroid``).  Ties choose the lower axis index.
    ``tris`` must be a non-empty sequence of triangles.

    Raises:
        BvhError: if ``tris`` is empty or malformed.
    """
    return _choose_axis_of(_checked_tris(tris))


def _split_items(items, axis, bins):
    """Spatial-bin split of ``(key, triangle)`` pairs.

    Returns ``(left_items, right_items)``; both sides are non-empty, every
    item occurs exactly once, and the original relative order is preserved
    inside each side.
    """
    centroids = [_centroid(tri)[axis] for _, tri in items]
    cmin = min(centroids)
    cmax = max(centroids)
    if not cmax > cmin:
        # All centroids identical: stable index fallback, still two groups.
        mid = len(items) // 2
        return items[:mid], items[mid:]
    width = (cmax - cmin) / bins
    bin_ids = []
    for c in centroids:
        b = int((c - cmin) / width)
        if b >= bins:
            b = bins - 1
        elif b < 0:
            b = 0
        bin_ids.append(b)
    parent_area = _surface_area(_aabb_of([tri for _, tri in items]))
    best_cost = None
    best_boundary = None
    lo = min(bin_ids)
    hi = max(bin_ids)
    for boundary in range(lo + 1, hi + 1):
        left = [it for it, b in zip(items, bin_ids) if b < boundary]
        right = [it for it, b in zip(items, bin_ids) if b >= boundary]
        if not left or not right:
            continue
        cost = sah_cost(
            {"aabb": _aabb_of([t for _, t in left]), "count": len(left)},
            {"aabb": _aabb_of([t for _, t in right]), "count": len(right)},
            parent_area,
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_boundary = boundary
    if best_boundary is None:
        mid = len(items) // 2
        return items[:mid], items[mid:]
    left = [it for it, b in zip(items, bin_ids) if b < best_boundary]
    right = [it for it, b in zip(items, bin_ids) if b >= best_boundary]
    return left, right


def split_node(tris, axis, bins):
    """Partition triangles into two non-empty spatial bin groups.

    Triangles are binned by the position of their centroid along ``axis``
    using ``bins`` equal-width bins.  Candidate boundaries are scored with
    :func:`sah_cost` (surface area based) and the cheapest split is returned
    as ``(left, right)``.  Every input triangle occurs exactly once, the
    original relative order inside each side is preserved, and both sides must
    be non-empty.  If every centroid is identical, a stable index fallback
    still produces two non-empty groups.

    ``axis`` must be 0, 1, or 2.  ``bins`` must be at least 2.  At least two
    triangles are required.

    Raises:
        BvhError: for an invalid axis, ``bins < 2``, fewer than two
            triangles, or malformed triangles.
    """
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise BvhError(f"axis must be 0, 1, or 2, got {axis!r}")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise BvhError(f"bins must be an integer >= 2, got {bins!r}")
    checked = _checked_tris(tris)
    if len(checked) < 2:
        raise BvhError("split_node requires at least two triangles")
    left, right = _split_items(list(enumerate(checked)), axis, bins)
    return [t for _, t in left], [t for _, t in right]


def _median_split(items, axis):
    """Depth-guard fallback: split in half by centroid along ``axis``."""
    ordered = sorted(items, key=lambda it: _centroid(it[1])[axis])
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def build_tree(tris, leaf_size):
    """Recursively build a flat BVH over ``tris``.

    ``leaf_size`` must be a positive integer.  Every leaf contains between 1
    and ``leaf_size`` triangles.  Internal nodes have exactly two non-empty
    children, and each node AABB is the tight union of its subtree geometry.
    No empty/orphan nodes may be produced.  The number of leaves is
    ``ceil(len(tris) / leaf_size)``.  Tree depth is at most
    ``ceil(log2(max(1, len(tris) / leaf_size))) + 1`` for non-empty input.

    Returns the tree dictionary documented in the module docstring.

    Raises:
        BvhError: if ``tris`` is empty, ``leaf_size`` is not a positive
            integer, or a triangle is malformed.
    """
    _check_leaf_size(leaf_size)
    checked = _checked_tris(tris)
    n = len(checked)
    max_depth = _depth_budget(n, leaf_size)

    # Iterative build with an explicit stack: no recursion, so deep or
    # degenerate inputs cannot overflow the interpreter stack.
    nodes = [None]
    stack = [(list(enumerate(checked)), 0, 1)]  # (items, node index, depth)
    while stack:
        items, index, depth = stack.pop()
        tris_here = [t for _, t in items]
        aabb = _aabb_of(tris_here)
        count = len(items)
        if count <= leaf_size:
            nodes[index] = {
                "aabb": aabb,
                "count": count,
                "leaf": True,
                "left": None,
                "right": None,
                "indices": [i for i, _ in items],
                "triangles": list(tris_here),
            }
            continue
        axis = _choose_axis_of(tris_here)
        left_items, right_items = _split_items(items, axis, _BUILD_BINS)
        remaining = max_depth - depth
        if (
            _depth_budget(len(left_items), leaf_size) > remaining
            or _depth_budget(len(right_items), leaf_size) > remaining
        ):
            # SAH split is too unbalanced for the depth budget; halve instead.
            left_items, right_items = _median_split(items, axis)
        left_index = len(nodes)
        nodes.append(None)
        right_index = len(nodes)
        nodes.append(None)
        nodes[index] = {
            "aabb": aabb,
            "count": count,
            "leaf": False,
            "left": left_index,
            "right": right_index,
            "indices": [],
            "triangles": [],
        }
        stack.append((right_items, right_index, depth + 1))
        stack.append((left_items, left_index, depth + 1))
    return {"root": 0, "leaf_size": leaf_size, "nodes": nodes}


def refit(tree, moved):
    """Update AABBs affected by moved triangles without rebuilding topology.

    ``moved`` is a mapping ``{triangle_index: new_triangle}`` where indices
    refer to the order passed to :func:`build_tree`.  For every leaf that owns
    a moved index, replace its triangle copy and recompute that leaf AABB.
    Recompute every ancestor AABB as the union of its two children; untouched
    leaves must keep their exact prior AABB.  The structural node layout and
    all indices remain unchanged.  Returns the same tree object.  Comparisons
    use an absolute tolerance of ``1e-6``.

    Raises:
        BvhError: for an unknown triangle index, malformed replacement
            triangle, or an invalid tree.
    """
    if not isinstance(tree, dict):
        raise BvhError("invalid tree: expected a dict")
    nodes = tree.get("nodes")
    root = tree.get("root")
    if (
        not isinstance(nodes, list)
        or not nodes
        or isinstance(root, bool)
        or not isinstance(root, int)
        or not 0 <= root < len(nodes)
    ):
        raise BvhError("invalid tree: bad root or node pool")
    try:
        pairs = list(moved.items())
    except AttributeError:
        raise BvhError("moved must be a mapping of index to triangle") from None
    normed = {}
    for idx, tri in pairs:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise BvhError(f"unknown triangle index: {idx!r}")
        normed[idx] = _checked_triangle(tri)

    # Defensive traversal: parents are recorded before children, so the
    # reversed order below recomputes children before their ancestors.
    order = []
    seen = set()
    stack = [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        node = nodes[i]
        if not isinstance(node, dict):
            raise BvhError(f"invalid tree: node {i} is malformed")
        order.append(i)
        if node.get("leaf"):
            continue
        for side in ("left", "right"):
            child = node.get(side)
            if (
                isinstance(child, bool)
                or not isinstance(child, int)
                or not 0 <= child < len(nodes)
            ):
                raise BvhError(
                    f"invalid tree: node {i} has bad {side} child {child!r}"
                )
            stack.append(child)

    present = set()
    for i in order:
        node = nodes[i]
        if node.get("leaf"):
            present.update(node.get("indices") or [])
    for idx in normed:
        if idx not in present:
            raise BvhError(f"unknown triangle index: {idx}")

    # Replace moved triangle copies and recompute each affected leaf AABB
    # from the moved triangles it owns.
    for i in order:
        node = nodes[i]
        if not node.get("leaf"):
            continue
        indices = node.get("indices") or []
        moved_here = [t for t in indices if t in normed]
        if not moved_here:
            continue
        triangles = node.get("triangles")
        if not isinstance(triangles, list) or len(triangles) != len(indices):
            raise BvhError(f"invalid tree: leaf {i} indices/triangles mismatch")
        for pos, t in enumerate(indices):
            if t in normed:
                triangles[pos] = normed[t]
        node["aabb"] = _aabb_of([normed[t] for t in moved_here])

    # Recompute every internal AABB bottom-up as the union of its children.
    # Untouched subtrees recompute to exactly their prior bounds.
    for i in reversed(order):
        node = nodes[i]
        if node.get("leaf"):
            continue
        node["aabb"] = _union(
            nodes[node["left"]]["aabb"], nodes[node["right"]]["aabb"]
        )
    return tree


def validate_tree(tree, tris):
    """Return a list of structural problems found in ``tree``, or ``[]``.

    ``tris`` is the original triangle sequence used to build the tree.  The
    check is defensive: malformed child links are reported instead of being
    followed, so this function must not raise ``IndexError`` for a corrupted
    tree.  At least these checks are required:

    * every internal node's ``left`` and ``right`` are valid node indices;
    * every leaf's triangle count is in ``1..leaf_size`` and its
      ``indices``/``triangles`` lengths agree;
    * every stored triangle index is within ``tris``;
    * there are no empty nodes (``count <= 0``) and no unreachable nodes.

    Problem strings are intended for humans but must start with these stable
    prefixes:

    * ``"child out of bounds:"``
    * ``"leaf over capacity:"``
    * ``"empty node:"``

    The result is an empty list for a well-formed tree.
    """
    problems = []
    if not isinstance(tree, dict):
        return ["invalid tree: expected a dict"]
    nodes = tree.get("nodes")
    root = tree.get("root")
    leaf_size = tree.get("leaf_size")
    if not isinstance(nodes, list):
        return ["invalid tree: nodes is not a list"]
    n = len(nodes)
    try:
        n_tris = len(tris)
    except TypeError:
        n_tris = 0
        problems.append("invalid tris: not a sized sequence")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            problems.append(f"malformed node: {i}")
            continue
        count = node.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            problems.append(f"empty node: {i}")
        if node.get("leaf"):
            indices = node.get("indices") or []
            triangles = node.get("triangles") or []
            if len(indices) != len(triangles):
                problems.append(
                    f"leaf index/triangle mismatch: node {i} has "
                    f"{len(indices)} indices but {len(triangles)} triangles"
                )
            if isinstance(leaf_size, int) and not isinstance(leaf_size, bool):
                held = max(len(indices), count if isinstance(count, int) else 0)
                if held > leaf_size:
                    problems.append(
                        f"leaf over capacity: node {i} holds {held} "
                        f"triangles (leaf_size={leaf_size})"
                    )
            for idx in indices:
                if (
                    not isinstance(idx, int)
                    or isinstance(idx, bool)
                    or idx < 0
                    or idx >= n_tris
                ):
                    problems.append(
                        f"triangle index out of range: node {i} holds {idx!r}"
                    )
        else:
            for side in ("left", "right"):
                child = node.get(side)
                if (
                    not isinstance(child, int)
                    or isinstance(child, bool)
                    or not 0 <= child < n
                ):
                    problems.append(
                        f"child out of bounds: node {i} {side}={child!r}"
                    )

    # Reachability: follow only in-bounds child links, never raising.
    seen = set()
    if isinstance(root, int) and not isinstance(root, bool) and 0 <= root < n:
        stack = [root]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            node = nodes[i]
            if isinstance(node, dict) and not node.get("leaf"):
                for side in ("left", "right"):
                    child = node.get(side)
                    if (
                        isinstance(child, int)
                        and not isinstance(child, bool)
                        and 0 <= child < n
                    ):
                        stack.append(child)
    else:
        problems.append(f"child out of bounds: root={root!r}")
    for i in range(n):
        if i not in seen:
            problems.append(f"unreachable node: {i}")
    return problems


def build_batch(meshes, config):
    """Build one independent BVH per mesh, transactionally.

    ``meshes`` is a sequence of ``{"name": str, "tris": [triangle, ...]}``
    dictionaries.  ``config`` is ``{"leaf_size": positive_int}``.  The return
    value is::

        {
            "leaf_size": int,
            "trees": [
                {"name": str, "tree": tree_dict},
                ...  # exactly the input order
            ],
        }

    Every mesh gets its own node pool; no nodes, lists, or tree dictionaries
    may be shared between meshes.  If any mesh, leaf size, or triangle is
    invalid, raise ``BvhError`` and leave no partial result or shared mutable
    state behind (all-or-nothing rollback).  Successful builds preserve the
    input mesh order.
    """
    if not isinstance(config, dict):
        raise BvhError("config must be a dict with a positive integer leaf_size")
    _check_leaf_size(config.get("leaf_size"))
    leaf_size = config["leaf_size"]
    if isinstance(meshes, (str, bytes)) or not isinstance(meshes, (list, tuple)):
        raise BvhError("meshes must be a sequence of mesh dictionaries")
    trees = []
    for mesh in meshes:
        if not isinstance(mesh, dict):
            raise BvhError("each mesh must be a dict with name and tris")
        name = mesh.get("name")
        if not isinstance(name, str):
            raise BvhError("each mesh must have a string name")
        if "tris" not in mesh:
            raise BvhError(f"mesh {name!r} is missing tris")
        # Each build allocates a fresh node pool; on any BvhError the partial
        # results stay local to this call and are discarded (rollback).
        tree = build_tree(mesh["tris"], leaf_size)
        trees.append({"name": name, "tree": tree})
    return {"leaf_size": leaf_size, "trees": trees}
