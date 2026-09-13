"""Mask-derived aorta endpoints and local tangents, in physical image axes.

No anatomical labels or trained parameters are needed. An isotropic skeleton
supplies the main geodesic path; only its two terminal portions are used.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _straight_path(mask, spacing, bin_mm):
    """Fallback for thin/symmetric tubes erased by voxel skeleton thinning.

    Only use PCA when the mask is strongly elongated. Centroids of successive
    perpendicular bins form the path; this fallback is not used for broad,
    curved masks whose local direction cannot be inferred this way.
    """
    points = np.argwhere(mask) * spacing
    if len(points) < 3:
        return np.empty((0, 3))
    centre = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((points - centre).T))
    if values[-1] < 16 * max(values[-2], 1e-9):
        return np.empty((0, 3))
    axis = vectors[:, -1]
    projection = (points - centre) @ axis
    bins = np.floor((projection - projection.min()) / bin_mm).astype(int)
    counts = np.bincount(bins)
    sums = np.column_stack([np.bincount(bins, weights=points[:, i]) for i in range(3)])
    return sums[counts > 0] / counts[counts > 0, None]


def main_path(mask, spacing, grid_mm=1.0):
    """Approximate skeleton diameter, returned in physical (z,y,x) mm.

    Work on the largest mask component, then the largest skeleton component.
    Return an empty path when no reliable terminal path is available.
    """
    lab, count = ndi.label(mask)
    if count == 0:
        return np.empty((0, 3))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    mask = lab == sizes.argmax()
    spacing = np.asarray(spacing, float)
    # Explicit coordinates avoid zoom's shape-dependent effective spacing.
    shape = np.ceil((np.array(mask.shape) - 1) * spacing / grid_mm).astype(int) + 1
    coords = np.meshgrid(*(np.arange(n) * grid_mm / s
                           for n, s in zip(shape, spacing)), indexing="ij", sparse=True)
    iso = ndi.map_coordinates(mask.astype(np.uint8), np.broadcast_arrays(*coords),
                              order=0, mode="constant") > 0
    points = np.argwhere(skeletonize(np.pad(iso, 1), method="lee")) - 1
    if len(points) < 3:
        return _straight_path(mask, spacing, grid_mm)
    pairs = cKDTree(points).query_pairs(np.sqrt(3) + 1e-6, output_type="ndarray")
    if not len(pairs):
        return _straight_path(mask, spacing, grid_mm)
    weights = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1) * grid_mm
    graph = coo_matrix((np.tile(weights, 2),
                        (np.r_[pairs[:, 0], pairs[:, 1]],
                         np.r_[pairs[:, 1], pairs[:, 0]])),
                       shape=(len(points), len(points))).tocsr()
    _, components = connected_components(graph)
    keep = np.flatnonzero(components == np.bincount(components).argmax())
    graph, points = graph[keep][:, keep], points[keep]
    ends = np.flatnonzero(np.diff(graph.indptr) == 1)
    if len(ends) < 2:
        return _straight_path(mask, spacing, grid_mm)
    distance = dijkstra(graph, indices=int(ends[0]))
    first = int(ends[np.argmax(distance[ends])])
    distance, previous = dijkstra(graph, indices=first, return_predecessors=True)
    last = int(ends[np.argmax(distance[ends])])
    path = [last]
    while path[-1] != first:
        path.append(int(previous[path[-1]]))
    return points[path] * grid_mm


def terminal_frames(vol, tangent_mm=10.0):
    """Return (cap centre, outward local unit tangent, lumen radius) per end.

    Skeleton ends lie inside the lumen. Extend their local tangent until it
    exits the mask to locate the terminal surface. Coordinates are millimetres
    in cropped image axes, not world coordinates or voxel-index vectors.
    """
    sp = np.asarray(vol.spacing_zyx)
    path = main_path(vol.aorta, sp)
    if len(path) < 3:
        return []
    radius = ndi.distance_transform_edt(vol.aorta, sampling=sp)
    frames = []
    for ordered in (path, path[::-1]):
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))]
        ids = np.flatnonzero(arc <= tangent_mm)
        if len(ids) < 3 or arc[-1] < 2 * tangent_mm:
            continue
        local = ordered[ids]
        # Fit coordinates against path distance; the negative slope is outward.
        t = arc[ids] - arc[ids].mean()
        direction = -(t[:, None] * (local - local.mean(axis=0))).sum(axis=0)
        direction /= max(np.linalg.norm(direction), 1e-9)
        r = float(ndi.map_coordinates(radius, (local / sp).T, order=1).max())
        if r <= 0 or np.linalg.norm(direction) < 0.5:
            continue
        step = min(0.5, float(sp.min()) / 2)
        distances = np.arange(0.0, 3 * r + tangent_mm, step)
        ray = ordered[0] + distances[:, None] * direction
        inside = ndi.map_coordinates(vol.aorta.astype(np.uint8), (ray / sp).T,
                                     order=0, mode="constant") > 0
        exits = np.flatnonzero(~inside)
        if not len(exits) or exits[0] == 0:
            continue
        i = exits[0]
        centre = (ray[i - 1] + ray[i]) / 2
        frames.append((centre, direction, r))
    return frames
