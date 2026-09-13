"""Stage 1: find candidate ostia on the aortic wall.

Idea: everything that leaves the aorta must pass through a thin shell just
outside the mask. Bright blobs in that shell are branch candidates.
"""
import numpy as np
from scipy import ndimage as ndi
from .geometry import terminal_frames


def wall_field(vol, collar_mm=4.0, inner_mm=1.5):
    """Distance-from-aorta field and outward normals.

    Returns
    -------
    dist : (z,y,x) float32, mm outside the aorta (0 inside)
    nrm  : (3,z,y,x) float32, unit normal in physical image axes (z,y,x)
    collar : (z,y,x) bool, inner_mm < dist <= inner_mm + collar_mm
    """
    dist = ndi.distance_transform_edt(~vol.aorta,
                                      sampling=vol.spacing_zyx).astype(np.float32)
    g = np.gradient(dist, *vol.spacing_zyx)          # d/dz, d/dy, d/dx
    nrm = np.stack(g).astype(np.float32)
    mag = np.linalg.norm(nrm, axis=0)
    mag[mag == 0] = 1.0
    nrm /= mag
    # NOTE: inner_mm is not cosmetic. The supplied mask sits INSIDE the true
    # lumen, so a thin bright rim hugs it all the way round. Starting the
    # collar at dist=0 links every branch into one 6000-voxel shell component.
    collar = (dist > inner_mm) & (dist <= inner_mm + collar_mm)
    return dist, nrm, collar


def cap_mask(vol, nrm, cap_zone_mm=6.0, normal_tol=0.6, collar=None):
    """Reject outward-facing surfaces near the two mask-derived endpoints.

    Each end has its own tangent, so a curved or oblique aorta does not need
    to align with any array axis. Ambiguous skeletons produce no cap filter.
    """
    out = np.zeros(vol.aorta.shape, bool)
    idx = np.argwhere(~vol.aorta if collar is None else collar)
    points = idx * vol.mm_per_vox
    normals = nrm[:, idx[:, 0], idx[:, 1], idx[:, 2]].T
    reject = np.zeros(len(idx), bool)
    for centre, tangent, radius in terminal_frames(vol):
        delta = points - centre
        along = delta @ tangent
        lateral = np.linalg.norm(delta - along[:, None] * tangent, axis=1)
        near_end = (np.abs(along) <= cap_zone_mm) & (lateral <= radius + cap_zone_mm)
        reject |= near_end & (normals @ tangent > normal_tol)
    out[tuple(idx.T)] = reject
    return out


def candidates(vol, thr, ceil, collar_mm=4.0, inner_mm=1.5,
               min_area_mm2=2.0, cap_kwargs=None):
    """Connected bright blobs in the collar -> candidate branch origins.

    Returns a list of dicts with the component voxel indices and a first-pass
    ostium/direction estimate. Refinement happens in tracing.py.
    """
    dist, nrm, collar = wall_field(vol, collar_mm, inner_mm)
    bright = (vol.ct > thr) & (vol.ct < ceil)
    caps = cap_mask(vol, nrm, collar=collar, **(cap_kwargs or {}))
    cand = collar & bright & ~caps

    # one dilation-erosion to bridge single-voxel noise gaps in the lumen
    cand = ndi.binary_closing(cand, structure=np.ones((3, 3, 3)))
    cand &= collar & ~caps

    lab, n = ndi.label(cand, structure=np.ones((3, 3, 3)))
    if n == 0:
        return [], dist, nrm

    vox_area = vol.spacing_zyx[1] * vol.spacing_zyx[2]
    min_vox = max(3, int(min_area_mm2 / vox_area))

    out = []
    for i in range(1, n + 1):
        idx = np.argwhere(lab == i)
        if len(idx) < min_vox:
            continue
        d = dist[idx[:, 0], idx[:, 1], idx[:, 2]]

        # ostium = centre of the part of the blob that touches the wall
        foot = idx[d <= np.percentile(d, 25)]
        ostium = foot.mean(axis=0)
        # project back onto the mask wall along the inward normal
        fi = np.rint(ostium).astype(int)
        fi = np.clip(fi, 0, np.array(dist.shape) - 1)
        n_here = nrm[:, fi[0], fi[1], fi[2]]
        ostium = ostium - n_here * (dist[fi[0], fi[1], fi[2]] / vol.mm_per_vox)

        # first-pass direction: wall -> far end of the blob
        tip = idx[d >= np.percentile(d, 90)].mean(axis=0)
        v = tip - ostium
        if np.linalg.norm(v * vol.mm_per_vox) < 1e-6:
            continue

        out.append({
            "voxels": idx,
            "ostium_vox": ostium,
            "dir_vox": v,
            "n_vox": len(idx),
        })
    return out, dist, nrm
