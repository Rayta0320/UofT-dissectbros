"""Bounded blood-consistent expansion from the supplied aorta mask.

Inspired by Tahoces et al. (2020), sections 2.1.1-2.1.2. The bounded probe,
thin-rim partition and explicit geometric guards are prototype adaptations,
not a reproduction of the paper's thresholds or anatomical labeling.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from . import detection

FACE = ndi.generate_binary_structure(3, 1)


def neighbourhood(connectivity):
    """Voxel adjacency, separate from physical shared-face opening area."""
    if connectivity not in (6, 18, 26):
        raise ValueError("growth connectivity must be 6, 18 or 26")
    return ndi.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])


def contact_faces(region, aorta, spacing, offset=(0, 0, 0)):
    """Actual shared voxel-face centres, areas and outward normals.

    The opening centre is its physical-area-weighted centroid snapped to the
    closest contact face, so a non-planar opening cannot place it off-surface.
    Arrays can be cropped together; offset maps faces back to volume indices.
    """
    points, weights, normals = [], [], []
    parent = np.pad(aorta, 1)
    for axis in range(3):
        area = float(np.prod(spacing) / spacing[axis])
        for sign in (-1, 1):
            sl = [slice(1, n + 1) for n in aorta.shape]
            sl[axis] = slice(1 + sign, aorta.shape[axis] + 1 + sign)
            idx = np.argwhere(region & parent[tuple(sl)])
            if not len(idx):
                continue
            faces = idx.astype(float) + np.asarray(offset)
            faces[:, axis] += 0.5 * sign
            normal = np.zeros(3)
            normal[axis] = -sign
            points.append(faces)
            weights.extend([area] * len(idx))
            normals.extend([normal] * len(idx))
    if not points:
        return None
    points, weights, normals = np.concatenate(points), np.asarray(weights), np.asarray(normals)
    centroid = np.average(points, axis=0, weights=weights)
    ostium = points[np.argmin(np.linalg.norm((points - centroid) * spacing, axis=1))]
    normal = (normals * weights[:, None]).sum(axis=0) / weights.sum()
    return dict(points_vox=points, centre_vox=ostium, centroid_vox=centroid,
                area_mm2=float(weights.sum()), normal_coherence=float(np.linalg.norm(normal)))


def candidates(vol, model, growth_mm=12.0, probe_mm=5.0, rim_mm=1.5,
               min_volume_mm3=3.0, min_contact_mm2=1.0,
               max_volume_mm3=4000.0, max_expansion_ratio=5.0, connectivity=6,
               proximal_growth_guard=False):
    """Grow locally, separate protrusions, and return candidates plus diagnostics.

    Connectivity is selectable (6/18/26); 6 excludes diagonal-only links.
    Protrusion markers
    come from the grown-minus-parent probe outside the thin common lumen rim.
    A constrained watershed partitions the expanded difference; it cannot add
    voxels or cross gaps under the chosen adjacency. Diagonal connectivity can
    join edge/corner-touching structures without a finite-area opening, so actual
    ostium contact is still measured using shared faces. Rim recovery is bounded
    to avoid assigning the whole aortic
    surface to a single branch. The inferred rim partition remains uncertain
    when the supplied mask substantially undersegments the true lumen.
    """
    if not 0 <= rim_mm < probe_mm <= growth_mm:
        raise ValueError("require 0 <= rim_mm < probe_mm <= growth_mm")
    structure = neighbourhood(connectivity)
    distance, normals, _ = detection.wall_field(vol)
    shell = (distance > 0) & (distance <= growth_mm)
    caps = detection.cap_mask(vol, normals, collar=shell)
    blood = (vol.ct >= model["lower_hu"]) & (vol.ct <= model["upper_hu"])
    allowed = shell & blood & ~caps
    probe = ndi.binary_propagation(vol.aorta, structure=structure,
                                   mask=vol.aorta | (allowed & (distance <= probe_mm)))
    expanded = ndi.binary_propagation(probe, structure=structure, mask=vol.aorta | allowed)
    difference = expanded & ~vol.aorta
    markers, count = ndi.label(probe & ~vol.aorta & (distance > rim_mm), structure=structure)
    # Recover only the nearby common rim. All distal grown voxels remain in
    # scope; labels are kept even for rejected candidates to prevent reassignment.
    if count:
        near_marker = ndi.distance_transform_edt(markers == 0, sampling=vol.spacing_zyx)
        partition_domain = difference & ((distance > rim_mm)
                            | (near_marker <= rim_mm + max(vol.spacing_zyx)))
        labels = watershed(np.zeros(vol.ct.shape, np.uint8), markers,
                           mask=partition_domain, connectivity=structure).astype(np.int32)
    else:
        labels = np.zeros(vol.ct.shape, np.int32)
    voxel_mm3 = float(np.prod(vol.spacing_zyx))
    proposals, attempts = [], []
    objects = ndi.find_objects(labels)
    for label, bounds in enumerate(objects, 1):
        if bounds is None:
            continue
        # Pad each local box by one voxel so its parent contact faces are present.
        sl = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1))
                   for s, n in zip(bounds, vol.ct.shape))
        offset = np.array([s.start for s in sl])
        region = labels[sl] == label
        idx = np.argwhere(region) + offset
        d = distance[tuple(idx.T)]
        volume = float(len(idx) * voxel_mm3)
        metrics = dict(candidate_id=int(label), volume_mm3=volume,
                       probe_mm=float(probe_mm), rim_mm=float(rim_mm),
                       protrusion_volume_mm3=float(np.count_nonzero(d > rim_mm) * voxel_mm3),
                       extent_mm=float(d.max()), accepted=False)
        reason = None
        contact = contact_faces(region, vol.aorta[sl], vol.mm_per_vox, offset)
        if volume < min_volume_mm3 or len(idx) < 3:
            reason = "tiny_region"
        elif np.count_nonzero(d > rim_mm) < 3 or metrics["protrusion_volume_mm3"] < min_volume_mm3:
            reason = "tiny_protrusion"
        elif contact is None:
            reason = "no_surface_contact"
        else:
            area = contact["area_mm2"]
            # Differential growth per one-mm distance shell is an area proxy.
            # Use two-mm aggregated bins to reduce anisotropic voxel noise.
            edges = np.arange(rim_mm, growth_mm + 2.0, 2.0)
            shells = np.histogram(d[d > rim_mm], bins=edges)[0] * voxel_mm3 / 2.0
            proximal = max(float(shells[0]) if len(shells) else 0.0, area, 1e-6)
            expansion = float(max(shells, default=0) / proximal)
            early_shells = shells[((edges[:-1] + edges[1:]) / 2) <= probe_mm]
            # A large first shell must not normalize away an immediate leak.
            proximal_expansion = float(max(early_shells, default=0) / max(area, 1e-6))
            covariance = np.linalg.eigvalsh(np.cov((idx * vol.mm_per_vox).T))
            flatness = float(covariance[1] / max(covariance[0], min(vol.spacing_zyx) ** 2))
            metrics.update(contact_area_mm2=area,
                           contact_normal_coherence=contact["normal_coherence"],
                           expansion_ratio=expansion, shell_area_mm2=shells.tolist(),
                           proximal_expansion_ratio=proximal_expansion,
                           distal_expansion_warning=expansion > max_expansion_ratio,
                           flatness=flatness)
            if area < min_contact_mm2:
                reason = "tiny_contact"
            elif (volume > max_volume_mm3
                  or proximal_expansion > max_expansion_ratio
                  or (not proximal_growth_guard and expansion > max_expansion_ratio)):
                reason = "excessive_growth"
            elif contact["normal_coherence"] < 0.45:
                reason = "wrapping_contact"
            elif flatness > 12:
                reason = "sheet_like_growth"
        if reason is None:
            # Use the early protrusion, not a remote leaked centroid, to set the
            # initial direction. Ostium itself is determined solely by contact.
            foot = idx[(d > rim_mm) & (d <= rim_mm + 2.0)]
            if not len(foot):
                reason = "no_proximal_protrusion"
            else:
                direction = foot.mean(axis=0) - contact["centre_vox"]
                if np.linalg.norm(direction * vol.mm_per_vox) < 0.25:
                    reason = "ambiguous_direction"
                else:
                    proposals.append(dict(candidate_id=label, voxels=idx,
                        ostium_vox=contact["centre_vox"], dir_vox=direction, n_vox=len(idx),
                        contact_vox=contact["points_vox"], growth_metrics=metrics,
                        rim_mm=rim_mm))
        metrics["accepted"] = reason is None
        metrics["reason"] = reason or "proposed"
        attempts.append(metrics)
    debug = dict(expanded=expanded, difference=difference, labels=labels,
                 growth_connectivity=connectivity,
                 growth_attempts=attempts, blood_model=model,
                 grown_volume_mm3=float(difference.sum() * voxel_mm3))
    return proposals, debug
