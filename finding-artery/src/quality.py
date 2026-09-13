"""Interpretable validation of short proximal paths; no learned scores."""
import numpy as np
from scipy import ndimage as ndi


def local_direction(path_mm, radius_mm):
    """Origin-anchored line fit over at most three initial radii (Riffaud 2022)."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path_mm, axis=0), axis=1))]
    points = path_mm[(arc > 0) & (arc <= max(1.0, 3 * radius_mm))] - path_mm[0]
    if not len(points):
        points = path_mm[1:2] - path_mm[0]
    _, vectors = np.linalg.eigh(points.T @ points)
    direction = vectors[:, -1]
    if direction @ points.sum(axis=0) < 0:
        direction = -direction
    return direction


def validate(vol, lumen, path, model, seed_mm=5.0, shape_start_mm=1.5, curve=None,
             growth_guard=False):
    """Measure blood, radius, cross-section shape and curvature along the path.

    Thresholds below are conservative engineering guards, not thresholds
    validated by either attached paper. Metrics are returned on rejection too.
    """
    from .tracing import _section_stack, _root_component, point_at_arclength

    spacing = np.asarray(vol.spacing_zyx)
    length = curve.length_mm if curve else float(np.linalg.norm(np.diff(path, axis=0) * spacing, axis=1).sum())
    # Coalesce roundoff-equivalent endpoint samples before differentiating.
    samples = np.unique(np.round(np.r_[np.arange(0, length, 0.5), seed_mm, length], 8))
    samples = samples[samples <= length + 1e-8]
    points = curve.points_at(samples) if curve else np.array([
        point_at_arclength(path, spacing, s)[0] for s in samples])
    physical = points * spacing
    hu = ndi.map_coordinates(vol.ct, points.T, order=1, mode="constant", cval=np.nan)
    reliable = samples >= max(0.5, float(spacing.max()) / 2)
    # The opening/common rim is not a closed tubular cross-section. Record its
    # measurements, but apply radius/shape guards only beyond that proximal zone.
    shape_reliable = samples >= max(shape_start_mm, float(spacing.max()))
    if not reliable.any() or not shape_reliable.any():
        return dict(length_mm=length), "insufficient_resolution", path[-1] - path[0]
    inside_blood = (hu >= model["lower_hu"]) & (hu <= model["upper_hu"])
    blood_fraction = float(inside_blood[reliable].mean()) if reliable.any() else 0.0
    parent = ndi.map_coordinates(vol.aorta, points.T, order=0, mode="constant")
    directions = curve.tangents_at(samples) if curve else np.gradient(physical, samples, axis=0)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
    radii, areas, eccentricities, irregularities, clipped = [], [], [], [], []
    cleaned_sections = 0
    for p, d in zip(physical, directions):
        stack, offsets, pixel = _section_stack(lumen, p, d, spacing, [0.0], search_mm=8.0)
        # Do not move to a neighbouring component if the actual path left lumen.
        centre = tuple((np.array(stack[0].shape) - 1) // 2)
        if not stack[0][centre]:
            radii.append(0.0); areas.append(0.0)
            eccentricities.append(0.0); irregularities.append(0.0); clipped.append(False)
            continue
        section = _root_component(stack[0])
        # A sub-voxel neck can connect a valid section to a distant parallel
        # structure in the resliced binary mask. Remove only features narrower
        # than one source voxel; never add lumen or move the path centre.
        opening_radius = float(spacing.min()) / 2
        grid = np.arange(-int(np.ceil(opening_radius / pixel)),
                          int(np.ceil(opening_radius / pixel)) + 1) * pixel
        footprint = grid[:, None] ** 2 + grid[None, :] ** 2 <= opening_radius ** 2
        opened = ndi.binary_opening(section, structure=footprint)
        if opened[centre]:
            cleaned = _root_component(opened)
            cleaned_sections += int(np.any(section != cleaned))
            section = cleaned
        edt = ndi.distance_transform_edt(np.pad(section, 1), sampling=pixel)[1:-1, 1:-1]
        radius = float(edt[centre])
        area = float(section.sum() * pixel ** 2)
        idx = np.argwhere(section) * pixel
        eig = np.linalg.eigvalsh(np.cov(idx.T)) if len(idx) > 2 else np.array([0., 0.])
        eccentricity = float(np.sqrt(eig[-1] / max(eig[0], pixel ** 2 / 12)))
        padded = np.pad(section.astype(np.int8), 1)
        perimeter = sum(np.abs(np.diff(padded, axis=a)).sum() for a in (0, 1)) * pixel
        radii.append(radius); areas.append(area); eccentricities.append(eccentricity)
        irregularities.append(float(perimeter ** 2 / max(4 * np.pi * area, 1e-9)))
        clipped.append(bool((np.linalg.norm(offsets[section], axis=1) >= 8 - pixel).any()))
    radii, areas = np.array(radii), np.array(areas)
    # Use uniform physical intervals, at least one native voxel, so subvoxel
    # sampling and a short endpoint interval do not inflate turning angles.
    coarse_s = (np.linspace(0, length, max(3, int(np.floor(length / max(1., spacing.max()))) + 1))
                if curve else samples[::2])
    coarse = ((curve.points_at(coarse_s) if curve else np.array([
        point_at_arclength(path, spacing, s)[0] for s in coarse_s])) * spacing)
    segments = np.diff(coarse, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    tangents = segments / np.maximum(segment_lengths[:, None], 1e-9)
    angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1, 1))
    curvature = angles / np.maximum((segment_lengths[:-1] + segment_lengths[1:]) / 2, 1e-9)
    analytic_curvature = curve.curvature_at(samples) if curve else None
    if analytic_curvature is not None:
        curvature = analytic_curvature
    reliable_r = radii[shape_reliable]
    adjacent_ratio = np.maximum(reliable_r[2:], reliable_r[:-2]) / np.maximum(
        np.minimum(reliable_r[2:], reliable_r[:-2]), 0.25)
    seed_index = int(np.argmin(np.abs(samples - seed_mm)))
    positive = reliable_r[reliable_r > 0]
    clipped_reliable = np.asarray(clipped) & shape_reliable
    radius_expansion = ((reliable_r[2:] > 2.5 * np.maximum(reliable_r[:-2], 0.25))
                        & (reliable_r[2:] - reliable_r[:-2] > 1.0))
    expansion_samples = samples[shape_reliable][2:][radius_expansion]
    radius_contraction = ((reliable_r[:-2] > 2.5 * np.maximum(reliable_r[2:], 0.25))
                          & (reliable_r[:-2] - reliable_r[2:] > 1.0))
    contraction_samples = samples[shape_reliable][2:][radius_contraction]
    proximal_area = float(np.median(areas[shape_reliable][:3]))
    proximal_diameter = float(2 * np.sqrt(proximal_area / np.pi))
    area_expansion = shape_reliable & (areas > 5.0 * max(proximal_area, 1e-6))
    radius_variation = float(np.std(positive) / np.mean(positive)) if len(positive) else 0.0
    metrics = dict(sample_arclength_mm=samples.tolist(), hu=hu.tolist(), radius_mm=radii.tolist(),
                   cross_section_clipped=clipped,
                   first_overflow_mm=float(samples[clipped_reliable][0]) if clipped_reliable.any() else None,
                   first_radius_expansion_mm=float(expansion_samples[0]) if len(expansion_samples) else None,
                   first_radius_contraction_mm=float(contraction_samples[0]) if len(contraction_samples) else None,
                   first_area_expansion_mm=float(samples[area_expansion][0]) if growth_guard and area_expansion.any() else None,
                   proximal_area_mm2=proximal_area,
                   proximal_equivalent_diameter_mm=proximal_diameter,
                   cleaned_cross_section_count=cleaned_sections,
                   cross_section_opening_radius_mm=float(spacing.min()) / 2,
                   cross_section_area_mm2=areas.tolist(), blood_fraction=blood_fraction,
                   blood_z_p90=float(np.percentile(np.abs(hu[reliable] - model["median_hu"])
                                                  / model["sigma_hu"], 90)),
                   radius_cv=radius_variation,
                   max_radius_ratio_1mm=float(max(adjacent_ratio, default=1.0)),
                   shape_start_mm=float(max(shape_start_mm, float(spacing.max()))),
                   eccentricity_p90=float(np.percentile(np.array(eccentricities)[shape_reliable], 90)),
                   irregularity_p90=float(np.percentile(np.array(irregularities)[shape_reliable], 90)),
                   max_turn_deg=float(np.degrees(max(angles, default=0.0))),
                   max_curvature_per_mm=float(max(curvature, default=0.0)),
                   curvature_method="spline_derivatives" if analytic_curvature is not None else "discrete_turn_per_length",
                   turn_sampling_mm=float(coarse_s[1] - coarse_s[0]),
                   tortuosity=float(length / max(np.linalg.norm(physical[-1] - physical[0]), 1e-9)),
                   radius_at_seed_mm=float(radii[seed_index]))
    reason = None
    if not np.isfinite(hu).all() or blood_fraction < 0.85:
        reason = "blood_inconsistent"
    elif parent[reliable].any():
        reason = "parent_reentry"
    elif (reliable_r <= 0).any() or radii[seed_index] <= 0:
        reason = "unreliable_lumen"
    elif proximal_diameter < 2.0:
        reason = "below_minimum_diameter"
    elif np.asarray(clipped)[shape_reliable].any():
        reason = "cross_section_overflow"
    elif growth_guard and area_expansion.any():
        reason = "expanding_lumen"
    elif metrics["max_radius_ratio_1mm"] > 2.5 and np.ptp(reliable_r) > 1.0:
        reason = "abrupt_radius_change"
    elif metrics["eccentricity_p90"] > 4.0 or metrics["irregularity_p90"] > 4.0:
        reason = "irregular_cross_section"
    elif metrics["max_turn_deg"] > 75 or metrics["tortuosity"] > 1.8:
        reason = "irregular_centerline"
    initial_radius = float(np.median(positive[:3])) if len(positive) else 1.0
    direction = local_direction(physical, initial_radius) / spacing
    return metrics, reason, direction
