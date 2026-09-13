"""Combine parent separation with opening and proximal-lumen evidence.

These are physical engineering heuristics, not proof of anatomical ownership.
Near-wall vessels need stronger opening and radius consistency instead of
being unconditionally discarded. Record the decision for each candidate.
"""
import numpy as np
from scipy import ndimage as ndi


def validate_departure(wall_distance, path, growth_metrics, min_departure_mm=4.0,
                       probe_mm=5.0, quality_metrics=None, spacing_zyx=None):
    """Require separation or stronger opening/lumen support for a near-wall path.

    ``wall_distance`` is the physical EDT outside the original supplied mask;
    ``path`` uses cropped z,y,x voxel coordinates. Path length and distance to
    the parent are intentionally separate measurements. No case ID, reference
    landmarks, or expected artery counts are used.
    """
    if not np.isfinite([min_departure_mm, probe_mm]).all() or min_departure_mm < 0 or probe_mm <= 0:
        raise ValueError("departure must be nonnegative and probe must be positive")
    points = np.asarray(path, dtype=float)
    distances = ndi.map_coordinates(wall_distance, points.T, order=1,
                                    mode="constant", cval=np.nan)
    finite = bool(len(distances) and np.isfinite(distances).all())
    maximum = float(distances.max()) if finite and len(distances) else 0.0
    quality = quality_metrics or {}
    coherence = float(growth_metrics.get("contact_normal_coherence", 0.0))
    radius_ratio = float(quality.get("max_radius_ratio_1mm", np.inf))
    diameter = float(quality.get("proximal_equivalent_diameter_mm", 0.0))
    area = float(quality.get("proximal_area_mm2", 0.0))
    contact_area = float(growth_metrics.get("contact_area_mm2", 0.0))
    required_diameter = (2.0 * float(np.max(spacing_zyx))
                         if spacing_zyx is not None else None)
    resolved_lumen = (required_diameter is not None and np.isfinite(diameter)
                      and diameter >= required_diameter)
    contact_ratio = contact_area / area if area > 0 else np.inf
    compact_contact = np.isfinite(contact_ratio) and 0 < contact_ratio <= 5.0
    # For an oblique vessel, little radial separation can coexist with a
    # coherent opening and a stable tubular path. Require both forms of
    # evidence, resolution of at least two voxels across the proximal lumen,
    # and a contact area that is not disproportionate to that lumen. A broad
    # wall attachment is not rescued just because its normals are coherent.
    coherent_opening = np.isfinite(coherence) and np.sqrt(0.5) <= coherence <= 1.0 + 1e-9
    stable_radius = np.isfinite(radius_ratio) and 1.0 <= radius_ratio <= 2.0
    distal_warning = bool(growth_metrics.get("distal_expansion_warning", False))
    near_wall_supported = (coherent_opening and stable_radius and resolved_lumen
                           and compact_contact and not distal_warning)
    metrics = dict(max_wall_distance_mm=maximum,
                   minimum_wall_departure_mm=float(min_departure_mm),
                   wall_distance_along_path_mm=distances.tolist(),
                   distal_expansion_warning=distal_warning,
                   coherent_opening=bool(coherent_opening),
                   stable_proximal_radius=bool(stable_radius),
                   resolved_proximal_lumen=bool(resolved_lumen),
                   required_proximal_diameter_mm=required_diameter,
                   compact_contact=bool(compact_contact),
                   contact_to_lumen_area_ratio=contact_ratio if np.isfinite(contact_ratio) else None,
                   contact_normal_coherence=coherence,
                   max_radius_ratio_1mm=radius_ratio if np.isfinite(radius_ratio) else None,
                   near_wall_supported=bool(near_wall_supported),
                   decision="outward_departure")
    if not finite:
        metrics["decision"] = "invalid_wall_distance"
        return metrics, "invalid_wall_distance"
    if maximum + 1e-6 < min_departure_mm:
        if not near_wall_supported:
            metrics["decision"] = "unsupported_near_wall_path"
            return metrics, "insufficient_wall_departure"
        metrics["decision"] = "coherent_opening_and_stable_lumen"
    if metrics["distal_expansion_warning"] and maximum + 1e-6 < probe_mm:
        metrics["decision"] = "unsupported_distal_growth"
        return metrics, "distal_growth_without_departure"
    return metrics, None
