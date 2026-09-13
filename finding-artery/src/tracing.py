"""Stage 2: follow each candidate outward, then measure it.

The spec wants, per daughter: an ostium, a seed 5 mm along the vessel, a local
radius at that seed, and a unit direction. All four come out of one short
traced path.
"""
import numpy as np
from scipy import ndimage as ndi


def _sample(arr, p):
    """Nearest-neighbour sample at float voxel coord p=(z,y,x)."""
    i = np.rint(p).astype(int)
    if np.any(i < 0) or np.any(i >= np.array(arr.shape)):
        return None
    return arr[i[0], i[1], i[2]]


def _section_stack(lumen, p_mm, direction, spacing, distances, search_mm):
    """Sample parallel physical cross-sections normal to the current tangent."""
    reference = np.eye(3)[np.argmin(np.abs(direction))]
    u = np.cross(direction, reference)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    pixel_mm = min(0.5, float(np.min(spacing)))
    count = int(np.ceil(search_mm / pixel_mm))
    a, b = np.meshgrid(np.arange(-count, count + 1) * pixel_mm,
                       np.arange(-count, count + 1) * pixel_mm, indexing="ij")
    offsets = a[..., None] * u + b[..., None] * v
    positions = (p_mm + np.asarray(distances)[:, None, None, None] * direction
                 + offsets[None])
    samples = ndi.map_coordinates(lumen, np.moveaxis(positions / spacing, -1, 0),
                                  order=0, mode="constant", cval=0).astype(bool)
    samples &= (a * a + b * b <= search_mm ** 2)[None]
    return samples, offsets, pixel_mm


def _root_component(section):
    """Choose the component containing the predicted centre, or the nearest."""
    lab, count = ndi.label(section, structure=np.ones((3, 3)))
    if not count:
        return np.zeros_like(section)
    centre = (np.array(section.shape) - 1) // 2
    value = lab[tuple(centre)]
    if value == 0:
        idx = np.argwhere(section)
        closest = idx[np.argmin(np.sum((idx - centre) ** 2, axis=1))]
        value = lab[tuple(closest)]
    return lab == value


def _connected_sections(stack, root):
    """Keep only voxels reachable forward from the current lumen component.

    Connectivity advances one plane at a time: it cannot go backwards through
    an unrelated downstream connection to import a neighbouring vessel.
    """
    connected = np.zeros_like(stack)
    connected[0] = root
    for i in range(1, len(stack)):
        lab, _ = ndi.label(stack[i], structure=np.ones((3, 3)))
        touching = ndi.binary_dilation(connected[i - 1], structure=np.ones((3, 3)))
        ids = np.unique(lab[touching & stack[i]])
        connected[i] = np.isin(lab, ids) & stack[i]
    return connected


def _persistent_split(connected, pixel_mm, min_area_mm2):
    """Two separate next-plane lobes must each survive through the lookahead.

    This detects a *resolved* lumen split; the precise anatomical junction can
    precede it by a vessel radius. A tiny spur or a transient hole is insufficient.
    """
    lab, count = ndi.label(connected[1], structure=np.ones((3, 3)))
    minimum = max(3, int(np.ceil(min_area_mm2 / pixel_mm ** 2)))
    ids = [i for i in range(1, count + 1) if np.count_nonzero(lab == i) >= minimum]
    if len(ids) < 2:
        return False
    # Labelling only the forward volume prevents the shared upstream trunk
    # from merging the two arms. Arms that immediately rejoin are not a split.
    forward, _ = ndi.label(connected[1:], structure=np.ones((3, 3, 3)))
    survivors = set()
    for i in ids:
        labels = np.unique(forward[0][lab == i])
        for value in labels[labels > 0]:
            if all(np.count_nonzero(plane == value) >= minimum for plane in forward):
                survivors.add(int(value))
    return len(survivors) >= 2


def trace_branch(vol, bright, start_vox, dir_vox,
                 step_mm=0.5, max_mm=10.0, search_mm=2.5,
                 bifurcation_confirm_mm=1.0, bifurcation_min_area_mm2=0.75,
                 return_reason=False, strict_initial=False):
    """Follow connected lumen, stopping before a persistent cross-section split.

    Return voxel path and its measured physical arc length. With return_reason,
    also return a diagnostic stop reason. The public two-value form is retained.
    """
    if step_mm <= 0 or max_mm <= 0 or search_mm <= 0 or bifurcation_confirm_mm <= 0:
        raise ValueError("trace distances must be positive")
    sp = np.asarray(vol.spacing_zyx)
    d = np.asarray(dir_vox, float) * sp
    norm = np.linalg.norm(d)
    if norm == 0:
        result = (None, 0.0, "invalid_direction")
        return result if return_reason else result[:2]
    d /= norm
    p = np.asarray(start_vox, float) * sp
    path = [p / sp]
    travelled, reason = 0.0, "max_length"
    lumen = bright & ~vol.aorta
    # Bound the loop even if re-centring produces very short physical steps.
    for _ in range(int(np.ceil(max_mm / step_mm)) * 4):
        if travelled >= max_mm - 1e-9:
            break
        step = min(step_mm, max_mm - travelled)
        distances = np.r_[0.0, step + np.arange(
            int(np.ceil(bifurcation_confirm_mm / step_mm)) + 1) * step_mm]
        stack, offsets, pixel = _section_stack(lumen, p, d, sp, distances, search_mm)
        root = _root_component(stack[0])
        # The initial ostium can lie on the mask boundary; allow the first
        # outside-aorta cross-section to establish connectivity.
        if not root.any() and len(path) == 1:
            connected = np.zeros_like(stack)
            connected[1:] = _connected_sections(stack[1:], _root_component(stack[1]))
        else:
            connected = _connected_sections(stack, root)
        if not connected[1].any():
            reason = "lumen_lost"
            break
        if _persistent_split(connected, pixel, bifurcation_min_area_mm2):
            reason = "bifurcation"
            break
        component = _root_component(connected[1])
        correction = offsets[component].mean(axis=0)
        # All offsets lie in the plane, so re-centring has no forward component.
        p_new = p + d * step + correction
        if not _sample(lumen, p_new / sp):
            # A crescent's centroid can fall outside it. Choose its nearest
            # sampled lumen point rather than accepting a background seed.
            choices = offsets[component]
            correction = choices[np.argmin(np.linalg.norm(choices - correction, axis=1))]
            p_new = p + d * step + correction
        new_d = p_new - p
        length = np.linalg.norm(new_d)
        if length > max_mm - travelled:
            p_new = p + new_d * ((max_mm - travelled) / length)
            new_d = p_new - p
            length = np.linalg.norm(new_d)
        # Do not bridge a gap between the previous point and the new centre.
        checks = p + np.linspace(0, 1, max(2, int(np.ceil(length / (sp.min() / 2))))
                                  + 1)[1:, None] * new_d
        if len(path) > 1 and not ndi.map_coordinates(
                lumen, (checks / sp).T, order=0, mode="constant").all():
            reason = "lumen_lost"
            break
        if len(path) == 1 and strict_initial:
            connected_start = (ndi.map_coordinates(lumen, (checks / sp).T, order=0, mode="constant")
                               | ndi.map_coordinates(vol.aorta, (checks / sp).T, order=0, mode="constant"))
            if not connected_start.all():
                reason = "lumen_lost"
                break
        if _sample(vol.aorta, p_new / sp):
            reason = "parent_reentry"
            break
        d = 0.65 * d + 0.35 * new_d / max(length, 1e-9)
        d /= np.linalg.norm(d)
        travelled += length
        p = p_new
        path.append(p / sp)
    else:
        reason = "iteration_limit"
    result = (np.asarray(path), float(travelled), reason)
    return result if return_reason else result[:2]


def point_at_arclength(path, spacing, target_mm):
    """Interpolate the point `target_mm` along a traced path."""
    sp = np.array(spacing)
    if len(path) < 2 or target_mm <= 0:
        return path[0], 0.0
    seg = np.linalg.norm(np.diff(path, axis=0) * sp, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= target_mm:
        return path[-1], cum[-1]
    i = int(np.searchsorted(cum, target_mm))
    t = (target_mm - cum[i - 1]) / max(seg[i - 1], 1e-9)
    return path[i - 1] + t * (path[i] - path[i - 1]), target_mm


def radius_field(vol, bright):
    """EDT inside the bright non-aortic lumen -> local radius everywhere."""
    return ndi.distance_transform_edt(bright & ~vol.aorta,
                                      sampling=vol.spacing_zyx).astype(np.float32)


def measure(vol, bright, radfield, cand, seed_mm=5.0, max_mm=10.0,
            min_length_mm=5.0, bifurcation_confirm_mm=1.0,
            bifurcation_min_area_mm2=0.75, diagnostics=None, quality_model=None,
            curve_geometry=True, truncate_leakage=False):
    """Trace one candidate and turn it into a daughter record, or None."""
    path, length, reason = trace_branch(
        vol, bright, cand["ostium_vox"], cand["dir_vox"], max_mm=max_mm,
        bifurcation_confirm_mm=bifurcation_confirm_mm,
        bifurcation_min_area_mm2=bifurcation_min_area_mm2, return_reason=True,
        strict_initial=quality_model is not None)
    curve = None
    raw_length = length
    if quality_model is not None and curve_geometry and path is not None and len(path) >= 2:
        from .centerline import Centerline
        curve = Centerline(path, vol.spacing_zyx, lumen=bright & ~vol.aorta,
                           parent=vol.aorta, max_mm=max_mm)
        length = curve.length_mm
        path = curve.sampled_path()
    eligible = path is not None and min(length, raw_length) + 1e-9 >= max(min_length_mm, seed_mm)
    if diagnostics is not None:
        diagnostics.update(ostium_vox=cand["ostium_vox"], length_mm=length,
                           stop_reason=reason, eligible=eligible,
                           rejection_reason=None if eligible else "insufficient_length")
        if quality_model is not None:
            diagnostics.update(candidate_id=cand.get("candidate_id"), path_vox=path)
            diagnostics["curve_geometry"] = curve.diagnostics() if curve else None
    if not eligible:
        return None                              # fails the 5 mm eligibility rule

    seed = curve.points_at(seed_mm) if curve else point_at_arclength(path, vol.spacing_zyx, seed_mm)[0]

    if quality_model is not None:
        from .quality import validate
        metrics, rejection, dvec = validate(vol, bright, path, quality_model, seed_mm,
                                            shape_start_mm=cand.get("rim_mm", 1.5), curve=curve,
                                            growth_guard=truncate_leakage)
        # A distal leak or narrowing need not erase a valid proximal tube.
        # Stop at the last section before the change; the complete prefix
        # must still satisfy every quality check and the same 5 mm requirement.
        leakage_events = [metrics[k] for k in ("first_overflow_mm", "first_radius_expansion_mm", "first_radius_contraction_mm", "first_area_expansion_mm")
                          if metrics.get(k) is not None]
        if truncate_leakage and rejection in ("cross_section_overflow", "abrupt_radius_change", "expanding_lumen") and leakage_events:
            first_bad = min(leakage_events)
            earlier = np.asarray(metrics["sample_arclength_mm"])
            earlier = earlier[earlier < first_bad - 1e-8]
            cutoff = float(earlier[-1]) if len(earlier) else 0.
            if cutoff + 1e-9 >= max(min_length_mm, seed_mm):
                if curve:
                    curve.length_mm = cutoff
                    path = curve.sampled_path()
                else:
                    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0) * vol.mm_per_vox, axis=1))]
                    endpoint = point_at_arclength(path, vol.spacing_zyx, cutoff)[0]
                    path = np.vstack([path[arcs < cutoff], endpoint])
                narrowing = metrics.get("first_radius_contraction_mm") == first_bad
                length, reason = cutoff, "distal_narrowing" if narrowing else "distal_leakage"
                metrics, rejection, dvec = validate(vol, bright, path, quality_model, seed_mm,
                    shape_start_mm=cand.get("rim_mm", 1.5), curve=curve, growth_guard=True)
                metrics["truncated_before_narrowing_mm" if narrowing else "truncated_before_leakage_mm"] = first_bad
        if diagnostics is not None:
            diagnostics.update(quality=metrics, rejection_reason=rejection,
                               eligible=rejection is None, length_mm=length,
                               stop_reason=reason, path_vox=path,
                               curve_geometry=curve.diagnostics() if curve else None)
        if rejection:
            return None
        return dict(ostium_vox=path[0], seed_vox=seed,
                    radius_mm=metrics["radius_at_seed_mm"], dir_vox=dvec, path=path,
                    length_mm=float(length), stop_reason=reason, quality=metrics,
                    curve_geometry=curve.diagnostics() if curve else None,
                    candidate_id=cand.get("candidate_id"))

    r = _sample(radfield, seed)
    if r is None or r <= 0:
        r = float(np.mean(vol.mm_per_vox))       # fall back to ~1 voxel

    # direction = ostium -> seed, which follows the proximal path rather than
    # the straight line to the far tip
    dvec = seed - path[0]

    return {
        "ostium_vox": path[0],
        "seed_vox": seed,
        "radius_mm": float(r),
        "dir_vox": dvec,
        "path": path,
        "length_mm": float(length),
        "stop_reason": reason,
    }
