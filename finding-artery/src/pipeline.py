"""Glue: volume -> candidates -> traced daughters -> JSON dict."""
import json
import numpy as np

from . import volume as V
from . import detection, tracing


DEFAULTS = dict(
    method="growth",
    margin_mm=25.0,
    thr_pct=5.0,
    thr_margin_hu=15.0,
    collar_mm=4.0,
    min_area_mm2=2.0,
    seed_mm=5.0,
    max_trace_mm=10.0,
    min_length_mm=5.0,
    bifurcation_confirm_mm=1.0,
    bifurcation_min_area_mm2=0.75,
    dedup_mm=4.0,
    growth_mm=12.0,
    probe_mm=5.0,
    rim_mm=1.5,
    min_wall_departure_mm=4.0,
    growth_connectivity=26,
    growth_intensity="adaptive",
    curve_geometry=True,
)


def dedup(daughters, spacing, tol_mm):
    """Merge nearby openings only when their proximal paths also overlap."""
    from scipy.spatial.distance import cdist
    from .tracing import point_at_arclength

    def proximal_path(d):
        path = np.asarray(d["path"])
        length = np.linalg.norm(np.diff(path, axis=0) * spacing, axis=1).sum()
        return np.array([point_at_arclength(path, spacing, s)[0] * spacing
                         for s in np.linspace(1.5, min(5.0, length), 15)])

    kept = []
    paths = []
    for d in sorted(daughters, key=lambda x: -x["length_mm"]):
        o = np.array(d["ostium_vox"]) * np.array(spacing)
        path = proximal_path(d)
        duplicate = False
        for k, other in zip(kept, paths):
            if np.linalg.norm(o - np.array(k["ostium_vox"]) * spacing) > tol_mm:
                continue
            distances = cdist(path, other)
            overlap = min(np.mean(distances.min(axis=0) <= 1.5),
                          np.mean(distances.min(axis=1) <= 1.5))
            if overlap >= 0.8:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
            paths.append(path)
    return kept


def run_case(image_path, mask_path, case_id="case", **kw):
    p = {**DEFAULTS, **kw}
    if not 0 <= p["min_wall_departure_mm"] <= p["growth_mm"]:
        raise ValueError("minimum wall departure must be between zero and growth_mm")
    vol = V.load(image_path, mask_path, margin_mm=p["margin_mm"])

    growth_debug, model = {}, None
    jobs = []
    if p["method"] == "growth":
        from . import growth
        model = V.blood_model(vol)
        thr, ceil = model["lower_hu"], model["upper_hu"]
        if p["growth_intensity"] not in ("strict", "adaptive"):
            raise ValueError("growth_intensity must be strict or adaptive")
        models = [("core", model)]
        if p["growth_intensity"] == "adaptive":
            boundary = V.boundary_blood_model(vol, model)
            # Near-identical ranges mostly duplicate noisy boundary candidates.
            # Require a material difference relative to this patient's noise.
            if boundary["lower_hu"] < model["lower_hu"] - model["sigma_hu"]:
                models.append(("boundary", boundary))
        cands, all_attempts, labels_by_pass, models_by_pass = [], [], {}, {}
        bright = np.zeros_like(vol.aorta)
        expanded = vol.aorta.copy()
        id_offset = 0
        rim_retries = {}
        for source, current_model in models:
            retry = rim_retries.get(source)
            rim = retry[0] if retry else p["rim_mm"]
            proposed, current_debug = growth.candidates(
                vol, current_model, growth_mm=p["growth_mm"], probe_mm=p["probe_mm"], rim_mm=rim,
                connectivity=p["growth_connectivity"], proximal_growth_guard=source != "core")
            if retry:
                # Only revisit regions rejected as excessive growth. A wider
                # marker rim can separate a one-voxel wall bridge on coarse CT;
                # this partitions existing lumen without lowering its HU range.
                old_labels, rejected_ids = retry[1:]
                proposed = [c for c in proposed if np.mean(np.isin(
                    old_labels[tuple(c["voxels"].T)], rejected_ids)) >= 0.8]
                retained = {c["candidate_id"] for c in proposed}
                for attempt in current_debug["growth_attempts"]:
                    if attempt["accepted"] and attempt["candidate_id"] not in retained:
                        attempt.update(accepted=False, reason="outside_rejected_region")
            labels = current_debug["labels"]
            local_max = int(labels.max())
            labels = np.where(labels > 0, labels + id_offset, 0)
            for attempt in current_debug["growth_attempts"]:
                attempt["candidate_id"] += id_offset
                attempt["intensity_pass"] = source
            for c in proposed:
                c["candidate_id"] += id_offset
                c["intensity_pass"] = source
                jobs.append((c, labels, current_model))
            cands.extend(proposed)
            all_attempts.extend(current_debug["growth_attempts"])
            bright |= current_debug["difference"]
            expanded |= current_debug["expanded"]
            labels_by_pass[source], models_by_pass[source] = labels, current_model
            wider_rim = max(p["rim_mm"], 1.5 * min(vol.spacing_zyx))
            if (not retry and source != "core" and p["rim_mm"] < wider_rim < p["probe_mm"]):
                rejected = [a["candidate_id"] for a in current_debug["growth_attempts"]
                            if a["reason"] == "excessive_growth"]
                if rejected:
                    retry_name = source + "_rim_retry"
                    rim_retries[retry_name] = (wider_rim, labels, rejected)
                    models.append((retry_name, current_model))
            id_offset += local_max
        growth_debug = dict(difference=bright, expanded=expanded,
            labels_by_pass=labels_by_pass, blood_models_by_pass=models_by_pass,
            growth_attempts=all_attempts, blood_model=model,
            growth_connectivity=p["growth_connectivity"], growth_intensity=p["growth_intensity"],
            grown_volume_mm3=float(bright.sum() * np.prod(vol.spacing_zyx)))
    elif p["method"] == "collar":
        thr = V.lumen_threshold(vol, p["thr_pct"], p["thr_margin_hu"])
        ceil = V.bone_ceiling(vol)
        bright = (vol.ct > thr) & (vol.ct < ceil)
        cands, _, _ = detection.candidates(
            vol, thr, ceil, collar_mm=p["collar_mm"], min_area_mm2=p["min_area_mm2"])
        jobs = [(c, None, None) for c in cands]
    else:
        raise ValueError("method must be 'collar' or 'growth'")

    radf = tracing.radius_field(vol, bright) if model is None else None
    if model is not None:
        from scipy import ndimage as ndi
        from .origin_validation import validate_departure
        wall_distance = ndi.distance_transform_edt(~vol.aorta, sampling=vol.spacing_zyx)

    found = []
    attempts = []
    for c, labels, current_model in jobs:
        diagnostic = dict(intensity_pass=c.get("intensity_pass"))
        # Growth candidates can only trace through their own connected region.
        lumen = bright if current_model is None else labels == c["candidate_id"]
        m = tracing.measure(vol, lumen, radf, c,
                          seed_mm=p["seed_mm"],
                          max_mm=p["max_trace_mm"],
                          min_length_mm=p["min_length_mm"],
                          bifurcation_confirm_mm=p["bifurcation_confirm_mm"],
                          bifurcation_min_area_mm2=p["bifurcation_min_area_mm2"],
                          diagnostics=diagnostic, quality_model=current_model,
                          curve_geometry=p["curve_geometry"],
                          truncate_leakage=current_model is not None and p["growth_intensity"] == "adaptive")
        attempts.append(diagnostic)
        if m is not None:
            if current_model is not None:
                origin_metrics, rejection = validate_departure(
                    wall_distance, m["path"], c["growth_metrics"],
                    min_departure_mm=p["min_wall_departure_mm"], probe_mm=p["probe_mm"],
                    quality_metrics=m.get("quality"), spacing_zyx=vol.spacing_zyx)
                diagnostic["origin_validation"] = origin_metrics
                if rejection:
                    diagnostic.update(eligible=False, rejection_reason=rejection)
                    continue
            m["intensity_pass"] = c.get("intensity_pass")
            found.append(m)

    found = dedup(found, vol.spacing_zyx, p["dedup_mm"])

    daughters = []
    for i, d in enumerate(found, start=1):
        daughters.append({
            "instance_id": f"branch_{i:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(v, 3) for v in vol.to_mm(d["ostium_vox"])],
            "seed_xyz_mm": [round(v, 3) for v in vol.to_mm(d["seed_vox"])],
            "radius_mm": round(d["radius_mm"], 3),
            "direction_xyz": [round(v, 4) for v in vol.direction_to_mm(d["dir_vox"])],
        })

    result = {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }
    debug = dict(vol=vol, bright=bright, thr=thr, ceil=ceil,
                 cands=cands, traced=found, trace_attempts=attempts, method=p["method"],
                 **growth_debug)
    return result, debug


def write_json(result, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=lambda obj: obj.tolist(), allow_nan=False)


def write_diagnostics(debug, result, path):
    """Compact auditable metrics; omit the large voxel arrays."""
    def finite_json(value):
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value

    payload = dict(case_id=result["case_id"], method=debug["method"],
                    coordinate_system="SimpleITK physical LPS millimetres",
                    growth_connectivity=debug.get("growth_connectivity"),
                    growth_intensity=debug.get("growth_intensity"),
                    blood_model=debug.get("blood_model"),
                    blood_models_by_pass=debug.get("blood_models_by_pass"),
                    grown_volume_mm3=debug.get("grown_volume_mm3"),
                    growth_attempts=debug.get("growth_attempts", []),
                    trace_attempts=debug["trace_attempts"],
                    kept_candidate_ids=[d.get("candidate_id") for d in debug["traced"]],
                    kept_paths_lps_mm=[dict(instance_id=meta["instance_id"],
                        candidate_id=d.get("candidate_id"), intensity_pass=d.get("intensity_pass"),
                        curve_geometry=d.get("curve_geometry"),
                        points=[debug["vol"].to_mm(point) for point in d["path"]])
                        for meta, d in zip(result["daughters"], debug["traced"])],
                    daughter_count=len(result["daughters"]))
    write_json(finite_json(payload), path)
