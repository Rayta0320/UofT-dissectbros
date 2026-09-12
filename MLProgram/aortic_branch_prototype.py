#!/usr/bin/env python3
"""Prototype detector for vessel branches that originate from a supplied aorta mask.

The default path is deliberately training-free: multiscale 3-D Frangi proposes
candidate branches, a short ridge tracker verifies them, and a geometric score
acts as the branch probability.  If a trained Tiny3DCNN checkpoint is supplied,
its probability replaces the heuristic score.

Coordinates written to JSON default to LPS millimetres for SeqSeg/SimpleITK;
RAS+ NIfTI world coordinates are available with ``--coordinate-system RAS``.
This is research software and is not intended for clinical use.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import frangi


EPS = 1.0e-8


@dataclass
class Config:
    search_mm: float = 10.0
    trace_mm: float = 12.0
    spacing_mm: float = 0.8
    sigmas_mm: tuple[float, ...] = (0.8, 1.2, 1.8, 2.6, 3.6)
    anchor_min_mm: float = 2.0
    anchor_max_mm: float = 8.0
    maximum_candidates: int = 96
    maximum_branches: int = 16
    probability_threshold: float = 0.42
    deduplicate_mm: float = 5.0
    patch_size: int = 32


@dataclass
class Branch:
    probability: float
    ostium: list[float]
    seed: list[float]
    radius_mm: float
    direction: list[float]
    centerline: list[list[float]]
    trace_length_mm: float
    proposal_response: float
    scoring_method: str


@contextlib.contextmanager
def open_nifti(path: Path) -> Iterator[nib.spatialimages.SpatialImage]:
    """Open ordinary NIfTI and gzip data accidentally named ``*.nii``."""
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        is_gzip = stream.read(2) == b"\x1f\x8b"

    temporary_name: str | None = None
    try:
        load_path = path
        if is_gzip and not path.name.endswith(".gz"):
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as target:
                temporary_name = target.name
                with path.open("rb") as source:
                    shutil.copyfileobj(source, target)
            load_path = Path(temporary_name)
        yield nib.load(str(load_path))
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length > EPS:
        return vector / length
    if fallback is not None:
        return _unit(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _resample_crop(
    array: np.ndarray,
    voxel_spacing: np.ndarray,
    output_spacing: float,
    order: int,
) -> np.ndarray:
    scale = output_spacing / voxel_spacing
    output_shape = tuple(
        max(1, int(math.floor((size - 1) / factor)) + 1)
        for size, factor in zip(array.shape, scale, strict=True)
    )
    return ndi.affine_transform(
        array,
        matrix=np.diag(scale),
        offset=0.0,
        output_shape=output_shape,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def _isotropic_affine(
    original_affine: np.ndarray,
    crop_start: np.ndarray,
    voxel_spacing: np.ndarray,
    output_spacing: float,
) -> np.ndarray:
    mapping = np.eye(4, dtype=np.float64)
    mapping[:3, :3] = np.diag(output_spacing / voxel_spacing)
    mapping[:3, 3] = crop_start
    return original_affine @ mapping


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.atleast_2d(points).astype(np.float64)
    return points @ affine[:3, :3].T + affine[:3, 3]


def load_local_roi(
    ct_path: Path,
    mask_path: Path,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with open_nifti(mask_path) as mask_image:
        mask = np.asanyarray(mask_image.dataobj) > 0
        mask_shape = mask_image.shape[:3]
        mask_affine = np.asarray(mask_image.affine, dtype=np.float64)
        spacing = np.asarray(mask_image.header.get_zooms()[:3], dtype=np.float64)

    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("The aorta mask must be a non-empty 3-D NIfTI volume.")

    occupied = np.argwhere(mask)
    padding_voxels = np.ceil((config.search_mm + 5.0) / spacing).astype(int)
    crop_start = np.maximum(occupied.min(axis=0) - padding_voxels, 0)
    crop_stop = np.minimum(occupied.max(axis=0) + padding_voxels + 1, mask.shape)
    crop_slices = tuple(slice(int(a), int(b)) for a, b in zip(crop_start, crop_stop, strict=True))
    mask_crop = mask[crop_slices]
    del mask, occupied

    with open_nifti(ct_path) as ct_image:
        if ct_image.shape[:3] != mask_shape:
            raise ValueError(f"CT shape {ct_image.shape} does not match mask shape {mask_shape}.")
        if not np.allclose(ct_image.affine, mask_affine, atol=1.0e-3):
            raise ValueError("CT and aorta mask affines do not match.")
        ct_crop = np.asarray(ct_image.dataobj[crop_slices], dtype=np.float32)

    ct_iso = _resample_crop(ct_crop, spacing, config.spacing_mm, order=1).astype(np.float32)
    mask_iso = _resample_crop(mask_crop.astype(np.uint8), spacing, config.spacing_mm, order=0) > 0
    affine_iso = _isotropic_affine(mask_affine, crop_start, spacing, config.spacing_mm)

    aorta_values = ct_iso[mask_iso]
    p05, p10, median, p90, p95 = np.percentile(aorta_values, [5, 10, 50, 90, 95])
    statistics = {
        "aorta_p05_hu": float(p05),
        "aorta_p10_hu": float(p10),
        "aorta_median_hu": float(median),
        "aorta_p90_hu": float(p90),
        "aorta_p95_hu": float(p95),
    }
    return ct_iso, mask_iso, affine_iso, statistics


def compute_vesselness(
    ct: np.ndarray,
    mask: np.ndarray,
    config: Config,
    statistics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower = min(-100.0, statistics["aorta_p10_hu"] - 300.0)
    upper = max(lower + 300.0, statistics["aorta_p90_hu"] + 150.0)
    normalized = np.clip((ct - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)

    sigmas_voxels = tuple(max(0.5, sigma / config.spacing_mm) for sigma in config.sigmas_mm)
    vesselness = frangi(
        normalized,
        sigmas=sigmas_voxels,
        alpha=0.5,
        beta=0.5,
        gamma=None,
        black_ridges=False,
        mode="nearest",
    ).astype(np.float32)

    exterior_distance, nearest = ndi.distance_transform_edt(
        ~mask,
        sampling=config.spacing_mm,
        return_indices=True,
    )
    exterior_distance = exterior_distance.astype(np.float32)
    signed_distance = exterior_distance.copy()
    signed_distance[mask] = -ndi.distance_transform_edt(mask, sampling=config.spacing_mm)[mask]

    transition = max(25.0, 0.15 * (statistics["aorta_p90_hu"] - lower))
    plausible_floor = statistics["aorta_p10_hu"] - 180.0
    intensity_logit = np.clip((ct - plausible_floor) / transition, -60.0, 60.0)
    intensity_likelihood = 1.0 / (1.0 + np.exp(-intensity_logit))
    evidence = vesselness * (0.25 + 0.75 * intensity_likelihood.astype(np.float32))
    evidence[(exterior_distance <= 0.0) | (exterior_distance > config.search_mm)] = 0.0
    return vesselness, evidence, signed_distance, nearest


def propose_candidates(
    evidence: np.ndarray,
    exterior_distance: np.ndarray,
    config: Config,
) -> list[np.ndarray]:
    band = (
        (exterior_distance >= config.anchor_min_mm)
        & (exterior_distance <= min(config.anchor_max_mm, config.search_mm))
        & (evidence > 0.0)
    )
    values = evidence[band]
    if values.size == 0 or float(values.max()) <= 0.0:
        return []

    positive = values[values > 0.0]
    threshold = max(float(np.percentile(positive, 88.0)), 0.035 * float(positive.max()))
    separation_voxels = max(3, int(round(3.0 / config.spacing_mm)))
    if separation_voxels % 2 == 0:
        separation_voxels += 1
    local_maximum = evidence >= ndi.maximum_filter(evidence, size=separation_voxels, mode="nearest")
    valid_interior = np.ones(evidence.shape, dtype=bool)
    edge_margin = max(2, int(math.ceil(5.0 / config.spacing_mm)))
    for axis, length in enumerate(evidence.shape):
        if length <= 2 * edge_margin:
            continue
        leading = [slice(None)] * 3
        trailing = [slice(None)] * 3
        leading[axis] = slice(0, edge_margin)
        trailing[axis] = slice(length - edge_margin, length)
        valid_interior[tuple(leading)] = False
        valid_interior[tuple(trailing)] = False
    coordinates = np.argwhere(band & valid_interior & local_maximum & (evidence >= threshold))
    if coordinates.size == 0:
        return []
    responses = evidence[tuple(coordinates.T)]
    order = np.argsort(responses)[::-1][: config.maximum_candidates]
    return [coordinate.astype(np.float64) for coordinate in coordinates[order]]


def _sample_cube(
    center: np.ndarray,
    shape: Sequence[int],
    radius_voxels: int,
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    rounded = np.rint(center).astype(int)
    low = np.maximum(rounded - radius_voxels, 0)
    high = np.minimum(rounded + radius_voxels + 1, np.asarray(shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high, strict=True))
    grid = np.indices(tuple(high - low), dtype=np.float64).reshape(3, -1).T + low
    return grid, slices


def trace_candidate(
    anchor: np.ndarray,
    surface: np.ndarray,
    evidence: np.ndarray,
    signed_distance: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    direction = _unit(anchor - surface)
    current = anchor.copy()
    points = [surface.copy(), anchor.copy()]
    responses = [float(evidence[tuple(np.rint(anchor).astype(int))])]
    step_voxels = max(1.0, 1.0 / config.spacing_mm)
    radius_voxels = max(2, int(math.ceil(2.6 / config.spacing_mm)))
    number_steps = max(1, int(math.ceil(config.trace_mm)))

    for _ in range(number_steps):
        predicted = current + direction * step_voxels
        grid, slices = _sample_cube(predicted, evidence.shape, radius_voxels)
        local_evidence = evidence[slices].reshape(-1)
        local_distance = signed_distance[slices].reshape(-1)
        offsets = grid - predicted
        distances = np.linalg.norm(offsets, axis=1)
        forward = (grid - current) @ direction
        valid = (distances <= radius_voxels) & (forward > -0.25) & (local_distance > 0.0)
        if not np.any(valid):
            break
        weights = np.square(local_evidence) * np.exp(-0.5 * np.square(distances / max(radius_voxels, 1)))
        weights[~valid] = 0.0
        maximum_weight = float(weights.max())
        if maximum_weight <= EPS:
            break
        strong = weights >= 0.20 * maximum_weight
        new_point = np.average(grid[strong], axis=0, weights=weights[strong])
        movement = new_point - current
        if float(np.linalg.norm(movement)) < 0.25:
            break
        new_direction = _unit(movement, direction)
        if float(np.dot(new_direction, direction)) < 0.0:
            new_direction *= -1.0
        direction = _unit(0.55 * direction + 0.45 * new_direction, direction)
        current = new_point
        points.append(current.copy())
        responses.append(float(np.max(local_evidence[strong])))
        length = np.sum(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1)) * config.spacing_mm
        if length >= config.trace_mm:
            break

    return np.asarray(points, dtype=np.float64), np.asarray(responses, dtype=np.float64)


def _path_length(points: np.ndarray, spacing_mm: float) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) * spacing_mm)


def _interpolate_path(points: np.ndarray, distance_mm: float, spacing_mm: float) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) * spacing_mm
    traversed = 0.0
    for index, segment_length in enumerate(segment_lengths):
        if traversed + segment_length >= distance_mm and segment_length > EPS:
            fraction = (distance_mm - traversed) / segment_length
            return points[index] + fraction * (points[index + 1] - points[index])
        traversed += float(segment_length)
    return points[-1]


def estimate_radius(
    point: np.ndarray,
    direction: np.ndarray,
    ct: np.ndarray,
    config: Config,
    statistics: dict[str, float],
) -> float:
    direction = _unit(direction)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(trial, direction))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0])
    axis_one = _unit(np.cross(direction, trial))
    axis_two = _unit(np.cross(direction, axis_one))

    axes = tuple(np.arange(length, dtype=np.float64) for length in ct.shape)
    sampler = RegularGridInterpolator(axes, ct, bounds_error=False, fill_value=-2048.0)
    threshold = max(80.0, statistics["aorta_p10_hu"] - 150.0)
    radial_samples = np.arange(0.5, 10.01, 0.5)
    radii: list[float] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
        radial_direction = math.cos(angle) * axis_one + math.sin(angle) * axis_two
        sample_points = point + np.outer(radial_samples / config.spacing_mm, radial_direction)
        intensities = sampler(sample_points)
        below = np.flatnonzero(intensities < threshold)
        radii.append(float(radial_samples[below[0]]) if below.size else 10.0)
    return float(np.clip(np.median(radii), 0.5, 10.0))


def _extract_cnn_patch(
    center: np.ndarray,
    normalized_ct: np.ndarray,
    signed_distance: np.ndarray,
    vesselness: np.ndarray,
    size: int,
    search_mm: float,
) -> np.ndarray:
    half = size // 2
    center_i = np.rint(center).astype(int)
    channels = [normalized_ct, np.clip(signed_distance / search_mm, -1.0, 1.0), vesselness]
    patch = np.zeros((3, size, size, size), dtype=np.float32)
    for channel_index, volume in enumerate(channels):
        source_low = np.maximum(center_i - half, 0)
        source_high = np.minimum(center_i + (size - half), np.asarray(volume.shape))
        destination_low = source_low - (center_i - half)
        destination_high = destination_low + (source_high - source_low)
        source = tuple(slice(int(a), int(b)) for a, b in zip(source_low, source_high, strict=True))
        destination = tuple(slice(int(a), int(b)) for a, b in zip(destination_low, destination_high, strict=True))
        patch[(channel_index, *destination)] = volume[source]
    return patch


def _load_cnn(checkpoint: Path | None):
    if checkpoint is None:
        return None
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("PyTorch is required when --model is used.") from error

    class Tiny3DCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(3, 12, 3, padding=1), nn.InstanceNorm3d(12), nn.SiLU(), nn.MaxPool3d(2),
                nn.Conv3d(12, 24, 3, padding=1), nn.InstanceNorm3d(24), nn.SiLU(), nn.MaxPool3d(2),
                nn.Conv3d(24, 40, 3, padding=1), nn.InstanceNorm3d(40), nn.SiLU(),
                nn.AdaptiveAvgPool3d(1),
            )
            self.classifier = nn.Linear(40, 1)

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1)).squeeze(1)

    model = Tiny3DCNN()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()
    return model


def _score_candidate(
    anchor: np.ndarray,
    path: np.ndarray,
    responses: np.ndarray,
    evidence: np.ndarray,
    ct: np.ndarray,
    normalized_ct: np.ndarray,
    signed_distance: np.ndarray,
    vesselness: np.ndarray,
    config: Config,
    statistics: dict[str, float],
    cnn,
) -> tuple[float, str]:
    if cnn is not None:
        import torch

        patch = _extract_cnn_patch(
            anchor, normalized_ct, signed_distance, vesselness, config.patch_size, config.search_mm
        )
        with torch.inference_mode():
            probability = torch.sigmoid(cnn(torch.from_numpy(patch[None]))).item()
        return float(probability), "tiny_3d_cnn"

    maximum = max(float(evidence.max()), EPS)
    peak_ratio = float(evidence[tuple(np.rint(anchor).astype(int))]) / maximum
    response_ratio = float(np.mean(responses)) / maximum
    anchor_gap = float(np.linalg.norm(path[1] - path[0]) * config.spacing_mm)
    tracked_length = _path_length(path[1:], config.spacing_mm)
    target_tracked_length = max(4.0, config.trace_mm - anchor_gap)
    length_ratio = min(1.0, tracked_length / target_tracked_length)
    anchor_hu = float(ct[tuple(np.rint(anchor).astype(int))])
    scale = max(50.0, statistics["aorta_p95_hu"] - statistics["aorta_p05_hu"])
    intensity_ratio = math.exp(-0.5 * ((anchor_hu - statistics["aorta_median_hu"]) / (2.5 * scale)) ** 2)
    logit = -3.8 + 3.0 * peak_ratio + 2.1 * response_ratio + 1.8 * length_ratio + 1.1 * intensity_ratio
    probability = 1.0 / (1.0 + math.exp(-logit))
    return float(probability), "geometric_prototype"


def _deduplicate(branches: list[Branch], config: Config) -> list[Branch]:
    retained: list[Branch] = []
    for branch in sorted(branches, key=lambda item: item.probability, reverse=True):
        ostium = np.asarray(branch.ostium)
        direction = np.asarray(branch.direction)
        duplicate = False
        for existing in retained:
            distance = float(np.linalg.norm(ostium - np.asarray(existing.ostium)))
            cosine = float(np.dot(direction, np.asarray(existing.direction)))
            if distance < config.deduplicate_mm and cosine > 0.35:
                duplicate = True
                break
        if not duplicate:
            retained.append(branch)
        if len(retained) >= config.maximum_branches:
            break
    return retained


def detect_branches(
    ct_path: Path,
    mask_path: Path,
    config: Config,
    model_path: Path | None = None,
    debug_directory: Path | None = None,
) -> tuple[list[Branch], dict[str, object]]:
    ct, mask, affine, statistics = load_local_roi(ct_path, mask_path, config)
    vesselness, evidence, signed_distance, nearest = compute_vesselness(ct, mask, config, statistics)
    exterior_distance = np.maximum(signed_distance, 0.0)
    candidates = propose_candidates(evidence, exterior_distance, config)

    lower = min(-100.0, statistics["aorta_p10_hu"] - 300.0)
    upper = max(lower + 300.0, statistics["aorta_p90_hu"] + 150.0)
    normalized_ct = np.clip((ct - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    cnn = _load_cnn(model_path)
    branches: list[Branch] = []

    for anchor in candidates:
        anchor_index = np.clip(np.rint(anchor).astype(int), 0, np.asarray(mask.shape) - 1)
        surface = nearest[(slice(None), *anchor_index)].astype(np.float64)
        path, responses = trace_candidate(anchor, surface, evidence, signed_distance, config)
        trace_length = _path_length(path, config.spacing_mm)
        tracked_length = _path_length(path[1:], config.spacing_mm)
        if tracked_length < 3.0:
            continue

        direction_index = _unit(path[-1] - _interpolate_path(path, 2.0, config.spacing_mm))
        seed_index = _interpolate_path(path, min(3.0, 0.35 * trace_length), config.spacing_mm)
        radius_point = _interpolate_path(path, min(6.0, 0.60 * trace_length), config.spacing_mm)
        radius = estimate_radius(radius_point, direction_index, ct, config, statistics)
        probability, scoring_method = _score_candidate(
            anchor,
            path,
            responses,
            evidence,
            ct,
            normalized_ct,
            signed_distance,
            vesselness,
            config,
            statistics,
            cnn,
        )
        if probability < config.probability_threshold:
            continue

        world_path = _apply_affine(affine, path)
        world_ostium = world_path[0]
        world_seed = _apply_affine(affine, seed_index)[0]
        world_direction = _unit(affine[:3, :3] @ direction_index)
        branches.append(
            Branch(
                probability=round(probability, 6),
                ostium=np.round(world_ostium, 3).tolist(),
                seed=np.round(world_seed, 3).tolist(),
                radius_mm=round(radius, 3),
                direction=np.round(world_direction, 6).tolist(),
                centerline=np.round(world_path, 3).tolist(),
                trace_length_mm=round(trace_length, 3),
                proposal_response=round(float(evidence[tuple(anchor_index)]), 6),
                scoring_method=scoring_method,
            )
        )

    branches = _deduplicate(branches, config)
    metadata: dict[str, object] = {
        "raw_candidate_count": len(candidates),
        "accepted_branch_count": len(branches),
        "roi_shape": list(ct.shape),
        "aorta_statistics": statistics,
        "config": asdict(config),
    }

    if debug_directory is not None:
        debug_directory.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(vesselness, affine), debug_directory / "vesselness.nii.gz")
        nib.save(nib.Nifti1Image(evidence, affine), debug_directory / "candidate_evidence.nii.gz")
        nib.save(nib.Nifti1Image(signed_distance.astype(np.float32), affine), debug_directory / "aorta_signed_distance.nii.gz")
        candidate_map = np.zeros(ct.shape, dtype=np.uint16)
        for number, point in enumerate(candidates, start=1):
            candidate_map[tuple(np.rint(point).astype(int))] = number
        nib.save(nib.Nifti1Image(candidate_map, affine), debug_directory / "candidate_points.nii.gz")

    return branches, metadata


def write_result(
    output_path: Path,
    subject_name: str,
    branches: list[Branch],
    metadata: dict[str, object],
    coordinate_system: str = "LPS",
) -> None:
    if coordinate_system not in {"LPS", "RAS"}:
        raise ValueError("coordinate_system must be LPS or RAS")

    def convert(point: Sequence[float], is_direction: bool = False) -> list[float]:
        converted = np.asarray(point, dtype=np.float64).copy()
        if coordinate_system == "LPS":
            converted[:2] *= -1.0
        decimals = 6 if is_direction else 3
        return np.round(converted, decimals).tolist()

    branch_records = []
    legacy_seeds = []
    for number, branch in enumerate(branches, start=1):
        record = {
            "id": number,
            **asdict(branch),
            "ostium": convert(branch.ostium),
            "seed": convert(branch.seed),
            "direction": convert(branch.direction, is_direction=True),
            "centerline": [convert(point) for point in branch.centerline],
        }
        branch_records.append(record)
        seed = np.asarray(record["seed"], dtype=np.float64)
        direction = np.asarray(record["direction"], dtype=np.float64)
        legacy_seeds.append(
            [
                np.round(seed, 3).tolist(),
                np.round(seed + 4.0 * direction, 3).tolist(),
                branch.radius_mm,
            ]
        )

    payload = {
        "name": subject_name,
        "coordinate_system": f"{coordinate_system}_mm",
        "branches": branch_records,
        "seeds": legacy_seeds,
        "metadata": metadata,
        "warning": "Research prototype; not for clinical use.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path, help="Input CT NIfTI file.")
    parser.add_argument("--aorta-mask", required=True, type=Path, help="Binary aorta-mask NIfTI file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file.")
    parser.add_argument("--subject", help="Subject name; defaults to the CT filename stem.")
    parser.add_argument("--model", type=Path, help="Optional trained Tiny3DCNN checkpoint.")
    parser.add_argument("--debug-dir", type=Path, help="Optional directory for diagnostic NIfTI volumes.")
    parser.add_argument(
        "--coordinate-system",
        choices=("LPS", "RAS"),
        default="LPS",
        help="JSON coordinate convention. LPS is the SeqSeg/SimpleITK-compatible default.",
    )
    parser.add_argument("--search-mm", type=float, default=10.0)
    parser.add_argument("--trace-mm", type=float, default=12.0)
    parser.add_argument("--probability-threshold", type=float, default=0.42)
    parser.add_argument("--maximum-branches", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = Config(
        search_mm=arguments.search_mm,
        trace_mm=arguments.trace_mm,
        probability_threshold=arguments.probability_threshold,
        maximum_branches=arguments.maximum_branches,
    )
    branches, metadata = detect_branches(
        arguments.ct,
        arguments.aorta_mask,
        config,
        model_path=arguments.model,
        debug_directory=arguments.debug_dir,
    )
    subject = arguments.subject or arguments.ct.name.removesuffix(".gz").removesuffix(".nii")
    write_result(arguments.output, subject, branches, metadata, coordinate_system=arguments.coordinate_system)
    print(
        f"Detected {len(branches)} branches from {metadata['raw_candidate_count']} candidates; "
        f"wrote {arguments.output}"
    )


if __name__ == "__main__":
    main()
