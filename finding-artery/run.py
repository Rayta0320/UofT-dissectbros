#!/usr/bin/env python3
"""Submission entry point for direct aortic branch-origin detection."""
import argparse
import os
import sys
import time
from pathlib import Path

from src import pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="input CTA NIfTI file")
    ap.add_argument("--aorta-mask", required=True, help="input aorta-mask NIfTI file")
    ap.add_argument("--output", required=True, help="output prediction JSON file")
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--method", choices=("collar", "growth"), default="growth",
                    help="detection method (submission default: growth)")
    ap.add_argument("--diagnostics", help="optional per-candidate metrics JSON")
    ap.add_argument("--growth-connectivity", type=int, choices=(6, 18, 26), default=26,
                    help="adjacency for growth, markers and watershed; growth method only")
    ap.add_argument("--growth-intensity", choices=("strict", "adaptive"), default="adaptive",
                    help="adaptive adds a parent-boundary intensity hypothesis")
    ap.add_argument("--min-wall-departure-mm", type=float, default=4.0,
                    help="separation below which stronger opening/lumen evidence is required (default: 4 mm)")
    ap.add_argument("--curve-length", choices=("spline", "polyline"), default="spline",
                    help="growth centerline length: lumen-checked cubic arc integral or original polyline")
    ap.add_argument("--viz", default=None,
                    help="optional detailed review PNG; also writes *_overview.png")
    args = ap.parse_args()

    case_id = args.case_id or os.path.basename(os.path.dirname(os.path.abspath(args.image))) or "case"
    t0 = time.time()
    result, dbg = pipeline.run_case(args.image, args.aorta_mask, case_id=case_id,
                                    method=args.method, growth_connectivity=args.growth_connectivity,
                                    growth_intensity=args.growth_intensity,
                                    min_wall_departure_mm=args.min_wall_departure_mm,
                                    curve_geometry=args.curve_length == "spline")
    for target in (args.output, args.diagnostics, args.viz):
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(result, args.output)
    if args.diagnostics:
        pipeline.write_diagnostics(dbg, result, args.diagnostics)
    dt = time.time() - t0

    print(f"{case_id}: {len(result['daughters'])} daughters, {dt:.1f}s", file=sys.stderr)
    if args.viz:
        from src import viz
        viz.overlay(dbg, result, args.viz)


if __name__ == "__main__":
    main()
