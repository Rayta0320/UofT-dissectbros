#!/usr/bin/env python3
"""Run every case and write predictions plus visual checks."""
import argparse, glob, os, time, json
from src import pipeline, viz

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="predictions")
ap.add_argument("--looks", default="visual_checks")
ap.add_argument("--method", choices=("collar", "growth"), default="growth")
ap.add_argument("--growth-connectivity", type=int, choices=(6, 18, 26), default=26)
ap.add_argument("--growth-intensity", choices=("strict", "adaptive"), default="adaptive")
ap.add_argument("--curve-length", choices=("spline", "polyline"), default="spline")
ap.add_argument("--diagnostics", help="optional directory for candidate diagnostics")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs(a.looks, exist_ok=True)
if a.diagnostics:
    os.makedirs(a.diagnostics, exist_ok=True)

times = []
for d in sorted(glob.glob(os.path.join(a.data, "*"))):
    if not os.path.isdir(d):
        continue
    case = os.path.basename(d)
    img = (glob.glob(os.path.join(d, "orig*.nii*")) or [None])[0]
    msk = (glob.glob(os.path.join(d, "mask*.nii*")) or [None])[0]
    if not img or not msk:
        print(f"skip {case}: missing files"); continue
    t = time.time()
    try:
        res, dbg = pipeline.run_case(img, msk, case_id=case, method=a.method,
                                    growth_connectivity=a.growth_connectivity,
                                    growth_intensity=a.growth_intensity,
                                    curve_geometry=a.curve_length == "spline")
        pipeline.write_json(res, os.path.join(a.out, case + ".json"))
        if a.diagnostics:
            pipeline.write_diagnostics(dbg, res, os.path.join(a.diagnostics, case + ".json"))
        viz.overlay(dbg, res, os.path.join(a.looks, case + ".png"))
        dt = time.time() - t; times.append(dt)
        print(f"{case:<14} {len(res['daughters']):>3} daughters  {dt:>5.1f}s")
    except Exception as e:
        print(f"{case:<14} FAILED: {type(e).__name__}: {e}")
        pipeline.write_json({"case_id": case, "parent": {"instance_id": "aorta"},
                             "daughters": []}, os.path.join(a.out, case + ".json"))
if times:
    print(f"\nmean {sum(times)/len(times):.1f}s  max {max(times):.1f}s over {len(times)} cases")
