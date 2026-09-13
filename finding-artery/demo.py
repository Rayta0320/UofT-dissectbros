#!/usr/bin/env python3
"""Run Branchseed and create a self-contained interactive 3D CT review."""
import argparse
from pathlib import Path
import re
import sys
import time

from src import pipeline
from src.viz3d import write_html


def _study_name(image_path):
    """Return a filesystem-safe study name, normally the image's parent folder."""
    image = Path(image_path)
    name = image.parent.name
    if not name or name.lower() in {"data", "branchseed", "."}:
        name = image.name.removesuffix(".gz").removesuffix(".nii")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-") or "study"


def _output_paths(image_path, requested_html=None):
    """Place review artifacts in visual_checks with a study-specific prefix."""
    study = _study_name(image_path)
    output_dir = Path(__file__).resolve().parent / "visual_checks"
    output = output_dir / f"{study}_prediction.json"

    if requested_html:
        html_name = Path(requested_html).name
        if not html_name.lower().endswith(".html"):
            html_name += ".html"
        if not html_name.startswith(study + "_"):
            html_name = f"{study}_{html_name}"
        html = output_dir / html_name
    else:
        html = output.with_name(output.stem + "_3d.html")
    return study, output, html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="input CTA NIfTI file")
    parser.add_argument("--aorta-mask", required=True, help="input aorta-mask NIfTI file")
    parser.add_argument("--html",
                        help="optional HTML filename; also saved under visual_checks")
    parser.add_argument("--min-wall-departure-mm", type=float, default=4.0,
                        help="separation below which stronger opening/lumen evidence is required (default: 4 mm)")
    args = parser.parse_args()

    case_id, output, html = _output_paths(args.image, args.html)
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result, debug = pipeline.run_case(args.image, args.aorta_mask, case_id=case_id,
                                      method="growth", growth_connectivity=26,
                                      growth_intensity="adaptive", curve_geometry=True,
                                      min_wall_departure_mm=args.min_wall_departure_mm)
    pipeline.write_json(result, output)
    write_html(debug, result, html)
    elapsed = time.perf_counter() - start
    print(f"{case_id}: {len(result['daughters'])} daughters in {elapsed:.1f}s", file=sys.stderr)
    print(f"prediction: {output.resolve()}")
    print(f"3D view:    {html.resolve()}")


if __name__ == "__main__":
    main()
