import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from aortic_branch_prototype import Config, detect_branches, write_result


def test_synthetic_branch(tmp_path: Path) -> None:
    shape = (80, 80, 72)
    grid = np.indices(shape)
    ct = np.zeros(shape, dtype=np.float32)
    aorta = ((grid[0] - 35) ** 2 + (grid[1] - 38) ** 2 <= 10**2)
    branch = (
        (grid[0] >= 44)
        & (grid[0] <= 68)
        & ((grid[1] - 38) ** 2 + (grid[2] - 36) ** 2 <= 3**2)
    )
    ct[aorta | branch] = 350.0
    ct += np.random.default_rng(7).normal(0.0, 4.0, shape).astype(np.float32)

    ct_path = tmp_path / "ct.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_path = tmp_path / "branches.json"
    nib.save(nib.Nifti1Image(ct, np.eye(4)), ct_path)
    nib.save(nib.Nifti1Image(aorta.astype(np.uint8), np.eye(4)), mask_path)

    config = Config(
        spacing_mm=1.0,
        sigmas_mm=(1.0, 1.8, 2.6),
        search_mm=12.0,
        trace_mm=10.0,
        probability_threshold=0.20,
        maximum_candidates=48,
    )
    branches, metadata = detect_branches(ct_path, mask_path, config)
    write_result(output_path, "synthetic", branches, metadata, coordinate_system="RAS")
    result = json.loads(output_path.read_text())

    assert result["coordinate_system"] == "RAS_mm"
    assert result["branches"]
    expected_ostium = np.array([45.0, 38.0, 36.0])
    distances = [np.linalg.norm(np.asarray(item["ostium"]) - expected_ostium) for item in result["branches"]]
    assert min(distances) < 6.0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_synthetic_branch(Path(directory))
    print("Synthetic branch test passed.")
