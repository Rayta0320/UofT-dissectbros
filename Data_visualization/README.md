# TORALIS cohort label viewer

A zero-dependency local web viewer for comparing every TORALIS CT scan with its paired segmentation mask. It supports synchronized axial, coronal, and sagittal navigation; per-subject peak-label alignment; overlays, outlines, label-only and side-by-side modes; CT window presets; filtering; and adjustable grid density.

## Run

```bash
python3 server.py
```

Open <http://127.0.0.1:8000>. The default dataset is:

```text
/Users/raytamorimoto/Downloads/TORALIS CHALLENGE 
```

To use another dataset or port:

```bash
python3 server.py --dataset "/path/to/dataset" --port 8080
```

The dataset should contain `subjectNNN` folders, each with one `orig*.nii` scan and one `mask*.nii` label volume. Both uncompressed NIfTI files and gzip-compressed files with a `.nii` extension are detected automatically.

## Notes

- Only the current 2D slice is sent to the browser, so all 25 subjects can stay visible without loading the full 1.9 GB dataset into browser memory.
- **Peak label** is the default alignment and shows the densest labeled slice for every subject. Choose **Volume position** to navigate all volumes by the same relative percentage instead.
- The first view of compressed subjects 16–25 creates decompressed files under `.cache/nifti`. Later navigation reuses them. Delete that folder at any time to reclaim space; it is regenerated as needed.
- No Python or JavaScript packages are required.
