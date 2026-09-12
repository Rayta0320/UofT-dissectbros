# Aortic branch detector prototype

This program detects vessel-like structures that originate within a short physical
distance of a supplied aorta mask. It outputs the estimated ostium, an outward seed,
radius, direction, short centerline, and branch probability as JSON.

The default probability is a transparent geometric prototype score. A trained tiny
3-D CNN checkpoint can be supplied with `--model`; the CNN receives three channels:
normalized CT, aorta signed distance, and Frangi vesselness.

## Run on the supplied data

The existing `seqseg` conda environment already contains the required libraries:

```bash
/opt/anaconda3/envs/seqseg/bin/python \
  MLProgram/aortic_branch_prototype.py \
  --ct "/Users/raytamorimoto/Downloads/TORALIS CHALLENGE /subject001/orig1.nii" \
  --aorta-mask "/Users/raytamorimoto/Downloads/TORALIS CHALLENGE /subject001/mask1.nii" \
  --subject subject001 \
  --output MLProgram/prototype_output/subject001.json \
  --debug-dir MLProgram/prototype_output/subject001_debug
```

Subjects 16–25 are gzip-compressed but use a `.nii` filename. The loader recognizes
and handles those files without altering the originals.

Useful tuning switches:

```text
--search-mm 10
--trace-mm 12
--probability-threshold 0.42
--maximum-branches 16
```

Lower the probability threshold to inspect more proposals. Debug output includes the
vesselness volume, candidate evidence, signed distance, and candidate points. The
debug images use the original NIfTI RAS+ world coordinates.

Run the self-contained synthetic smoke test with:

```bash
/opt/anaconda3/envs/seqseg/bin/python MLProgram/test_aortic_branch_prototype.py
```

## JSON

`branches` contains the richer output. `seeds` is also emitted in the legacy SeqSeg
shape `[first_point, second_point, radius]`, with the second point placed 4 mm along
the predicted direction. JSON coordinates default to LPS millimetres for compatibility
with SeqSeg/SimpleITK. Pass `--coordinate-system RAS` when NIfTI/nibabel-style world
coordinates are needed. Diagnostic NIfTI files retain their ordinary NIfTI affine.

The program is a research prototype, not a clinically validated device. The default
score should be treated as a ranking score until it has been calibrated on held-out
subjects. Split evaluation by patient, never by candidate patch.

## Visually verify the branches

Run the local three-plane CT viewer from inside `MLProgram`:

```bash
/opt/anaconda3/envs/seqseg/bin/python ./visualize_branches.py \
  --ct "/Users/raytamorimoto/Downloads/TORALIS CHALLENGE /subject001/orig1.nii" \
  --aorta-mask "/Users/raytamorimoto/Downloads/TORALIS CHALLENGE /subject001/mask1.nii" \
  --predictions ./prototype_output/subject001.json \
  --open
```

The viewer stays on the local computer and shows synchronized axial, coronal, and
sagittal slices. Scroll over an image or use its slider to traverse the entire scan.
Click a numbered branch in the right panel to jump all three views to its predicted
ostium. The colored line is the short traced centerline; the cyan region is the
supplied aorta mask. Adjust the minimum score to hide lower-ranked candidates. Mark
each selected proposal as a true branch or false positive; the confirmed count updates
immediately and is retained in this browser. **Download verification JSON** saves the
manual audit as a separate file.
