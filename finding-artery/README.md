# Branchseed

Detect aortic daughter vessels from a CT and its supplied aorta mask.

## Place the data

Create a `data` folder inside `finding-artery`. Put every study in its own folder,

```text
finding-artery/
├── data/
│   ├── subject001/
│   │   ├── orig1.nii
│   │   └── mask1.nii
│   ├── subject002/
│   │   ├── orig2.nii
│   │   └── mask2.nii
│   └── ...
├── predictions/
├── visual_checks/
├── run.py
├── demo.py
├── batch.py
├── README.md
└── requirements.txt
```

The CT filename must start with `orig`, and the corresponding aorta-mask
filename must start with `mask`. Files may end in `.nii` or `.nii.gz`.

## Setup

Use Python 3.12. Open Terminal in the `finding-artery` folder, then run:

```bash
python -m pip install -r requirements.txt
```

Alternatively, create and activate the supplied Conda environment:

```bash
conda env create -f environment.yml && conda activate finding-artery
```

Include both `requirements.txt` and `environment.yml` with the submission.
The environment file uses the pinned dependencies in `requirements.txt`;
they cover detection and the optional visualization. No GPU is required.

## Run

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

- Replace `image.nii.gz` with that study's CT path, such as
`data/subject001/orig1.nii`. 
- Replace `aorta_mask.nii.gz` with its aorta-mask
path, such as `data/subject001/mask1.nii`.

The result is saved as
`prediction.json` inside the `finding-artery` folder.

For an interactive 3D CT review, run:

```bash
python demo.py --image image.nii --aorta-mask aorta_mask.nii
```

- Replace `image.nii.gz` with that study's CT path, such as
`data/subject001/orig1.nii`. 
- Replace `aorta_mask.nii.gz` with its aorta-mask
path, such as `data/subject001/mask1.nii`.

The demo automatically saves both files under `visual_checks/` and prefixes
them with the study-folder name. For example, running it on
`data/subject001/orig1.nii`
creates:

```text
visual_checks/subject001_prediction.json
visual_checks/subject001_prediction_3d.html
```
Open the HTML in a browser to view three CT planes and a 3D overview.
