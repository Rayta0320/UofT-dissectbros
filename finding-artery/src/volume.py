"""Loading, cropping and coordinate handling.

The ONE rule in this file: SimpleITK indexes are (x, y, z); numpy arrays from
GetArrayFromImage are (z, y, x). Everything inside the pipeline works in
*cropped numpy (z, y, x) voxel space*. Only Volume.to_mm() converts out.
"""
from dataclasses import dataclass
import gzip
from pathlib import Path
import shutil
import tempfile
import warnings
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


@dataclass
class Volume:
    ct: np.ndarray          # (z, y, x) float32, HU, cropped
    aorta: np.ndarray       # (z, y, x) bool, cropped
    offset: tuple           # (z0, y0, x0) crop origin in full-volume voxels
    spacing_zyx: tuple      # (sz, sy, sx) in mm
    ref: sitk.Image         # original image, for coordinate transforms
    full_shape: tuple       # (nz, ny, nx) of the uncropped volume

    def to_mm(self, zyx):
        """Cropped (z, y, x) voxel coords -> physical mm. Accepts floats."""
        z, y, x = zyx
        idx = (float(x) + self.offset[2],
               float(y) + self.offset[1],
               float(z) + self.offset[0])
        return list(self.ref.TransformContinuousIndexToPhysicalPoint(idx))

    def direction_to_mm(self, vec_zyx):
        """Voxel-space direction -> physical unit vector.

        Scale by spacing, then rotate by the direction cosine matrix. Do NOT
        use to_mm() on two points and subtract if you can avoid it; this is
        the same thing but explicit about what it is doing.
        """
        vz, vy, vx = vec_zyx
        v = np.array([vx * self.spacing_zyx[2],
                      vy * self.spacing_zyx[1],
                      vz * self.spacing_zyx[0]])
        D = np.array(self.ref.GetDirection()).reshape(3, 3)
        v = D @ v
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).tolist()

    @property
    def mm_per_vox(self):
        return np.array(self.spacing_zyx)


def _read_image(path):
    """Also accept the supplied gzip streams that have a plain .nii suffix."""
    path = Path(path)
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    if compressed and path.suffix.lower() == ".nii":
        with tempfile.TemporaryDirectory(prefix="branchseed-nifti-") as tmp:
            unpacked = Path(tmp) / "volume.nii"
            with gzip.open(path, "rb") as source, unpacked.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            return sitk.ReadImage(str(unpacked))
    return sitk.ReadImage(str(path))


def _resample_sheared_pair(image_path, mask_path):
    """Preserve NIfTI world geometry when ITK cannot represent a sheared grid.

    Both arrays are resampled to one orthogonal grid enclosing the source.
    CT uses linear interpolation; mask uses nearest neighbour. Merely replacing
    the header's direction matrix would move anatomy in physical space.
    """
    import nibabel as nib

    def read(path):
        with open(path, "rb") as source:
            compressed = source.read(2) == b"\x1f\x8b"
        if compressed and Path(path).suffix.lower() == ".nii":
            with gzip.open(path, "rb") as source:
                return nib.Nifti1Image.from_bytes(source.read())
        return nib.load(str(path))

    image, mask = read(image_path), read(mask_path)
    if len(image.shape) != 3 or len(mask.shape) != 3:
        raise ValueError("image and aorta mask must be three-dimensional")
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine,
                                                    rtol=0, atol=1e-5):
        raise ValueError("grid mismatch: image and mask NIfTI affines or sizes differ")
    source_affine = image.affine
    u, _, vt = np.linalg.svd(source_affine[:3, :3])
    rotation = u @ vt
    spacing = np.linalg.norm(source_affine[:3, :3], axis=0)
    basis = rotation * spacing
    corners = np.array(np.meshgrid(*[(0, n - 1) for n in image.shape],
                                   indexing="ij")).reshape(3, -1)
    projected = np.linalg.solve(basis, source_affine[:3, :3] @ corners)
    low, high = np.floor(projected.min(axis=1)), np.ceil(projected.max(axis=1))
    shape = tuple((high - low + 1).astype(int))
    destination = np.eye(4)
    destination[:3, :3] = basis
    destination[:3, 3] = source_affine[:3, 3] + basis @ low
    mapping = np.linalg.solve(source_affine, destination)
    ras_to_lps = np.diag([-1., -1., 1.])
    results = []
    for source, is_mask in ((image, False), (mask, True)):
        data = np.asarray(source.dataobj)
        data = (data > 0).astype(np.uint8) if is_mask else data.astype(np.float32)
        output = ndi.affine_transform(
            data, mapping[:3, :3], offset=mapping[:3, 3], output_shape=shape,
            order=0 if is_mask else 1, mode="constant", cval=0 if is_mask else -1024,
            prefilter=False)
        itk = sitk.GetImageFromArray(output.transpose(2, 1, 0))
        itk.SetSpacing(tuple(spacing))
        itk.SetOrigin(tuple(ras_to_lps @ destination[:3, 3]))
        itk.SetDirection(tuple((ras_to_lps @ rotation).ravel()))
        results.append(itk)
    warnings.warn("Resampled non-orthogonal CT/mask grid while preserving physical coordinates",
                  RuntimeWarning, stacklevel=2)
    return results


def load(image_path, mask_path, margin_mm=25.0):
    try:
        img = _read_image(image_path)
        msk = _read_image(mask_path)
    except RuntimeError as exc:
        if "orthonormal" not in str(exc):
            raise
        img, msk = _resample_sheared_pair(image_path, mask_path)

    if img.GetDimension() != 3 or msk.GetDimension() != 3:
        raise ValueError("image and aorta mask must be three-dimensional")

    if img.GetSize() != msk.GetSize():
        raise ValueError(f"grid mismatch: {img.GetSize()} vs {msk.GetSize()}")
    for name in ("Spacing", "Origin", "Direction"):
        if not np.allclose(getattr(img, "Get" + name)(), getattr(msk, "Get" + name)(),
                           rtol=0, atol=1e-5):
            raise ValueError(f"grid mismatch: image and mask {name.lower()} differ")

    ct_full = sitk.GetArrayFromImage(img).astype(np.float32)
    ao_full = sitk.GetArrayFromImage(msk) > 0
    if not ao_full.any():
        raise ValueError("aorta mask is empty")

    sx, sy, sz = img.GetSpacing()
    spacing_zyx = (sz, sy, sx)

    # crop to a box around the aorta -- this is the single biggest speedup
    idx = np.argwhere(ao_full)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    pad = np.ceil(margin_mm / np.array(spacing_zyx)).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, ao_full.shape)

    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return Volume(
        ct=ct_full[sl],
        aorta=ao_full[sl],
        offset=tuple(int(v) for v in lo),
        spacing_zyx=spacing_zyx,
        ref=img,
        full_shape=ao_full.shape,
    )


def lumen_threshold(vol, pct=5.0, margin_hu=0.0):
    """Per-case contrast threshold from the HU distribution inside the aorta.

    NEVER hardcode a number here. Contrast timing varies hugely between scans.
    """
    inside = vol.ct[vol.aorta]
    return float(np.percentile(inside, pct) - margin_hu)


def bone_ceiling(vol, k=2.5):
    """Upper bound so vertebral bone / calcification is not read as lumen."""
    inside = vol.ct[vol.aorta]
    return float(inside.mean() + k * inside.std() + 120.0)


def blood_model(vol, erosion_mm=1.5, sigma_limit=3.0):
    """Robust per-case blood statistics from the eroded mask interior.

    Median/MAD resist sparse calcification; erosion reduces boundary partial
    volume. The one-HU scale floor only handles quantized/constant test data.
    No neighbouring tissue or fixed blood-HU threshold is used.
    """
    depth = ndi.distance_transform_edt(np.pad(vol.aorta, 1), sampling=vol.spacing_zyx)
    depth = depth[1:-1, 1:-1, 1:-1]
    core = vol.aorta & (depth > erosion_mm) & np.isfinite(vol.ct)
    fallback = np.count_nonzero(core) < 32
    if fallback:
        interior = depth[vol.aorta]
        core = vol.aorta & (depth >= np.percentile(interior, 50)) & np.isfinite(vol.ct)
    values = vol.ct[core]
    if not len(values):
        raise ValueError("no finite aortic blood samples")
    centre = float(np.median(values))
    sigma = max(1.0, float(1.4826 * np.median(np.abs(values - centre))))
    return dict(median_hu=centre, sigma_hu=sigma,
                lower_hu=centre - sigma_limit * sigma,
                upper_hu=centre + sigma_limit * sigma,
                sample_count=int(len(values)), erosion_mm=float(erosion_mm),
                used_core_fallback=bool(fallback),
                q05_hu=float(np.percentile(values, 5)),
                q95_hu=float(np.percentile(values, 95)))


def boundary_blood_model(vol, core_model):
    """A second, partial-volume hypothesis learned only from the parent mask.

    The lower quartile of its inner boundary includes partially occupied blood
    voxels which erosion deliberately excludes from the core model. The strict
    core hypothesis is also run separately; this range never replaces it.
    The 25th percentile is an explicit prototype heuristic, not a learned prior.
    """
    depth = ndi.distance_transform_edt(np.pad(vol.aorta, 1), sampling=vol.spacing_zyx)[1:-1, 1:-1, 1:-1]
    width = max(1.5, float(min(vol.spacing_zyx)))
    boundary = vol.aorta & (depth <= width) & np.isfinite(vol.ct)
    values = vol.ct[boundary]
    lower = float(np.percentile(values, 25)) if len(values) >= 32 else core_model["lower_hu"]
    return dict(core_model, lower_hu=min(core_model["lower_hu"], lower),
                core_lower_hu=core_model["lower_hu"], intensity_source="parent_boundary_q25",
                boundary_sample_count=int(len(values)), boundary_width_mm=width,
                boundary_q25_hu=lower)
