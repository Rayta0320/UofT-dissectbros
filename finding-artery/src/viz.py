"""Branch review images from CT, aorta mask, and traced daughter paths."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def overview(dbg, result, out_png):
    """Whole-aorta overview. Useful for location, but not final review."""
    vol = dbg["vol"]
    ct, ao = vol.ct, vol.aorta
    sz, sy, sx = vol.spacing_zyx

    b = ct.copy()
    b[b > dbg["ceil"]] = -200                     # knock out bone

    ys = np.where(ao.any(axis=(0, 2)))[0]
    xs = np.where(ao.any(axis=(0, 1)))[0]
    cy, cx = int(ys.mean()), int(xs.mean())
    slab_y = max(6, int(25 / sy))
    slab_x = max(6, int(25 / sx))

    cor = b[:, max(0, cy - slab_y):cy + slab_y, :].max(axis=1)
    cor_m = ao[:, max(0, cy - slab_y):cy + slab_y, :].max(axis=1)
    sag = b[:, :, max(0, cx - slab_x):cx + slab_x].max(axis=2)
    sag_m = ao[:, :, max(0, cx - slab_x):cx + slab_x].max(axis=2)

    tr = dbg["traced"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    for ax, bg, mm, hax, name in [
            (axes[0], cor, cor_m, 2, "coronal MIP  (x-z)"),
            (axes[1], sag, sag_m, 1, "sagittal MIP  (y-z)")]:
        nz = bg.shape[0]
        ax.imshow(bg, cmap="gray", vmin=50, vmax=450,
                  origin="lower", aspect=sz / (sx if hax == 2 else sy))
        ax.contour(mm, levels=[0.5], colors="lime", linewidths=0.9, origin="lower")

        for d, meta in zip(tr, result["daughters"]):
            o, s = d["ostium_vox"], d["seed_vox"]
            ax.plot(o[hax], o[0], "o", ms=6, mfc="none", mec="red", mew=1.6)
            ax.annotate("", xy=(s[hax], s[0]), xytext=(o[hax], o[0]),
                        arrowprops=dict(arrowstyle="->", color="orange", lw=1.6))
            ax.text(o[hax] + 3, o[0] + 2, meta["instance_id"].replace("branch_", "b"),
                    color="yellow", fontsize=7)
            p = d["path"]
            ax.plot(p[:, hax], p[:, 0], "-", color="deepskyblue", lw=1.0, alpha=0.9)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    fig.suptitle(f"{result['case_id']}: {len(result['daughters'])} daughters "
                 f"(red = ostium, orange = direction, blue = traced path)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _bounded_slice(center, radius_mm, spacing, size):
    radius = int(np.ceil(radius_mm / spacing))
    centre = int(np.rint(center))
    return slice(max(0, centre - radius), min(size, centre + radius + 1))


def _local_projection(ct, mask, centre, path, spacing, horizontal_axis,
                      vertical_axis, depth_axis, crop_mm, slab_margin_mm):
    """Local MIP containing the full trace plus a small depth margin."""
    shape = np.array(ct.shape)
    hslice = _bounded_slice(centre[horizontal_axis], crop_mm, spacing[horizontal_axis],
                            shape[horizontal_axis])
    vslice = _bounded_slice(centre[vertical_axis], crop_mm, spacing[vertical_axis],
                            shape[vertical_axis])
    depth_lo = max(0, int(np.floor(path[:, depth_axis].min()
                                   - slab_margin_mm / spacing[depth_axis])))
    depth_hi = min(shape[depth_axis], int(np.ceil(path[:, depth_axis].max()
                                                  + slab_margin_mm / spacing[depth_axis])) + 1)
    slices = [slice(None)] * 3
    slices[horizontal_axis] = hslice
    slices[vertical_axis] = vslice
    slices[depth_axis] = slice(depth_lo, depth_hi)
    block = ct[tuple(slices)]
    mask_block = mask[tuple(slices)]
    # After removing depth, remaining axes retain their NumPy order. Reorder
    # them so imshow receives (vertical, horizontal).
    remaining = [axis for axis in range(3) if axis != depth_axis]
    projection = block.max(axis=depth_axis)
    mask_projection = mask_block.max(axis=depth_axis)
    if remaining != [vertical_axis, horizontal_axis]:
        projection = projection.T
        mask_projection = mask_projection.T
    extent = [hslice.start, hslice.stop - 1, vslice.start, vslice.stop - 1]
    return projection, mask_projection, extent


def review_sheet(dbg, result, out_png, crop_mm=15.0, slab_margin_mm=1.5):
    """Create one zoomable row of three local views for every detected branch."""
    vol = dbg["vol"]
    ct = vol.ct.copy()
    ct[ct > dbg["ceil"]] = -200
    traces = list(zip(dbg["traced"], result["daughters"]))
    traces.sort(key=lambda item: item[0]["ostium_vox"][0], reverse=True)
    if not traces:
        return overview(dbg, result, out_png)

    views = [
        # horizontal, vertical, projection-depth, title
        (2, 1, 0, "axial (x-y)"),
        (2, 0, 1, "coronal (x-z)"),
        (1, 0, 2, "sagittal (y-z)"),
    ]
    rows = len(traces)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 2.8 * rows), squeeze=False)
    vmin = max(-100.0, dbg["thr"] - 180.0)
    vmax = min(dbg["ceil"], dbg["thr"] + 300.0)

    for row, (daughter, meta) in enumerate(traces):
        ostium = np.asarray(daughter["ostium_vox"])
        seed = np.asarray(daughter["seed_vox"])
        path = np.asarray(daughter["path"])
        centre = (ostium + seed) / 2
        for column, (haxis, vaxis, daxis, title) in enumerate(views):
            ax = axes[row, column]
            image, mask, extent = _local_projection(
                ct, vol.aorta, centre, path, vol.spacing_zyx,
                haxis, vaxis, daxis, crop_mm, slab_margin_mm)
            ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="lower",
                      extent=extent,
                      aspect=vol.spacing_zyx[vaxis] / vol.spacing_zyx[haxis])
            if mask.any() and not mask.all():
                ax.contour(mask, levels=[0.5], colors="#39ff5a", linewidths=1.1,
                           origin="lower", extent=extent)
            visible = ((path[:, daxis] >= path[:, daxis].min() - 1)
                       & (path[:, daxis] <= path[:, daxis].max() + 1))
            ax.plot(path[visible, haxis], path[visible, vaxis], "-",
                    color="#00d9ff", lw=1.6)
            ax.plot(ostium[haxis], ostium[vaxis], "o", ms=7, mfc="none",
                    mec="#ff3030", mew=1.8)
            ax.plot(seed[haxis], seed[vaxis], marker="D", ms=5,
                    mfc="#ffe400", mec="black", mew=0.6)
            ax.annotate("", xy=(seed[haxis], seed[vaxis]),
                        xytext=(ostium[haxis], ostium[vaxis]),
                        arrowprops=dict(arrowstyle="->", color="#ff9d00", lw=1.1))
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if column == 0:
                xyz = meta["ostium_xyz_mm"]
                label = (f"{meta['instance_id']}   r={meta['radius_mm']:.1f} mm\n"
                         f"ostium=({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm")
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center",
                              labelpad=8)

    fig.suptitle(
        f"{result['case_id']} branch review — independently centered rows; IDs are case-local\n"
        "red circle: ostium   yellow diamond: 5 mm seed   cyan: trace   green: aorta mask",
        fontsize=13, y=0.998)
    # Reserve a fixed physical height for the two-line title, including small
    # review sheets produced by more selective candidate methods.
    fig.subplots_adjust(left=0.20, right=0.995, top=1 - min(0.30, 0.95 / (2.8 * rows)), bottom=0.01,
                        hspace=0.10, wspace=0.04)
    fig.savefig(out_png, dpi=140, facecolor="white")
    plt.close(fig)
    return out_png


def overlay(dbg, result, out_png):
    """Write the detailed review sheet and a smaller whole-aorta companion."""
    review_sheet(dbg, result, out_png)
    path = Path(out_png)
    overview_path = path.with_name(path.stem + "_overview" + path.suffix)
    overview(dbg, result, overview_path)
    return out_png
