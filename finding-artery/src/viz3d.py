"""Interactive 3D CT context and accepted branch predictions."""
import base64
import json
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes


def _to_lps(vol, points_zyx):
    """Convert cropped NumPy coordinates to SimpleITK physical LPS mm."""
    points = np.asarray(points_zyx, float).reshape(-1, 3)
    full_xyz = points[:, ::-1] + np.asarray(vol.offset)[::-1]
    scaled = full_xyz * np.asarray(vol.ref.GetSpacing())
    direction = np.asarray(vol.ref.GetDirection()).reshape(3, 3)
    return np.asarray(vol.ref.GetOrigin()) + scaled @ direction.T


def _mesh(mask, vol, step_size=2):
    """Return an LPS surface for a binary mask, cropped before marching cubes."""
    indices = np.argwhere(mask)
    if len(indices) < 2:
        return None
    low = np.maximum(indices.min(axis=0) - 1, 0)
    high = np.minimum(indices.max(axis=0) + 2, mask.shape)
    block = mask[tuple(slice(a, b) for a, b in zip(low, high))]
    block = np.pad(block, 1)
    try:
        vertices, faces, _, _ = marching_cubes(
            block.astype(np.uint8), 0.5, step_size=step_size, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    vertices += low - 1
    return _to_lps(vol, vertices), faces


def _slice_plane(vol, fixed_axis, fixed_index, target=110):
    """Create one downsampled CT plane in LPS coordinates."""
    axes = [axis for axis in range(3) if axis != fixed_axis]
    strides = [max(1, int(np.ceil(vol.ct.shape[axis] / target))) for axis in axes]
    coordinates = [np.arange(0, vol.ct.shape[axis], stride)
                   for axis, stride in zip(axes, strides)]
    grid = np.meshgrid(*coordinates, indexing="ij")
    points = np.empty(grid[0].shape + (3,), float)
    points[..., fixed_axis] = fixed_index
    points[..., axes[0]] = grid[0]
    points[..., axes[1]] = grid[1]
    colors = np.take(np.take(vol.ct, coordinates[0], axis=axes[0]),
                     coordinates[1], axis=axes[1])
    colors = np.take(colors, int(fixed_index), axis=fixed_axis)
    if colors.shape != grid[0].shape:
        colors = colors.T
    lps = _to_lps(vol, points.reshape(-1, 3)).reshape(points.shape)
    return lps, colors


def _viewer_payload(vol, debug, result, max_voxels=None):
    """Compact int16 CT and prediction geometry for browser-side slice controls."""
    stride = (max(1, int(np.ceil((vol.ct.size / max_voxels) ** (1 / 3))))
              if max_voxels else 1)
    coordinates = []
    for size in vol.ct.shape:
        values = np.arange(0, size, stride, dtype=int)
        if values[-1] != size - 1:
            values = np.r_[values, size - 1]
        coordinates.append(values)
    sampled = vol.ct[np.ix_(*coordinates)]
    sampled = np.clip(np.rint(sampled), -1024, 3071).astype("<i2", copy=False)
    aorta_voxels = np.argwhere(vol.aorta)
    centre = (np.rint(aorta_voxels.mean(axis=0)).astype(int)
              if len(aorta_voxels) else np.asarray(vol.ct.shape) // 2)
    initial = [int(np.argmin(np.abs(values - centre[axis])))
               for axis, values in enumerate(coordinates)]
    branches = []
    for daughter, metadata in zip(debug["traced"], result["daughters"]):
        branches.append(dict(
            label=metadata["instance_id"],
            radius_mm=metadata.get("radius_mm"),
            ostium=np.asarray(daughter["ostium_vox"], float).tolist(),
            seed=np.asarray(daughter["seed_vox"], float).tolist(),
            path=np.asarray(daughter["path"], float).tolist()))
    return dict(
        case_id=result.get("case_id", "Study"),
        shape=list(sampled.shape), coordinates=[v.tolist() for v in coordinates],
        initial=initial,
        ct_base64=base64.b64encode(sampled.tobytes()).decode("ascii"),
        spacing_zyx=list(vol.spacing_zyx), offset=list(vol.offset),
        spacing_xyz=list(vol.ref.GetSpacing()), origin_lps=list(vol.ref.GetOrigin()),
        direction=np.asarray(vol.ref.GetDirection()).reshape(3, 3).tolist(),
        branches=branches)


def _viewer_document(plot_html, payload):
    """Embed the local review layout and data into one portable HTML file."""
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    # Replace data first: Plotly's embedded library must not be searched/replaced.
    return template.replace("__VIEWER_DATA__", data).replace("__PLOT_HTML__", plot_html)


def write_html(debug, result, output_html):
    """Write a self-contained 3D view with linked CT slice controls."""
    import plotly.graph_objects as go

    vol = debug["vol"]
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    traces = []

    centre = np.rint(np.argwhere(vol.aorta).mean(axis=0)).astype(int)
    finite = vol.ct[np.isfinite(vol.ct)]
    cmin = float(max(-200, np.percentile(finite, 2)))
    cmax = float(min(700, np.percentile(finite, 99)))
    plane_names = ("Axial CT", "Coronal CT", "Sagittal CT")
    for number, (axis, name) in enumerate(zip((0, 1, 2), plane_names)):
        lps, values = _slice_plane(vol, axis, centre[axis])
        traces.append(go.Surface(
            x=lps[..., 0], y=lps[..., 1], z=lps[..., 2], surfacecolor=values,
            colorscale="Gray", cmin=cmin, cmax=cmax, opacity=0.62,
            showscale=False,
            colorbar=dict(title="HU", len=0.55, thickness=14) if number == 0 else None,
            name=name, showlegend=True,
            hovertemplate=f"{name}<br>HU=%{{surfacecolor:.0f}}<extra></extra>"))

    parent_mesh = _mesh(vol.aorta, vol)
    if parent_mesh is not None:
        vertices, faces = parent_mesh
        traces.append(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#35c46a", opacity=0.28, name="Supplied aorta mask",
            hoverinfo="name", flatshading=True))

    shown_lumen = False
    for daughter in debug["traced"]:
        source = daughter.get("intensity_pass")
        labels = debug.get("labels_by_pass", {}).get(source)
        if labels is None or daughter.get("candidate_id") is None:
            continue
        candidate_mesh = _mesh(labels == daughter["candidate_id"], vol, step_size=1)
        if candidate_mesh is None:
            continue
        vertices, faces = candidate_mesh
        traces.append(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#42d4f4", opacity=0.24,
            name="Accepted grown lumen", legendgroup="grown",
            showlegend=not shown_lumen, hoverinfo="skip", flatshading=True))
        shown_lumen = True

    for daughter, metadata in zip(debug["traced"], result["daughters"]):
        path = _to_lps(vol, daughter["path"])
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
            line=dict(color="#00bde3", width=7),
            name=f'{metadata["instance_id"]} centerline',
            legendgroup=metadata["instance_id"],
            hovertemplate=f'{metadata["instance_id"]}<br>centerline<extra></extra>'))

    daughters = result["daughters"]
    if daughters:
        ostia = np.asarray([d["ostium_xyz_mm"] for d in daughters])
        seeds = np.asarray([d["seed_xyz_mm"] for d in daughters])
        labels = [d["instance_id"] for d in daughters]
        radii = [d["radius_mm"] for d in daughters]
        custom = [[label, float(radius)] for label, radius in zip(labels, radii)]
        traces.append(go.Scatter3d(
            x=ostia[:, 0], y=ostia[:, 1], z=ostia[:, 2], mode="markers+text",
            marker=dict(size=6, color="#ff3333", line=dict(color="white", width=1)),
            text=labels, textposition="top center", name="Ostia",
            customdata=custom,
            hovertemplate="%{customdata[0]}<br>ostium<br>radius=%{customdata[1]:.2f} mm"
                          "<br>LPS=(%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>"))
        traces.append(go.Scatter3d(
            x=seeds[:, 0], y=seeds[:, 1], z=seeds[:, 2], mode="markers",
            marker=dict(size=6, color="#ffe433", symbol="diamond",
                        line=dict(color="#222", width=1)),
            name="5 mm seeds", customdata=np.asarray(labels)[:, None],
            hovertemplate="%{customdata[0]}<br>5 mm seed"
                          "<br>LPS=(%{x:.1f}, %{y:.1f}, %{z:.1f}) mm<extra></extra>"))

    figure = go.Figure(traces)
    figure.update_layout(
        template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0e1723", showlegend=False, uirevision="review",
        scene=dict(aspectmode="data", xaxis_title="X (LPS mm)",
                   yaxis_title="Y (LPS mm)", zaxis_title="Z (LPS mm)",
                   bgcolor="#0e1723"))
    payload = _viewer_payload(vol, debug, result)
    payload["cmin"], payload["cmax"] = cmin, cmax
    plot_html = figure.to_html(include_plotlyjs=True, full_html=False,
                               default_width="100%", default_height="100%",
                               div_id="branchseed-3d",
                               config={"displaylogo": False, "scrollZoom": True})
    output_html.write_text(_viewer_document(plot_html, payload), encoding="utf-8")
    return output_html
