#!/usr/bin/env python3
"""Local three-plane CT viewer for checking predicted aortic branches."""

from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from aortic_branch_prototype import open_nifti


PALETTE = (
    (255, 95, 105),
    (255, 184, 77),
    (106, 214, 176),
    (111, 180, 255),
    (193, 140, 255),
    (255, 126, 202),
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Aortic Branch Verification</title>
  <style>
    :root { color-scheme: dark; --bg:#071014; --panel:#101b21; --edge:#25353d; --text:#eaf1f4; --muted:#91a6af; --cyan:#35d8d0; --orange:#ffb84d; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 20% 0,#16303a 0,#071014 38rem); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }
    header { display:flex; gap:18px; align-items:center; justify-content:space-between; padding:18px 22px 10px; }
    h1 { margin:0; font-size:22px; letter-spacing:-.02em; }
    header p { margin:3px 0 0; color:var(--muted); }
    .badge { border:1px solid #3a505a; border-radius:999px; padding:7px 12px; color:#b8c9d0; white-space:nowrap; }
    main { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:12px; padding:10px 12px 22px; }
    .workspace { min-width:0; }
    .views { display:grid; grid-template-columns:minmax(0,1fr); gap:16px; scroll-snap-type:y proximity; }
    .view { min-height:calc(100vh - 24px); display:flex; flex-direction:column; scroll-snap-align:start; }
    .view,.panel { background:rgba(16,27,33,.94); border:1px solid var(--edge); border-radius:13px; overflow:hidden; box-shadow:0 12px 34px #0005; }
    .view-head { display:flex; justify-content:space-between; padding:9px 11px; border-bottom:1px solid var(--edge); color:#cadae0; }
    .image-wrap { position:relative; flex:1; min-height:min(72vh,760px); display:flex; align-items:center; justify-content:center; background:#020404; overflow:hidden; cursor:crosshair; }
    .image-wrap img { display:block; width:100%; height:100%; object-fit:contain; image-rendering:auto; user-select:none; }
    .coordinates { position:absolute; z-index:2; min-width:132px; padding:6px 8px; border:1px solid #53717c; border-radius:6px; background:#071014e8; color:#eaf1f4; font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; pointer-events:none; box-shadow:0 5px 16px #0008; }
    .slice-controls { padding:9px 11px 11px; }
    input[type=range] { width:100%; accent-color:var(--cyan); }
    .controls { display:flex; flex-wrap:wrap; align-items:center; gap:12px; margin-bottom:10px; padding:10px 12px; background:rgba(16,27,33,.92); border:1px solid var(--edge); border-radius:12px; }
    .controls label { display:flex; align-items:center; gap:6px; color:#bbccd2; }
    input[type=number] { width:76px; padding:5px 7px; color:var(--text); background:#071014; border:1px solid #34464f; border-radius:6px; }
    button { border:1px solid #3b5059; background:#17262d; color:var(--text); border-radius:7px; padding:6px 9px; cursor:pointer; }
    button:hover { background:#21343c; border-color:#55717d; }
    .mask-mode { display:flex; align-items:center; gap:4px; }
    .mask-mode > span { margin-right:2px; color:#bbccd2; }
    .mask-mode button.active { border-color:var(--cyan); background:#12363a; color:#8ff4ef; }
    aside { min-width:0; }
    .panel { padding:14px; max-height:calc(100vh - 92px); display:flex; flex-direction:column; }
    .summary { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:10px; }
    .stat { background:#091318; border:1px solid #213139; border-radius:9px; padding:10px; }
    .stat strong { display:block; font-size:20px; color:#fff; }
    .stat span { color:var(--muted); font-size:12px; }
    .warning { padding:9px 10px; margin-bottom:10px; border-left:3px solid var(--orange); background:#2a2113; color:#f1d8a8; font-size:12px; }
    .branch-list { overflow:auto; padding-right:3px; }
    .branch { display:grid; grid-template-columns:29px 1fr auto; gap:8px; align-items:center; width:100%; text-align:left; margin-bottom:6px; padding:9px; border:1px solid #24363e; background:#0b171c; border-radius:9px; }
    .branch:hover,.branch.active { border-color:var(--cyan); background:#102329; }
    .branch.accepted { box-shadow:inset 3px 0 #35d899; }
    .branch.rejected { opacity:.48; box-shadow:inset 3px 0 #ff5f69; }
    .dot { width:20px; height:20px; border-radius:50%; display:grid; place-items:center; color:#061014; font-weight:800; font-size:11px; }
    .branch small { display:block; color:var(--muted); }
    .prob { font-variant-numeric:tabular-nums; color:#dbe9ee; }
    .legend { display:flex; gap:14px; color:var(--muted); font-size:12px; margin:9px 2px 0; }
    .legend i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    .review { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; margin:0 0 10px; }
    .review button { padding:8px 4px; }
    .review .yes { border-color:#2d8f6d; }.review .no { border-color:#a34d55; }
    .export { width:100%; margin-bottom:10px; }
    @media (max-width:1100px) { main { grid-template-columns:1fr; } .panel { max-height:none; } }
    @media (max-width:780px) { .views { grid-template-columns:1fr; } .image-wrap { height:56vh; } }
  </style>
</head>
<body>
  <header><div><h1>Aortic branch verification</h1><p id="subtitle">Loading scan…</p></div><div class="badge">Local viewer · no data uploaded</div></header>
  <main>
    <section class="workspace">
      <div class="controls">
        <label>Window center <input id="center" type="number" value="250"></label>
        <label>Width <input id="width" type="number" value="700" min="1"></label>
        <button data-window="250,700">CTA</button><button data-window="50,400">Soft tissue</button><button data-window="500,1800">Bone</button>
        <div class="mask-mode" id="maskMode"><span>Aorta mask</span><button data-mask="1" class="active">Show</button><button data-mask="0">Hide</button></div>
        <label><input id="branches" type="checkbox" checked> Predictions</label>
        <label>Minimum score <input id="cutoff" type="range" min="0" max="1" step="0.01" value="0.42"><span id="cutoffValue">0.42</span></label>
      </div>
      <div class="views" id="views"></div>
      <div class="legend"><span><i style="background:#35d8d0"></i>Aorta-mask outline</span><span><i style="background:#ffb84d"></i>Selected branch</span><span>Move the mouse over a scan for voxel coordinates; click to reposition the crosshair.</span></div>
    </section>
    <aside><div class="panel">
      <div class="summary"><div class="stat"><strong id="count">–</strong><span>predicted</span></div><div class="stat"><strong id="accepted">0</strong><span>confirmed</span></div><div class="stat"><strong id="unreviewed">–</strong><span>unreviewed</span></div></div>
      <div class="warning">Scores marked <b>geometric_prototype</b> are ranking scores, not trained probabilities. Every proposed branch still needs visual confirmation.</div>
      <div class="review"><button class="yes" id="accept">✓ True branch</button><button class="no" id="reject">× False positive</button><button id="clearReview">Unreview</button></div>
      <button class="export" id="exportReview">Download verification JSON</button>
      <div class="branch-list" id="branchList"></div>
    </div></aside>
  </main>
<script>
const planes = [
  {id:'axial', axis:'z', dimension:2},
  {id:'coronal', axis:'y', dimension:1},
  {id:'sagittal', axis:'x', dimension:0},
];
let meta, selected=1, state={x:0,y:0,z:0}, renderTimer, reviews={}, showMask=true;
const $ = id => document.getElementById(id);
function currentIndex(p){ return state[p.axis]; }
function setIndex(p,value){ state[p.axis]=Math.max(0,Math.min(meta.shape[p.dimension]-1,Math.round(value))); }
function viewHTML(p){ const max=meta.shape[p.dimension]-1; return `<article class="view"><div class="view-head"><b>${p.id[0].toUpperCase()+p.id.slice(1)}</b><span id="${p.id}Label"></span></div><div class="image-wrap"><img id="${p.id}Image" data-plane="${p.id}" draggable="false"><span class="coordinates" id="${p.id}Coordinates" hidden></span></div><div class="slice-controls"><input id="${p.id}Slider" type="range" min="0" max="${max}" value="${currentIndex(p)}"></div></article>`; }
function imageURL(p){ const q=new URLSearchParams({plane:p.id,index:currentIndex(p),center:$('center').value,width:$('width').value,mask:showMask?1:0,branches:$('branches').checked?1:0,cutoff:$('cutoff').value,selected,x:state.x,y:state.y,z:state.z,t:Date.now()}); return '/api/slice.png?'+q; }
function scheduleRender(){ clearTimeout(renderTimer); renderTimer=setTimeout(render,35); }
function render(){ planes.forEach(p=>{ $(p.id+'Label').textContent=`${p.axis.toUpperCase()} ${currentIndex(p)} / ${meta.shape[p.dimension]-1}`; $(p.id+'Slider').value=currentIndex(p); $(p.id+'Image').src=imageURL(p); }); highlight(); }
function reviewKey(){ return `aortic-review:${meta.subject}`; }
function saveReviews(){ localStorage.setItem(reviewKey(),JSON.stringify(reviews)); updateReviewUI(); }
function updateReviewUI(){ const values=Object.values(reviews),accepted=values.filter(v=>v==='accepted').length,reviewed=values.filter(v=>v==='accepted'||v==='rejected').length;$('accepted').textContent=accepted;$('unreviewed').textContent=meta.branches.length-reviewed;document.querySelectorAll('.branch').forEach((el,i)=>{const value=reviews[meta.branches[i].id];el.classList.toggle('accepted',value==='accepted');el.classList.toggle('rejected',value==='rejected')}); }
function highlight(){ document.querySelectorAll('.branch').forEach((el,i)=>el.classList.toggle('active',i+1===selected)); updateReviewUI(); }
function jumpTo(branch,index){ const p=branch.ostium_voxel; state={x:Math.round(p[0]),y:Math.round(p[1]),z:Math.round(p[2])}; selected=index; render(); }
function branchCards(){ $('branchList').innerHTML=meta.branches.map((b,i)=>{const color=`rgb(${b.color.join(',')})`; return `<button class="branch" data-index="${i}"><span class="dot" style="background:${color}">${b.id}</span><span><b>Branch ${b.id}</b><small>radius ${b.radius_mm.toFixed(2)} mm · ${b.scoring_method}</small></span><span class="prob">${b.probability.toFixed(3)}</span></button>`}).join(''); document.querySelectorAll('.branch').forEach(el=>el.onclick=()=>jumpTo(meta.branches[+el.dataset.index],+el.dataset.index+1)); }
function pointerVoxel(p,image,event){
  const rect=image.getBoundingClientRect();
  const dimensions=p.id==='axial'?[meta.shape[0],meta.shape[1]]:p.id==='coronal'?[meta.shape[0],meta.shape[2]]:[meta.shape[1],meta.shape[2]];
  const scale=Math.min(rect.width/dimensions[0],rect.height/dimensions[1]);
  const renderedWidth=dimensions[0]*scale,renderedHeight=dimensions[1]*scale;
  const offsetX=(rect.width-renderedWidth)/2,offsetY=(rect.height-renderedHeight)/2;
  const u=(event.clientX-rect.left-offsetX)/renderedWidth,v=(event.clientY-rect.top-offsetY)/renderedHeight;
  if(u<0||u>1||v<0||v>1)return null;
  const point={x:state.x,y:state.y,z:state.z};
  if(p.id==='axial'){point.x=Math.round(u*(meta.shape[0]-1));point.y=Math.round((1-v)*(meta.shape[1]-1));}
  else if(p.id==='coronal'){point.x=Math.round(u*(meta.shape[0]-1));point.z=Math.round((1-v)*(meta.shape[2]-1));}
  else{point.y=Math.round(u*(meta.shape[1]-1));point.z=Math.round((1-v)*(meta.shape[2]-1));}
  return {point,rect};
}
function updateCoordinates(p,image,event){
  const badge=$(p.id+'Coordinates'),sample=pointerVoxel(p,image,event);
  if(!sample){badge.hidden=true;return;}
  badge.hidden=false;
  badge.textContent=`X ${sample.point.x} · Y ${sample.point.y} · Z ${sample.point.z}`;
  const localX=event.clientX-sample.rect.left,localY=event.clientY-sample.rect.top;
  badge.style.left=Math.max(8,Math.min(localX+14,sample.rect.width-badge.offsetWidth-8))+'px';
  badge.style.top=Math.max(8,Math.min(localY+14,sample.rect.height-badge.offsetHeight-8))+'px';
}
function bind(){
  planes.forEach(p=>{ const slider=$(p.id+'Slider'), image=$(p.id+'Image'),badge=$(p.id+'Coordinates'); slider.oninput=()=>{setIndex(p,+slider.value);scheduleRender()}; image.parentElement.onwheel=e=>{e.preventDefault();setIndex(p,currentIndex(p)+(e.deltaY>0?1:-1));scheduleRender()}; image.onmousemove=e=>updateCoordinates(p,image,e);image.onmouseleave=()=>{badge.hidden=true};image.onclick=e=>{const sample=pointerVoxel(p,image,e);if(!sample)return;state=sample.point;scheduleRender()}; });
  ['center','width','branches','cutoff'].forEach(id=>$(id).oninput=()=>{if(id==='cutoff')$('cutoffValue').textContent=(+$('cutoff').value).toFixed(2);scheduleRender()});
  document.querySelectorAll('[data-mask]').forEach(button=>button.onclick=()=>{showMask=button.dataset.mask==='1';document.querySelectorAll('[data-mask]').forEach(candidate=>candidate.classList.toggle('active',candidate===button));render()});
  document.querySelectorAll('[data-window]').forEach(b=>b.onclick=()=>{const [c,w]=b.dataset.window.split(',');$('center').value=c;$('width').value=w;render()});
  $('accept').onclick=()=>{if(selected){reviews[selected]='accepted';saveReviews()}};
  $('reject').onclick=()=>{if(selected){reviews[selected]='rejected';saveReviews()}};
  $('clearReview').onclick=()=>{if(selected){delete reviews[selected];saveReviews()}};
  $('exportReview').onclick=()=>{const records=meta.branches.map(b=>({id:b.id,status:reviews[b.id]||'unreviewed',probability:b.probability,radius_mm:b.radius_mm,ostium_voxel:b.ostium_voxel}));const payload={subject:meta.subject,confirmed_branch_count:records.filter(r=>r.status==='accepted').length,rejected_count:records.filter(r=>r.status==='rejected').length,unreviewed_count:records.filter(r=>r.status==='unreviewed').length,reviews:records};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=meta.subject+'_branch_verification.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
}
fetch('/api/meta').then(r=>r.json()).then(m=>{meta=m;try{reviews=JSON.parse(localStorage.getItem(reviewKey())||'{}')}catch(e){reviews={}}state={x:Math.round(m.initial_voxel[0]),y:Math.round(m.initial_voxel[1]),z:Math.round(m.initial_voxel[2])};$('subtitle').textContent=`${m.subject} · ${m.shape.join(' × ')} voxels · ${m.spacing.map(v=>v.toFixed(3)).join(' × ')} mm · ${m.raw_candidate_count} raw proposals`;$('count').textContent=m.branches.length;$('cutoff').value=m.probability_threshold;$('cutoffValue').textContent=m.probability_threshold.toFixed(2);$('views').innerHTML=planes.map(viewHTML).join('');branchCards();bind();render();}).catch(e=>{$('subtitle').textContent='Could not load scan: '+e});
</script>
</body></html>"""


class BranchViewer:
    def __init__(self, ct_path: Path, mask_path: Path, prediction_path: Path):
        with open_nifti(ct_path) as image:
            self.ct = np.asarray(image.dataobj, dtype=np.float32)
            self.affine = np.asarray(image.affine, dtype=np.float64)
            self.spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        with open_nifti(mask_path) as image:
            if image.shape[:3] != self.ct.shape:
                raise ValueError("CT and mask shapes differ")
            if not np.allclose(image.affine, self.affine, atol=1.0e-3):
                raise ValueError("CT and mask affines differ")
            self.mask = np.asarray(image.dataobj) > 0

        self.prediction_path = prediction_path
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        self.subject = str(payload.get("name", ct_path.stem))
        self.coordinate_system = str(payload.get("coordinate_system", "LPS_mm")).upper()
        self.raw_candidate_count = int(payload.get("metadata", {}).get("raw_candidate_count", 0))
        self.probability_threshold = float(
            payload.get("metadata", {}).get("config", {}).get("probability_threshold", 0.0)
        )
        inverse = np.linalg.inv(self.affine)
        self.branches = []
        for index, record in enumerate(payload.get("branches", [])):
            branch = dict(record)
            branch["color"] = PALETTE[index % len(PALETTE)]
            branch["ostium_voxel"] = self._world_to_voxel(record["ostium"], inverse).tolist()
            branch["seed_voxel"] = self._world_to_voxel(record["seed"], inverse).tolist()
            branch["centerline_voxel"] = [
                self._world_to_voxel(point, inverse).tolist() for point in record.get("centerline", [])
            ]
            self.branches.append(branch)

        if self.branches:
            self.initial_voxel = self.branches[0]["ostium_voxel"]
        else:
            self.initial_voxel = list(ndi.center_of_mass(self.mask))

    def _world_to_voxel(self, point, inverse: np.ndarray) -> np.ndarray:
        world = np.asarray(point, dtype=np.float64).copy()
        if self.coordinate_system.startswith("LPS"):
            world[:2] *= -1.0
        homogeneous = inverse @ np.r_[world, 1.0]
        return homogeneous[:3]

    def metadata(self) -> dict:
        public_branches = []
        for branch in self.branches:
            public_branches.append(
                {
                    "id": int(branch["id"]),
                    "probability": float(branch["probability"]),
                    "radius_mm": float(branch["radius_mm"]),
                    "scoring_method": str(branch["scoring_method"]),
                    "ostium_voxel": branch["ostium_voxel"],
                    "color": branch["color"],
                }
            )
        return {
            "subject": self.subject,
            "shape": list(self.ct.shape),
            "spacing": list(self.spacing),
            "coordinate_system": self.coordinate_system,
            "raw_candidate_count": self.raw_candidate_count,
            "probability_threshold": self.probability_threshold,
            "initial_voxel": self.initial_voxel,
            "branches": public_branches,
        }

    @staticmethod
    def _oriented(array: np.ndarray, plane: str, index: int) -> np.ndarray:
        if plane == "axial":
            return np.flipud(array[:, :, index].T)
        if plane == "coronal":
            return np.flipud(array[:, index, :].T)
        return np.flipud(array[index, :, :].T)

    def _project(self, point: np.ndarray, plane: str) -> tuple[float, float, float]:
        x, y, z = point
        if plane == "axial":
            return float(x), float(self.ct.shape[1] - 1 - y), float(z)
        if plane == "coronal":
            return float(x), float(self.ct.shape[2] - 1 - z), float(y)
        return float(y), float(self.ct.shape[2] - 1 - z), float(x)

    @lru_cache(maxsize=72)
    def render_slice(
        self,
        plane: str,
        index: int,
        center: float,
        width: float,
        show_mask: bool,
        show_branches: bool,
        cutoff: float,
        selected: int,
        cross_x: int,
        cross_y: int,
        cross_z: int,
    ) -> bytes:
        dimension = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
        index = int(np.clip(index, 0, self.ct.shape[dimension] - 1))
        pixels = self._oriented(self.ct, plane, index)
        low = center - max(width, 1.0) / 2.0
        gray = np.clip((pixels - low) * (255.0 / max(width, 1.0)), 0.0, 255.0).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)

        if show_mask:
            mask_slice = self._oriented(self.mask, plane, index)
            boundary = mask_slice & ~ndi.binary_erosion(mask_slice)
            rgb[mask_slice] = (0.72 * rgb[mask_slice] + 0.28 * np.array([25, 178, 174])).astype(np.uint8)
            rgb[boundary] = np.array([53, 216, 208], dtype=np.uint8)

        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        if show_branches:
            slab = max(1.25, 2.0 / self.spacing[dimension])
            for branch in self.branches:
                if float(branch["probability"]) < cutoff:
                    continue
                color = PALETTE[(int(branch["id"]) - 1) % len(PALETTE)]
                if int(branch["id"]) == selected:
                    color = (255, 184, 77)
                path = np.asarray(branch["centerline_voxel"], dtype=np.float64)
                projected = [self._project(point, plane) for point in path]
                for first, second in zip(projected, projected[1:]):
                    if abs(first[2] - index) <= slab and abs(second[2] - index) <= slab:
                        draw.line((first[0], first[1], second[0], second[1]), fill=color, width=2)
                ox, oy, depth = self._project(np.asarray(branch["ostium_voxel"]), plane)
                if abs(depth - index) <= slab:
                    radius = 6 if int(branch["id"]) == selected else 4
                    draw.ellipse((ox - radius, oy - radius, ox + radius, oy + radius), outline=color, width=2)
                    draw.text((ox + radius + 2, oy - radius - 2), str(branch["id"]), fill=color)

        crosshair = np.asarray([cross_x, cross_y, cross_z], dtype=np.float64)
        cx, cy, _ = self._project(crosshair, plane)
        draw.line((cx, 0, cx, image.height - 1), fill=(255, 196, 61), width=1)
        draw.line((0, cy, image.width - 1, cy), fill=(255, 196, 61), width=1)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False)
        return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    viewer: BranchViewer

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        if parsed.path == "/api/meta":
            return self._send(json.dumps(self.viewer.metadata()).encode("utf-8"), "application/json")
        if parsed.path == "/api/slice.png":
            try:
                query = parse_qs(parsed.query)
                plane = query.get("plane", ["axial"])[0]
                if plane not in {"axial", "coronal", "sagittal"}:
                    raise ValueError("Unknown plane")
                body = self.viewer.render_slice(
                    plane,
                    int(query.get("index", [0])[0]),
                    round(float(query.get("center", [250])[0]), 2),
                    round(max(1.0, float(query.get("width", [700])[0])), 2),
                    query.get("mask", ["1"])[0] == "1",
                    query.get("branches", ["1"])[0] == "1",
                    round(float(query.get("cutoff", [0])[0]), 3),
                    int(query.get("selected", [0])[0]),
                    int(query.get("x", [0])[0]),
                    int(query.get("y", [0])[0]),
                    int(query.get("z", [0])[0]),
                )
                return self._send(body, "image/png")
            except (KeyError, TypeError, ValueError) as error:
                return self._send(json.dumps({"error": str(error)}).encode(), "application/json", 400)
        return self._send(b"Not found", "text/plain", 404)

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        if self.path.startswith("/api/slice.png"):
            return
        super().log_message(format, *args)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--aorta-mask", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    viewer = BranchViewer(arguments.ct, arguments.aorta_mask, arguments.predictions)
    Handler.viewer = viewer
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    url = f"http://{arguments.host}:{arguments.port}"
    print(f"Loaded {viewer.subject}: {viewer.ct.shape}, {len(viewer.branches)} branch candidates")
    print(f"Open {url}  (press Ctrl+C to stop)")
    if arguments.open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
