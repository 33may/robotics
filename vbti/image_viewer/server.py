"""FastAPI server for COLMAP Frame Curator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io

from vbti.image_viewer.backend import (
    probe_video,
    init_work_dir,
    extract_frames,
    generate_histogram,
    export_accepted,
)

app = FastAPI(title="COLMAP Frame Curator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-user session state
session = {
    "work_dir": None,
    "frames": [],  # list of FrameData dicts (with absolute paths)
}


# ── Request/Response models ───────────────────────────────────────

class ProbeRequest(BaseModel):
    path: str

class ExtractRequest(BaseModel):
    video_path: str
    k: int = 2

class ExportRequest(BaseModel):
    accepted_indices: list[int]
    output_dir: str
    fmt: str = "jpg"
    quality: int = 90


# ── Helpers ───────────────────────────────────────────────────────

def frames_to_api(frames: list[dict]) -> list[dict]:
    """Convert absolute paths to API URLs for the frontend."""
    result = []
    for fd in frames:
        result.append({
            "frame_idx": fd["frame_idx"],
            "score": fd["score"],
            "timestamp_sec": fd["timestamp_sec"],
            "thumb_url": f"/api/media/thumbs/{Path(fd['thumb_path']).name}",
            "full_url": f"/api/media/full/{Path(fd['full_path']).name}",
            "preselected": fd.get("preselected", True),
        })
    return result


# ── API endpoints ─────────────────────────────────────────────────

@app.post("/api/probe")
def api_probe(req: ProbeRequest):
    try:
        info = probe_video(req.path)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/extract")
def api_extract(req: ExtractRequest):
    try:
        work_dir = init_work_dir(req.video_path)
        session["work_dir"] = work_dir

        frames = extract_frames(req.video_path, work_dir, req.k)
        session["frames"] = frames

        return {
            "frames": frames_to_api(frames),
            "count": len(frames),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/media/{subpath:path}")
def api_media(subpath: str):
    if not session["work_dir"]:
        raise HTTPException(status_code=404, detail="No active session")

    file_path = Path(session["work_dir"]) / subpath
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure the path is within work_dir
    try:
        file_path.resolve().relative_to(Path(session["work_dir"]).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(file_path)


@app.get("/api/histogram")
def api_histogram():
    if not session["frames"]:
        raise HTTPException(status_code=404, detail="No frames extracted")

    fig = generate_histogram(session["frames"])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#1f2937", edgecolor="none")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


@app.post("/api/export")
def api_export(req: ExportRequest):
    if not session["frames"]:
        raise HTTPException(status_code=400, detail="No frames to export")

    accepted = {idx: True for idx in req.accepted_indices}

    try:
        path, count = export_accepted(
            session["frames"], accepted, req.output_dir, req.fmt, req.quality
        )
        return {"path": path, "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Serve React build in production ──────────────────────────────

frontend_dist = Path(__file__).parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
