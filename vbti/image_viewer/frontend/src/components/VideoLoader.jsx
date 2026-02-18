import React, { useState } from "react";

export default function VideoLoader({ onExtracted }) {
  const [path, setPath] = useState("");
  const [videoInfo, setVideoInfo] = useState(null);
  const [k, setK] = useState(2);
  const [loading, setLoading] = useState(false);
  const [probing, setProbing] = useState(false);

  const estimate = videoInfo ? Math.round(videoInfo.duration_sec * k) : 0;

  async function handleLoad() {
    if (!path.trim()) return;
    setProbing(true);
    try {
      const res = await fetch("/api/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path.trim() }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Failed to load video");
        return;
      }
      setVideoInfo(await res.json());
    } catch (e) {
      alert("Connection error: " + e.message);
    } finally {
      setProbing(false);
    }
  }

  async function handleExtract() {
    setLoading(true);
    try {
      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: path.trim(), k }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Extraction failed");
        return;
      }
      const data = await res.json();
      onExtracted(data.frames, path.trim());
    } catch (e) {
      alert("Connection error: " + e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card">
      <div className="loader-grid">
        <div>
          <label>Video path (local)</label>
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/video.mov"
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
          />
          <div style={{ marginTop: 12 }}>
            <button
              className="btn btn-secondary"
              onClick={handleLoad}
              disabled={probing || !path.trim()}
            >
              {probing ? "Loading..." : "Load Video"}
            </button>
          </div>
        </div>

        <div className="video-info">
          {videoInfo ? (
            <>
              <div><span>Resolution:</span> {videoInfo.width} x {videoInfo.height}</div>
              <div><span>FPS:</span> {videoInfo.fps.toFixed(1)}</div>
              <div><span>Duration:</span> {videoInfo.duration_sec}s</div>
              <div><span>Total frames:</span> {videoInfo.total_frames.toLocaleString()}</div>
            </>
          ) : (
            <div className="empty-state">No video loaded</div>
          )}
        </div>
      </div>

      {videoInfo && (
        <div className="controls-row">
          <div className="slider-group">
            <label>Frames per second (k): {k}</label>
            <input
              type="range"
              min={1}
              max={5}
              step={1}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </div>
          <div className="estimate">~{estimate} frames</div>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleExtract}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 16, height: 16, display: "inline-block", verticalAlign: "middle", marginRight: 8 }} />
                Extracting...
              </>
            ) : (
              "Extract Frames"
            )}
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <div>Scoring and extracting frames... This may take a minute.</div>
        </div>
      )}
    </div>
  );
}
