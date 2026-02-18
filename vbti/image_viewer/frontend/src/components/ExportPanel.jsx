import React, { useState } from "react";

export default function ExportPanel({ acceptedIndices, suggestedDir }) {
  const [outputDir, setOutputDir] = useState(suggestedDir);
  const [fmt, setFmt] = useState("jpg");
  const [quality, setQuality] = useState(90);
  const [exporting, setExporting] = useState(false);
  const [result, setResult] = useState(null);

  async function handleExport() {
    setExporting(true);
    setResult(null);
    try {
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          accepted_indices: acceptedIndices,
          output_dir: outputDir,
          fmt,
          quality,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || "Export failed");
        return;
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      alert("Connection error: " + e.message);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="card">
      <div className="export-form">
        <div>
          <label>Output directory</label>
          <input
            type="text"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
          />
        </div>

        <div>
          <label>Format</label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                name="fmt"
                value="jpg"
                checked={fmt === "jpg"}
                onChange={() => setFmt("jpg")}
              />
              JPEG
            </label>
            <label>
              <input
                type="radio"
                name="fmt"
                value="png"
                checked={fmt === "png"}
                onChange={() => setFmt("png")}
              />
              PNG
            </label>
          </div>
        </div>

        {fmt === "jpg" && (
          <div>
            <label>JPEG Quality: {quality}</label>
            <input
              type="range"
              min={70}
              max={100}
              step={1}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
            />
          </div>
        )}

        <div style={{ marginTop: 24 }}>
          <button
            className="btn btn-primary btn-lg"
            onClick={handleExport}
            disabled={exporting || !outputDir.trim()}
          >
            {exporting ? "Exporting..." : `Export ${acceptedIndices.length} Frames`}
          </button>
        </div>

        {result && (
          <div className="export-result">
            Exported <strong>{result.count}</strong> frames to <code>{result.path}</code>
          </div>
        )}
      </div>
    </div>
  );
}
