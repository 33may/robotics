import React, { useState, useMemo } from "react";
import BatchView from "./BatchView";
import ThresholdView from "./ThresholdView";
import GridOverview from "./GridOverview";
import SelectedReview from "./SelectedReview";

function groupIntoBatches(frames) {
  const map = new Map();
  for (const f of frames) {
    const sec = Math.floor(f.timestamp_sec);
    if (!map.has(sec)) map.set(sec, []);
    map.get(sec).push(f);
  }
  return Array.from(map.entries())
    .sort(([a], [b]) => a - b)
    .map(([sec, batchFrames]) => ({ second: sec, frames: batchFrames }));
}

const VIEW_MODES = [
  { key: "batch", label: "Batch View" },
  { key: "selected", label: "Selected Review" },
  { key: "threshold", label: "Threshold" },
  { key: "grid", label: "Grid Overview" },
];

export default function FrameReview({
  frames,
  accepted,
  threshold,
  onToggleFrame,
  onApplyThreshold,
  onAdvanceExport,
}) {
  const [viewMode, setViewMode] = useState("batch");
  const [currentBatchIdx, setCurrentBatchIdx] = useState(0);

  const batches = useMemo(() => groupIntoBatches(frames), [frames]);

  const emptyBatchCount = batches.filter(
    (b) => !b.frames.some((f) => accepted.has(f.frame_idx))
  ).length;

  return (
    <div className="review-container">
      <div className="review-header">
        <div className="view-tabs">
          {VIEW_MODES.map(({ key, label }) => (
            <button
              key={key}
              className={viewMode === key ? "active" : ""}
              onClick={() => setViewMode(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="review-stats">
          <span className="stat">
            <strong>{accepted.size}</strong> / {frames.length} selected
          </span>
          {emptyBatchCount > 0 && (
            <span className="stat warning">
              {emptyBatchCount} empty {emptyBatchCount === 1 ? "batch" : "batches"}
            </span>
          )}
          <button className="btn btn-primary" onClick={onAdvanceExport}>
            Export {accepted.size} frames &rarr;
          </button>
        </div>
      </div>

      {viewMode === "batch" && (
        <BatchView
          batches={batches}
          currentBatchIdx={currentBatchIdx}
          onChangeBatch={setCurrentBatchIdx}
          accepted={accepted}
          onToggleFrame={onToggleFrame}
        />
      )}

      {viewMode === "selected" && (
        <SelectedReview
          frames={frames}
          accepted={accepted}
          onToggleFrame={onToggleFrame}
        />
      )}

      {viewMode === "threshold" && (
        <ThresholdView
          frames={frames}
          batches={batches}
          accepted={accepted}
          threshold={threshold}
          onThresholdChange={onApplyThreshold}
        />
      )}

      {viewMode === "grid" && (
        <GridOverview
          batches={batches}
          accepted={accepted}
          onSelectBatch={(idx) => {
            setCurrentBatchIdx(idx);
            setViewMode("batch");
          }}
        />
      )}
    </div>
  );
}
