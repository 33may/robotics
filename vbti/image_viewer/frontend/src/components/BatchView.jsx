import React, { useState, useEffect, useRef, useCallback } from "react";

export default function BatchView({
  batches,
  currentBatchIdx,
  onChangeBatch,
  accepted,
  onToggleFrame,
}) {
  const batch = batches[currentBatchIdx];
  const [viewingIdx, setViewingIdx] = useState(0); // index within batch.frames
  const stripRef = useRef(null);
  const activeBatchRef = useRef(null);

  // Reset viewing index when batch changes
  useEffect(() => {
    setViewingIdx(0);
  }, [currentBatchIdx]);

  // Auto-scroll batch strip
  useEffect(() => {
    if (activeBatchRef.current) {
      activeBatchRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
    }
  }, [currentBatchIdx]);

  // Keyboard: Shift+Arrow = batch, Arrow = frame, Space = toggle
  const handleKeyDown = useCallback(
    (e) => {
      if (!batch) return;

      if (e.shiftKey && e.key === "ArrowLeft") {
        e.preventDefault();
        if (currentBatchIdx > 0) onChangeBatch(currentBatchIdx - 1);
      } else if (e.shiftKey && e.key === "ArrowRight") {
        e.preventDefault();
        if (currentBatchIdx < batches.length - 1) onChangeBatch(currentBatchIdx + 1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setViewingIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setViewingIdx((prev) => Math.min(batch.frames.length - 1, prev + 1));
      } else if (e.key === " ") {
        e.preventDefault();
        const frame = batch.frames[viewingIdx];
        if (frame) onToggleFrame(frame.frame_idx);
      }
    },
    [batch, currentBatchIdx, batches.length, onChangeBatch, viewingIdx, onToggleFrame]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (!batch) return null;

  const viewingFrame = batch.frames[viewingIdx] || batch.frames[0];
  const selectedInBatch = batch.frames.filter((f) => accepted.has(f.frame_idx)).length;

  return (
    <div className="batch-view">
      {/* Blue navigation strip */}
      <div className="batch-nav">
        <button
          className="nav-arrow"
          onClick={() => onChangeBatch(Math.max(0, currentBatchIdx - 1))}
          disabled={currentBatchIdx === 0}
        >
          &#9664;
        </button>

        <div className="batch-strip" ref={stripRef}>
          {batches.map((b, i) => {
            const batchSelected = b.frames.some((f) => accepted.has(f.frame_idx));
            return (
              <button
                key={b.second}
                ref={i === currentBatchIdx ? activeBatchRef : null}
                className={`batch-chip ${i === currentBatchIdx ? "active" : ""} ${!batchSelected ? "empty" : ""}`}
                onClick={() => onChangeBatch(i)}
              >
                <img src={b.frames[0].thumb_url} alt="" />
                <span>{b.second}s</span>
              </button>
            );
          })}
        </div>

        <button
          className="nav-arrow"
          onClick={() => onChangeBatch(Math.min(batches.length - 1, currentBatchIdx + 1))}
          disabled={currentBatchIdx === batches.length - 1}
        >
          &#9654;
        </button>
      </div>

      {/* Main preview */}
      {viewingFrame && (
        <div className="main-preview">
          <img src={viewingFrame.full_url} alt={`Frame ${viewingFrame.frame_idx}`} />
          <div className="preview-controls">
            <label className="frame-checkbox">
              <input
                type="checkbox"
                checked={accepted.has(viewingFrame.frame_idx)}
                onChange={() => onToggleFrame(viewingFrame.frame_idx)}
              />
              <span>Selected</span>
            </label>
            <span className="preview-meta">
              #{viewingFrame.frame_idx} &nbsp;|&nbsp; Score: {viewingFrame.score.toFixed(1)}
              &nbsp;|&nbsp; {viewingFrame.timestamp_sec.toFixed(1)}s
              &nbsp;|&nbsp; Frame {viewingIdx + 1}/{batch.frames.length}
            </span>
            <span className="preview-shortcuts">
              &#8592;&#8594; frames &nbsp; Shift+&#8592;&#8594; batches &nbsp; Space toggle
            </span>
          </div>
        </div>
      )}

      {/* Batch frame thumbnails */}
      <div className="batch-frames">
        <div className="batch-frames-label">
          Batch {batch.second}s &mdash; {selectedInBatch}/{batch.frames.length} selected
        </div>
        <div className="batch-frames-row">
          {batch.frames.map((f, i) => (
            <div
              key={f.frame_idx}
              className={`batch-thumb ${i === viewingIdx ? "viewing" : ""} ${accepted.has(f.frame_idx) ? "accepted" : "rejected"}`}
              onClick={() => setViewingIdx(i)}
            >
              <img src={f.thumb_url} alt={`Frame ${f.frame_idx}`} />
              <div className="thumb-footer">
                <label className="thumb-check" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={accepted.has(f.frame_idx)}
                    onChange={() => onToggleFrame(f.frame_idx)}
                  />
                </label>
                <span className="thumb-score">{f.score.toFixed(0)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
