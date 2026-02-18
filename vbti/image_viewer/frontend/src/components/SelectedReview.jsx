import React, { useState, useEffect, useCallback, useMemo } from "react";

export default function SelectedReview({ frames, accepted, onToggleFrame }) {
  const [currentIdx, setCurrentIdx] = useState(0);

  // Only selected frames, chronological
  const selectedFrames = useMemo(
    () => frames.filter((f) => accepted.has(f.frame_idx)),
    [frames, accepted]
  );

  // Clamp index if frames got deselected
  useEffect(() => {
    if (currentIdx >= selectedFrames.length && selectedFrames.length > 0) {
      setCurrentIdx(selectedFrames.length - 1);
    }
  }, [selectedFrames.length, currentIdx]);

  const frame = selectedFrames[currentIdx];

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCurrentIdx((prev) => Math.max(0, prev - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCurrentIdx((prev) => Math.min(selectedFrames.length - 1, prev + 1));
      } else if (e.key === " " || e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        if (frame) onToggleFrame(frame.frame_idx);
      }
    },
    [selectedFrames.length, frame, onToggleFrame]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  if (selectedFrames.length === 0) {
    return <div className="empty-state">No selected frames to review</div>;
  }

  if (!frame) return null;

  return (
    <div className="filmstrip-view">
      {/* Header with navigation */}
      <div className="filmstrip-header">
        <button
          className="btn btn-secondary"
          onClick={() => setCurrentIdx((prev) => Math.max(0, prev - 1))}
          disabled={currentIdx === 0}
        >
          &#9664; Prev
        </button>

        <span className="filmstrip-counter">
          <strong>{currentIdx + 1}</strong> / {selectedFrames.length}
        </span>

        <button
          className="btn btn-secondary"
          onClick={() =>
            setCurrentIdx((prev) => Math.min(selectedFrames.length - 1, prev + 1))
          }
          disabled={currentIdx === selectedFrames.length - 1}
        >
          Next &#9654;
        </button>

        <button
          className="btn btn-danger"
          onClick={() => onToggleFrame(frame.frame_idx)}
        >
          Deselect
        </button>
      </div>

      {/* Large preview */}
      <div className="filmstrip-preview">
        <img src={frame.full_url} alt={`Frame ${frame.frame_idx}`} />
      </div>

      {/* Info bar */}
      <div className="filmstrip-info">
        <span>#{frame.frame_idx}</span>
        <span>Score: <strong>{frame.score.toFixed(1)}</strong></span>
        <span>Time: {frame.timestamp_sec.toFixed(1)}s</span>
        <span className="preview-shortcuts">
          &#8592;&#8594; navigate &nbsp; Space/Delete deselect
        </span>
      </div>
    </div>
  );
}
