import React, { useState, useCallback } from "react";
import VideoLoader from "./components/VideoLoader";
import FrameReview from "./components/FrameReview";
import ExportPanel from "./components/ExportPanel";

const STEPS = ["Load Video", "Review & Curate", "Export"];

export default function App() {
  const [step, setStep] = useState(0);
  const [frames, setFrames] = useState([]);
  const [accepted, setAccepted] = useState(new Set());
  const [threshold, setThreshold] = useState(0);
  const [videoPath, setVideoPath] = useState("");

  const handleExtracted = useCallback((extractedFrames, path) => {
    setFrames(extractedFrames);
    setAccepted(
      new Set(extractedFrames.filter((f) => f.preselected).map((f) => f.frame_idx))
    );
    setThreshold(0);
    setVideoPath(path);
    setStep(1);
  }, []);

  const toggleFrame = useCallback((frameIdx) => {
    setAccepted((prev) => {
      const next = new Set(prev);
      if (next.has(frameIdx)) {
        next.delete(frameIdx);
      } else {
        next.add(frameIdx);
      }
      return next;
    });
  }, []);

  const applyThreshold = useCallback(
    (value) => {
      setThreshold(value);
      setAccepted(
        new Set(frames.filter((f) => f.score >= value).map((f) => f.frame_idx))
      );
    },
    [frames]
  );

  const suggestedDir = videoPath
    ? videoPath.replace(/\.[^.]+$/, "") + "_colmap"
    : "/tmp/colmap_export";

  const acceptedIndices = [...accepted].sort((a, b) => a - b);

  return (
    <div className="app">
      <div className="app-header">
        <h1>COLMAP Frame Curator</h1>
        <div className="step-nav">
          {STEPS.map((label, i) => (
            <button
              key={i}
              className={`${step === i ? "active" : ""} ${i < step ? "completed" : ""}`}
              onClick={() => i <= step && setStep(i)}
            >
              {i + 1}. {label}
            </button>
          ))}
        </div>
      </div>

      {step === 0 && <VideoLoader onExtracted={handleExtracted} />}

      {step === 1 && (
        <FrameReview
          frames={frames}
          accepted={accepted}
          threshold={threshold}
          onToggleFrame={toggleFrame}
          onApplyThreshold={applyThreshold}
          onAdvanceExport={() => setStep(2)}
        />
      )}

      {step === 2 && (
        <ExportPanel
          acceptedIndices={acceptedIndices}
          suggestedDir={suggestedDir}
        />
      )}
    </div>
  );
}
