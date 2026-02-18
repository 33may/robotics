import React, { useMemo, useState } from "react";

const W = 1000;
const H = 340;
const PAD = { top: 20, right: 30, bottom: 40, left: 60 };
const PLOT_W = W - PAD.left - PAD.right;
const PLOT_H = H - PAD.top - PAD.bottom;

export default function ThresholdView({
  frames,
  batches,
  accepted,
  threshold,
  onThresholdChange,
}) {
  const [hovered, setHovered] = useState(null);

  const { minScore, maxScore, scores } = useMemo(() => {
    const scores = frames.map((f) => f.score);
    return {
      minScore: Math.floor(Math.min(...scores)),
      maxScore: Math.ceil(Math.max(...scores)),
      scores,
    };
  }, [frames]);

  const range = maxScore - minScore || 1;

  const xScale = (i) => PAD.left + (i / Math.max(frames.length - 1, 1)) * PLOT_W;
  const yScale = (score) => PAD.top + PLOT_H - ((score - minScore) / range) * PLOT_H;

  // Line path
  const linePath = frames
    .map((f, i) => `${i === 0 ? "M" : "L"} ${xScale(i)} ${yScale(f.score)}`)
    .join(" ");

  // Threshold line Y
  const threshY = yScale(threshold);

  // Stats
  const aboveCount = frames.filter((f) => f.score >= threshold).length;
  const emptyBatches = batches.filter(
    (b) => !b.frames.some((f) => f.score >= threshold)
  );

  // Y-axis ticks
  const tickCount = 6;
  const yTicks = Array.from({ length: tickCount }, (_, i) => {
    const val = minScore + (range * i) / (tickCount - 1);
    return Math.round(val);
  });

  return (
    <div className="threshold-view">
      <div className="threshold-chart-container">
        <svg viewBox={`0 0 ${W} ${H}`} className="threshold-chart">
          {/* Grid lines */}
          {yTicks.map((tick) => (
            <line
              key={tick}
              x1={PAD.left}
              y1={yScale(tick)}
              x2={W - PAD.right}
              y2={yScale(tick)}
              stroke="#334155"
              strokeWidth="0.5"
            />
          ))}

          {/* Y axis labels */}
          {yTicks.map((tick) => (
            <text
              key={`label-${tick}`}
              x={PAD.left - 8}
              y={yScale(tick) + 4}
              textAnchor="end"
              fill="#94a3b8"
              fontSize="11"
            >
              {tick}
            </text>
          ))}

          {/* X axis label */}
          <text
            x={PAD.left + PLOT_W / 2}
            y={H - 4}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
          >
            Frames (chronological)
          </text>

          {/* Y axis label */}
          <text
            x={14}
            y={PAD.top + PLOT_H / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}
          >
            Sharpness
          </text>

          {/* Score line */}
          <polyline
            points={linePath}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="1.5"
            opacity="0.6"
          />

          {/* Data points */}
          {frames.map((f, i) => {
            const above = f.score >= threshold;
            return (
              <circle
                key={f.frame_idx}
                cx={xScale(i)}
                cy={yScale(f.score)}
                r={hovered === i ? 5 : 3}
                fill={above ? "#3b82f6" : "#64748b"}
                opacity={above ? 1 : 0.4}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer" }}
              />
            );
          })}

          {/* Threshold line */}
          <line
            x1={PAD.left}
            y1={threshY}
            x2={W - PAD.right}
            y2={threshY}
            stroke="#ef4444"
            strokeWidth="2"
            strokeDasharray="8 4"
          />
          <text
            x={W - PAD.right + 4}
            y={threshY + 4}
            fill="#ef4444"
            fontSize="12"
            fontWeight="600"
          >
            {threshold.toFixed(0)}
          </text>

          {/* Below-threshold shading */}
          <rect
            x={PAD.left}
            y={threshY}
            width={PLOT_W}
            height={PAD.top + PLOT_H - threshY}
            fill="#ef4444"
            opacity="0.05"
          />

          {/* Tooltip */}
          {hovered !== null && (
            <g>
              <rect
                x={Math.min(xScale(hovered), W - 160)}
                y={yScale(frames[hovered].score) - 48}
                width={150}
                height={38}
                rx={4}
                fill="#1e293b"
                stroke="#334155"
              />
              <text
                x={Math.min(xScale(hovered), W - 160) + 8}
                y={yScale(frames[hovered].score) - 30}
                fill="#e2e8f0"
                fontSize="11"
              >
                #{frames[hovered].frame_idx} | {frames[hovered].timestamp_sec.toFixed(1)}s
              </text>
              <text
                x={Math.min(xScale(hovered), W - 160) + 8}
                y={yScale(frames[hovered].score) - 16}
                fill="#94a3b8"
                fontSize="11"
              >
                Score: {frames[hovered].score.toFixed(1)}
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Threshold slider */}
      <div className="threshold-controls">
        <label>
          Threshold: <strong>{threshold.toFixed(0)}</strong>
        </label>
        <input
          type="range"
          min={minScore}
          max={maxScore}
          step={1}
          value={threshold}
          onChange={(e) => onThresholdChange(Number(e.target.value))}
        />
        <div className="threshold-stats">
          <span>
            <strong>{aboveCount}</strong> / {frames.length} frames above threshold (
            {((aboveCount / frames.length) * 100).toFixed(0)}%)
          </span>
          {emptyBatches.length > 0 && (
            <span className="warning">
              {emptyBatches.length} {emptyBatches.length === 1 ? "batch" : "batches"} with 0 frames
              {emptyBatches.length <= 5 && (
                <> &mdash; at {emptyBatches.map((b) => `${b.second}s`).join(", ")}</>
              )}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
