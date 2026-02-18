import React from "react";

export default function GridOverview({ batches, accepted, onSelectBatch }) {
  return (
    <div className="grid-overview">
      <div className="grid-cards">
        {batches.map((batch, idx) => {
          const selectedCount = batch.frames.filter((f) =>
            accepted.has(f.frame_idx)
          ).length;
          const isEmpty = selectedCount === 0;

          return (
            <div
              key={batch.second}
              className={`grid-card ${isEmpty ? "empty" : ""}`}
              onClick={() => onSelectBatch(idx)}
            >
              <div className="grid-card-img">
                <img src={batch.frames[0].thumb_url} alt="" />
                {isEmpty && <div className="grid-card-warning">NO FRAMES</div>}
              </div>
              <div className="grid-card-info">
                <span className="grid-card-time">{batch.second}s</span>
                <span className={`grid-card-count ${isEmpty ? "empty" : ""}`}>
                  {selectedCount}/{batch.frames.length}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
