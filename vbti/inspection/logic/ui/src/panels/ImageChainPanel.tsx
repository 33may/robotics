/**
 * The identity chain for the newest capture, as a 2x2 grid.
 *
 * Reads left-to-right, top-to-bottom, in the order the loop actually runs:
 *
 *   1 capture        what the camera saw
 *   2 prompt         the cloud so far, reprojected, and the box built from it
 *   3 mask           what the segmenter called object inside that box
 *   4 accepted       the points that entered the object cloud
 *
 * APP-OWNED PANEL, and it should stay that way: "prompt / mask / accepted" is
 * this pipeline's vocabulary, not a generic toolkit concept.
 *
 * The images are static files served over the same `/captures` mount as every
 * other capture image (../AGENTS.md §7) — the topic carries URLs, so the
 * browser fetches and decodes them off the main thread instead of competing
 * with the WebGL panels, and a panel opened mid-run shows the last chain from
 * the retained message immediately.
 */

import { useTopicPayload } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';

/** Stage key -> the caption shown under it. Order here IS the grid order. */
const STAGES = [
  ['rgb', 'capture', 'what the camera saw'],
  ['prompt', 'prompt', 'cloud so far, reprojected + box'],
  ['mask', 'mask', 'what the segmenter called object'],
  ['kept', 'accepted', 'points that entered the cloud'],
] as const;

export interface ChainPayload {
  readonly images?: Readonly<Record<string, string>>;
  readonly pose_id?: number;
  /** 'mask' when segmentation decided this view, 'depth' when it fell back. */
  readonly source?: 'mask' | 'depth';
  readonly score?: number | null;
  readonly mask_px?: number | null;
  readonly offered?: number;
  readonly kept?: number;
  readonly dropped?: number;
  readonly cloud?: number;
  readonly extent_mm?: readonly number[] | null;
}

export interface ImageChainPanelConfig extends Record<string, unknown> {
  topic: string;
}

export const imageChainPanelDefaultConfig: ImageChainPanelConfig = {
  topic: 'chain/latest',
};

export function ImageChainPanel({ config }: PanelProps<ImageChainPanelConfig>) {
  const chain = useTopicPayload<ChainPayload>(config.topic);

  if (!chain?.images) {
    return <div className="inspection-chain-empty">no capture yet</div>;
  }

  const { images } = chain;
  const fellBack = chain.source === 'depth';

  return (
    <div className="inspection-chain">
      <div className="inspection-chain-head">
        {chain.pose_id !== undefined ? <span>view {chain.pose_id}</span> : null}
        {/* The one thing worth seeing at a glance: did a model decide this
            view, or did it fall back to depth growth? */}
        <span className="inspection-chain-source" data-source={chain.source}>
          {fellBack ? 'depth fallback' : 'mask'}
          {chain.score != null ? ` ${chain.score.toFixed(2)}` : ''}
        </span>
        {chain.kept !== undefined ? <span>{chain.kept} pts kept</span> : null}
        {chain.dropped ? (
          <span className="inspection-chain-warn">{chain.dropped} gated</span>
        ) : null}
        {chain.cloud !== undefined ? <span>cloud {chain.cloud}</span> : null}
        {chain.extent_mm ? (
          <span>{chain.extent_mm.map((v) => v.toFixed(0)).join(' × ')} mm</span>
        ) : null}
      </div>

      <div className="inspection-chain-grid">
        {STAGES.map(([key, title, hint]) => (
          <figure key={key} className="inspection-chain-cell">
            {images[key] ? (
              <img className="inspection-chain-img" src={images[key]} alt={title} />
            ) : (
              // A stage is absent when it did not happen — a depth fallback
              // has no mask. Say so rather than showing a broken image.
              <div className="inspection-chain-missing">not produced</div>
            )}
            <figcaption className="inspection-chain-caption">
              <b>{title}</b>
              <span>{hint}</span>
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
