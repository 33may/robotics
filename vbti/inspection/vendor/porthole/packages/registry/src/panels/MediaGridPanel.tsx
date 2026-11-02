/**
 * A grid of live images and videos.
 *
 * REGISTRY COMPONENT. Copy it into your app and edit it; see
 * docs/component-model.md.
 *
 * ## Why one component and not an ImageGrid plus a VideoGrid
 *
 * On the bus there is no such thing as a video. A camera is a topic carrying
 * `$img` payloads at 30 Hz; a still is a topic carrying one `$img` payload that
 * never updates. Same payload type, same decode path, same canvas — the only
 * difference is how often a message arrives, and rate is not a type.
 *
 * So the split that would justify two components does not exist, and inventing
 * it would mean two copies of the layout solver, the fit rules and the labels.
 * There is a second kind of source — a file the backend never streamed, served
 * over HTTP — and that one really is different, because the browser plays it
 * with its own element. It is a variant of `MediaGridSource`, not a variant of
 * the panel.
 *
 * ## What this panel is for
 *
 * Any fixed set of simultaneous views: a multi-camera rig, a viewpoint
 * candidate sheet, before/after captures, a contact sheet of the last N frames.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { RefObject } from 'react';
import { useImageTopicCanvas } from '@porthole/framework';
import type { PanelProps } from '@porthole/framework';

/** A bus-fed cell: still or moving, the panel cannot tell and does not care. */
export interface MediaGridTopicSource {
  readonly topic: string;
  readonly label?: string;
}

/** A file-fed cell, served over HTTP by the same server that serves the UI. */
export interface MediaGridUrlSource {
  readonly url: string;
  readonly kind: 'image' | 'video';
  readonly label?: string;
}

export type MediaGridSource = MediaGridTopicSource | MediaGridUrlSource;

export interface MediaGridPanelConfig extends Record<string, unknown> {
  /** One cell per entry, laid out in reading order. */
  sources: readonly MediaGridSource[];
  /** Fixed column count, or `'auto'` to fit the panel's current shape. */
  columns: number | 'auto';
  /** `contain` shows the whole frame; `cover` fills the cell and crops. */
  fit: 'contain' | 'cover';
  /** Pixels between cells. */
  gap: number;
  showLabels: boolean;
  /**
   * Frame aspect (width / height) assumed before the first frame arrives.
   *
   * Only affects layout, and only until a frame lands. 16/9 is right for
   * essentially every camera; set it for a portrait or square sensor so the
   * first painted layout is not visibly wrong.
   */
  assumedAspect: number;
}

export const mediaGridPanelDefaultConfig: MediaGridPanelConfig = {
  sources: [{ topic: 'camera/wrist', label: 'wrist' }],
  columns: 'auto',
  fit: 'contain',
  gap: 2,
  showLabels: true,
  assumedAspect: 16 / 9,
};

function isUrlSource(source: MediaGridSource): source is MediaGridUrlSource {
  return 'url' in source;
}

function sourceKey(source: MediaGridSource, index: number): string {
  return `${index}:${isUrlSource(source) ? source.url : source.topic}`;
}

function sourceLabel(source: MediaGridSource): string {
  if (source.label !== undefined) return source.label;
  return isUrlSource(source) ? source.url.split('/').pop() ?? source.url : source.topic;
}

/**
 * Column count that shows the most pixels per cell in a given panel shape.
 *
 * Exported and pure so it can be tested without a DOM, and so an app that wants
 * a different rule (fill width, prefer square) can compare against this one.
 *
 * Under `contain`, a cell renders at height `min(cellHeight, cellWidth /
 * aspect)`; the rendered area is that height squared times the aspect. Maximise
 * it over every column count, which is a search over at most `count` integers —
 * cheaper and more obviously correct than any closed form.
 */
export function bestGridColumns(
  count: number,
  panelWidth: number,
  panelHeight: number,
  frameAspect: number,
  gap: number,
): number {
  if (count <= 1 || panelWidth <= 0 || panelHeight <= 0 || frameAspect <= 0) return 1;

  let best = 1;
  let bestArea = -1;
  for (let columns = 1; columns <= count; columns += 1) {
    const rows = Math.ceil(count / columns);
    const cellWidth = (panelWidth - gap * (columns - 1)) / columns;
    const cellHeight = (panelHeight - gap * (rows - 1)) / rows;
    if (cellWidth <= 0 || cellHeight <= 0) continue;

    const renderedHeight = Math.min(cellHeight, cellWidth / frameAspect);
    const area = renderedHeight * renderedHeight * frameAspect;
    if (area > bestArea) {
      bestArea = area;
      best = columns;
    }
  }
  return best;
}

/**
 * The grid's natural width/height ratio — what to size a window to.
 *
 * A panel cannot resize the window it lives in, and should not try. Call this
 * from the app that opens the window:
 *
 *     const aspect = mediaGridAspect(4, 2, 16 / 9);   // 3.55
 *     open_window(url, width=1280, height=1280 / aspect)
 */
export function mediaGridAspect(
  count: number,
  columns: number,
  frameAspect: number,
): number {
  const rows = Math.max(1, Math.ceil(count / Math.max(1, columns)));
  return (Math.max(1, columns) * frameAspect) / rows;
}

/** Panel size, tracked so `columns: 'auto'` can respond to it. */
function usePanelSize(): [RefObject<HTMLDivElement | null>, { width: number; height: number }] {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      const box = entry?.contentRect;
      if (!box) return;
      setSize((previous) =>
        Math.abs(previous.width - box.width) < 1 && Math.abs(previous.height - box.height) < 1
          ? previous
          : { width: box.width, height: box.height },
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}

interface MediaGridContentProps {
  readonly fit: 'contain' | 'cover';
  readonly onAspect: (aspect: number) => void;
}

/**
 * A bus-fed cell.
 *
 * Split from the URL cell rather than branching inside one component because
 * `useImageTopicCanvas` is a hook: a component that calls it only for some
 * sources either violates the rules of hooks or subscribes to a fake topic.
 * Two small components are the honest shape.
 */
function MediaGridTopicCell({
  source,
  fit,
  onAspect,
}: MediaGridContentProps & { readonly source: MediaGridTopicSource }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reported = useRef(0);

  // onFrame fires per frame. Compare before propagating, or a 30 Hz stream
  // becomes 30 parent re-renders a second, each running the layout solver to
  // produce the answer it already had.
  useImageTopicCanvas(source.topic, canvasRef, {
    onFrame: (width, height) => {
      if (height <= 0) return;
      const aspect = width / height;
      if (Math.abs(aspect - reported.current) < 0.001) return;
      reported.current = aspect;
      onAspect(aspect);
    },
  });

  return <canvas ref={canvasRef} data-fit={fit} />;
}

/** A file-fed cell: the browser decodes and, for video, plays it. */
function MediaGridUrlCell({
  source,
  fit,
  onAspect,
}: MediaGridContentProps & { readonly source: MediaGridUrlSource }) {
  if (source.kind === 'video') {
    return (
      <video
        src={source.url}
        data-fit={fit}
        autoPlay
        loop
        muted
        playsInline
        onLoadedMetadata={(event) => {
          const element = event.currentTarget;
          if (element.videoHeight > 0) onAspect(element.videoWidth / element.videoHeight);
        }}
      />
    );
  }
  return (
    <img
      src={source.url}
      data-fit={fit}
      alt=""
      onLoad={(event) => {
        const element = event.currentTarget;
        if (element.naturalHeight > 0) onAspect(element.naturalWidth / element.naturalHeight);
      }}
    />
  );
}

export function MediaGridPanel({ config }: PanelProps<MediaGridPanelConfig>) {
  const [gridRef, size] = usePanelSize();
  const [frameAspect, setFrameAspect] = useState(config.assumedAspect);
  const [focused, setFocused] = useState<number | null>(null);

  const sources = config.sources ?? [];
  const visible = focused === null ? sources : sources.slice(focused, focused + 1);

  const columns = useMemo(() => {
    if (config.columns !== 'auto') return Math.max(1, Math.min(config.columns, visible.length || 1));
    return bestGridColumns(visible.length, size.width, size.height, frameAspect, config.gap);
  }, [config.columns, config.gap, visible.length, size.width, size.height, frameAspect]);

  return (
    <div
      ref={gridRef}
      className="porthole-media-grid"
      style={{
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gridTemplateRows: `repeat(${Math.ceil((visible.length || 1) / columns)}, 1fr)`,
        gap: `${config.gap}px`,
      }}
    >
      {visible.map((source, index) => {
        // `index` walks `visible`, which is the full list unless one cell is
        // focused — and then focusing again is always "unfocus", so the real
        // index is only ever needed in the unfocused case.
        const realIndex = focused === null ? index : focused;
        return (
          <div
            key={sourceKey(source, realIndex)}
            className="porthole-media-cell"
            data-focused={focused !== null}
            onDoubleClick={() => setFocused(focused === null ? index : null)}
            title={sourceLabel(source)}
          >
            {isUrlSource(source) ? (
              <MediaGridUrlCell source={source} fit={config.fit} onAspect={setFrameAspect} />
            ) : (
              <MediaGridTopicCell source={source} fit={config.fit} onAspect={setFrameAspect} />
            )}
            {config.showLabels ? (
              <span className="porthole-media-label">{sourceLabel(source)}</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
