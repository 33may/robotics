/**
 * Paint an image topic onto a canvas.
 *
 * This lives in the framework rather than in a panel because every panel that
 * shows imagery needs the same three non-obvious behaviours, and getting any of
 * them wrong is a slow leak rather than a visible bug:
 *
 * 1. **Drop, don't queue.** Decoding is async and frames arrive faster than they
 *    decode. Keeping only the newest pending frame means the canvas shows
 *    current reality; queueing means it falls progressively further behind and
 *    never recovers.
 * 2. **Copy the bytes before wrapping them in a Blob.** A msgpack payload view
 *    usually aliases a much larger receive buffer, and `new Blob([view])`
 *    captures the whole thing.
 * 3. **`createImageBitmap`, not an object URL.** Object URLs must be revoked or
 *    they leak; at 15 fps a forgotten revoke leaks tens of KB per frame.
 *
 * It is deliberately indifferent to whether the topic carries one still image or
 * a 30 Hz video stream. On the wire those are the same thing — a topic carrying
 * `$img` payloads — and the only difference is how often a message arrives.
 */

import { useEffect, useRef } from 'react';
import type { RefObject } from 'react';

import type { EncodedImage } from '../bus/types.js';
import { useTopicEffect } from './hooks.js';

export interface ImageTopicCanvasOptions {
  /**
   * Called after a frame is painted, with the frame's pixel dimensions.
   *
   * Called on every frame, so treat it as a hot path: compare before setting
   * state, or a 30 Hz stream becomes 30 re-renders a second.
   */
  readonly onFrame?: (width: number, height: number) => void;
}

/**
 * Subscribe to `topic` and paint each `$img` payload into `canvas`.
 *
 * The canvas is resized to the frame's natural pixel size; scale it with CSS
 * (`object-fit`-style layout belongs to the panel, not to the decoder).
 */
export function useImageTopicCanvas(
  topic: string,
  canvas: RefObject<HTMLCanvasElement | null>,
  options: ImageTopicCanvasOptions = {},
): void {
  const pending = useRef<EncodedImage | null>(null);
  const busy = useRef(false);
  const alive = useRef(true);
  const onFrame = useRef(options.onFrame);
  onFrame.current = options.onFrame;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      pending.current = null;
    };
  }, []);

  useTopicEffect<EncodedImage>(topic, (envelope) => {
    pending.current = envelope.payload;
    void drain();
  });

  async function drain(): Promise<void> {
    if (busy.current) return;
    busy.current = true;
    try {
      while (pending.current && alive.current) {
        const frame = pending.current;
        pending.current = null;
        await paint(frame);
      }
    } finally {
      busy.current = false;
    }
  }

  async function paint(frame: EncodedImage): Promise<void> {
    const target = canvas.current;
    if (!target || !frame?.bytes) return;

    const blob = new Blob([frame.bytes.slice()], { type: frame.mime });
    const bitmap = await createImageBitmap(blob);

    // The panel may have unmounted during the await; drawing into a detached
    // canvas is harmless but closing the bitmap is not optional.
    if (!alive.current) {
      bitmap.close();
      return;
    }

    if (target.width !== bitmap.width || target.height !== bitmap.height) {
      target.width = bitmap.width;
      target.height = bitmap.height;
    }
    target.getContext('2d')?.drawImage(bitmap, 0, 0);
    const { width, height } = bitmap;
    bitmap.close();
    onFrame.current?.(width, height);
  }
}
