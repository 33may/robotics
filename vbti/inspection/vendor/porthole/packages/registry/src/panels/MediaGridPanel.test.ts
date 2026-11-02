/**
 * The media grid's layout solver.
 *
 * This is the only non-trivial pure logic in the panel, and its failure mode is
 * subtle: a wrong column count still renders every cell, still looks like a
 * grid, and just wastes half the panel on letterboxing. Nobody files a bug for
 * that, so it gets a test.
 */

import { describe, expect, it } from 'vitest';

import { bestGridColumns, mediaGridAspect } from './MediaGridPanel.js';

const WIDE = 16 / 9;

describe('bestGridColumns', () => {
  it('puts everything in one row when the panel is much wider than tall', () => {
    expect(bestGridColumns(4, 1920, 300, WIDE, 2)).toBe(4);
  });

  it('stacks into one column when the panel is tall and narrow', () => {
    expect(bestGridColumns(4, 400, 1600, WIDE, 2)).toBe(1);
  });

  it('picks a square-ish shape for a square panel', () => {
    expect(bestGridColumns(4, 1000, 1000, WIDE, 2)).toBe(2);
  });

  it('never returns more columns than there are cells', () => {
    for (const width of [200, 1000, 4000]) {
      expect(bestGridColumns(3, width, 600, WIDE, 2)).toBeLessThanOrEqual(3);
    }
  });

  it('handles a single cell and an empty grid without dividing by zero', () => {
    expect(bestGridColumns(1, 800, 600, WIDE, 2)).toBe(1);
    expect(bestGridColumns(0, 800, 600, WIDE, 2)).toBe(1);
  });

  it('falls back to one column before the panel has been measured', () => {
    // ResizeObserver has not fired yet on the first render; a solver that
    // divided by a zero width would produce NaN and an unrenderable grid.
    expect(bestGridColumns(6, 0, 0, WIDE, 2)).toBe(1);
  });

  it('accounts for portrait sources', () => {
    // Six portrait frames in a wide panel should spread out further than six
    // landscape ones, because each needs less width.
    const portrait = bestGridColumns(6, 1600, 900, 9 / 16, 2);
    const landscape = bestGridColumns(6, 1600, 900, 16 / 9, 2);
    expect(portrait).toBeGreaterThan(landscape);
  });

  it('is not fooled by a large gap into an absurd column count', () => {
    expect(bestGridColumns(8, 400, 400, WIDE, 40)).toBeLessThanOrEqual(4);
  });
});

describe('mediaGridAspect', () => {
  it('reports the grid shape a window should be sized to', () => {
    expect(mediaGridAspect(4, 2, WIDE)).toBeCloseTo((2 * 16) / 9 / 2);
    expect(mediaGridAspect(4, 4, WIDE)).toBeCloseTo((4 * 16) / 9);
  });

  it('counts a partial row as a full one', () => {
    // 5 cells in 2 columns is 3 rows, not 2.5 — the last row is half empty but
    // it still occupies height.
    expect(mediaGridAspect(5, 2, 1)).toBeCloseTo(2 / 3);
  });
});
