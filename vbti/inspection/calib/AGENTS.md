# calib — registration & calibration artifacts

## Purpose
Scripts and data that pin frames to reality. Run rarely, versioned
forever: the numbers here are load-bearing for everything downstream.

## Files
None yet. Planned:
- Table probe fit script — 5 freedrive touches → base→table transform
  (probed once; robot is welded to the table, transform is permanent).
- TCP definitions — ruler measurement + 2-orientation check.
- Future hand-eye outputs for the UR5e rig.

## Contracts & decisions
- Data files are welcome here and are committed/versioned — calibration
  artifacts are source, not build products.
- Results feed `cell/cell.yaml` frames; scripts here write those
  numbers, they do not get imported at runtime.

## Does NOT belong here
- Runtime perception (perception/), the world model itself (cell/),
  anything imported by the live pipeline.
