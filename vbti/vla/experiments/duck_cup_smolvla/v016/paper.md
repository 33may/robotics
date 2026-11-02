# v016 — Adding Depth Information to SmolVLA

## Research question
Does adding D405 depth (especially gripper-mounted) to the SmolVLA observation pipeline improve manipulation performance on the duck-cup task, given the hard constraint that existing RGB-only datasets (v013/v014/v015) must remain usable in training?

## Background
See `notes.md` for the full design brief. Headline facts:
- All 4 cameras are Intel RealSense D405. Depth hardware is dormant — `cameras.py` only enables `rs.stream.color`, and the gripper currently routes through OpenCV (no depth path at all).
- D405 is purpose-built for close-range manipulation (7–50 cm, sub-mm precision) — physically the right sensor for gripper depth.
- Constraint: v013/v014/v015 datasets are RGB-only and must be co-trainable with new RGB-D data.

## Design space

### Decided (2026-04-29)
- **Q2 path: B1 (extra camera).** SmolVLA treats each camera as an independent ViT pass; adding `observation.images.gripper_depth` requires zero source edits. exp01 confirmed B3 (channel concat) is dead and B2 (side-branch) is overkill for a first pilot.
- **Scope: gripper depth only.** One extra camera, not four. Minimizes token-budget impact (one extra ViT pass instead of four) and tests the hypothesis at the most contact-critical viewpoint first.
- **Q2-B2 (gripper-tailored side-branch with cross-attention into the action expert) deferred to v017** if v016 shows the depth signal carries value.

### Open (informed by upcoming experiments)
- **Q1** — handling old RGB-only datasets (backfill estimated depth / estimate-everywhere / dual-path). exp02 directly informs by characterizing the real-vs-estimated distribution gap.
- **Q3** — depth representation on disk (grayscale vs colorized, fixed-clip vs per-frame-norm, uint16 vs uint8). exp03 once real frames are in hand.

## Methodology
Sequential experiments, each answering one open question from `notes.md`. Findings flow into `results.md`; decisions and surprises into `flow.md`. Once enough evidence accumulates, this document gains Methods / Results / Discussion sections per the standard paper structure.

## Results
*Pending — see `results.md` as experiments complete.*

## Discussion
*Deferred until results accumulate.*

## Conclusion
*Deferred.*
