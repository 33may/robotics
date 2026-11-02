# Intel RealSense Depth Camera D405 — Product Page + D400-Series Datasheet (rev 337029-017)

**Product page**: https://www.intelrealsense.com/depth-camera-d405/ (DNS-unreachable from this sandbox; substituted https://www.realsenseai.com/products/stereo-depth-camera-d405/, current RealSense-spun-off site, functionally identical content)
**Datasheet PDF**: https://www.realsenseai.com/download/21345/?tmstv=1787826626 — "Intel® RealSense™ Depth Camera D410, D415, D430, D450, D401 Product Family Datasheet" (doc 337029-017, 159 pp., covers the whole D400 family incl. D405/D401 module — D405-specific tables extracted below by exact table number)
**Branch**: Scoring a single view: metrics and local optimality
**Read**: 2026-08-27 (full PDF converted with `pdftotext -layout` and grepped/read directly, not summarized secondhand)

## Core content

This is a manufacturer spec sheet, not a study — it bounds the sensor's operating envelope and geometry; it does **not** contain an incidence-angle-vs-error characterization (grep for "incidence"/"grazing"/"surface angle" over the full 159 pp. returns zero hits). Two things in it are more consequential than the numbers the branch went in expecting:

**1. D405 is a passive visible-light stereo camera, not active/structured-light.** Table 2-2 (Depth Camera Product SKU Descriptions, p.21) shows D405's "IR Projector" columns (Standard and Wide) both **X** — no projector at all, "N/A" in Table 3-49. Its left/right imager filter type (Table 3-15, D401 module, p.41) is an **IR-cut filter**, i.e. the stereo pair sees visible light only. Footnote 4 to the depth-quality table (Table 4-14, p.86) states this explicitly: *"KPIs for D405, a passive camera, are measured using a textured target and typical ambient room light (~250 Lux). All other models are active and measured using a texture-less (white) target with default laser power (150mW) and auto exposure enabled."* D405 depends on **scene texture + ambient light** for stereo correspondence; every other D4xx camera depends on its own IR dot projector and works fine on texture-less surfaces because it paints texture on.

**2. D405's color image and one depth imager are the same sensor.** Table 3-15 (D401 L/R imager) and Table 3-20 (D401 color sensor) are both OmniVision OV9782, 1.93 mm focal length, f/2.0, IR-cut filter, identical FOV (H84°/V58°/D92°) — the same part twice. Table 2-2 footnote 4: *"Left depth sensor provides RGB data to color ISP."* RGB and depth are natively pixel-registered on D405 (no separate offset color sensor to extrinsically calibrate against depth), unlike D435/D455 which use a physically distinct RGB sensor.

## Concrete results / numbers (all D401/D405-specific, exact table cited)

**Range / standoff (Table 4-11, Min-Z Depth, p.84 — resolution-dependent, not a single "7 cm" number):**

| Depth resolution | Min-Z, D401/D405 |
|---|---|
| 1280×720 (full HD) | **100 mm** |
| 848×480 | 70 mm |
| 640×360 | 55 mm |
| 480×270 | 45 mm |
| 424×240 | 40 mm |
| 640×480 | not supported (—) |

The marketed "7 cm" minimum (product page, and QVIRO/DigiKey summaries) is only true at 848×480 or lower — at full 1280×720 (needed for max image-space resolution) the datasheet's own Min-Z is **10 cm, not 7 cm**. This directly bears on the branch's "hard minimum shell radius vs. sensor near-field claim" conflict flagged in the task: the conflict is resolution-dependent, and gets worse (min standoff grows) if the pipeline wants full-res frames.

Top-end range: feature bullet list (p.16, "RealSense™ Depth Camera D405 Features") states **"Range 7 cm to 1 m (varies with lighting conditions)"** — wider than the "ideal 7–50 cm" product-page claim, but explicitly conditional on ambient light/texture since there's no projector to guarantee it.

**Depth accuracy (Table 4-14, Depth Quality Specification, D401/D405 column, "≤0.5 m and 80% ROI, HD Resolution," p.85–86):**

| Metric | D401/D405 |
|---|---|
| Z-accuracy (Absolute Error) | ± 2% |
| Fill rate | ≥ 99.5% |
| RMS Error (Spatial Noise) | ≤ 1% |
| Temporal Noise | ≤ 0.5% |
| Lifetime | 5 years |

This is a **band spec**, not a curve: "±2%" is the worst-case bound anywhere inside the ≤0.5 m / 80%-ROI qualification region, measured on a *textured* target under ~250 lux ambient — not a value tied to a specific point like "50 cm" and not decomposed by range or angle. Beyond 0.5 m, fill rate/accuracy are **unspecified** by this document. Separately, "<1 mm minimum object detection at 7 cm" (product page) is a *feature/gap-resolution* claim, a different metric from Z-accuracy — don't conflate the two numbers.

**Geometry / optics (Table 3-49, Depth Camera SKU Properties, p.67):**

| Property | D405 value |
|---|---|
| Depth module | D401 |
| Baseline | **18 mm** |
| Imager type | Wide, IR-cut filter (visible light) |
| Depth FOV (HD, 16:9) | H:84° / V:58° / D:92° |
| IR projector | N/A |
| Color sensor | OV9782, same FOV as depth |
| IMU | None |
| Filter | IR-cut |

Footnote 2 to this table: *"Depth FOV specified at 20 cm for D401 and at 2 m for other modules"* — D405's FOV was characterized at its actual near-field working distance, unlike D415/D435/D455 characterized at 2 m; a minor methodological point (fixed-focus optics shouldn't have range-dependent FOV) but worth knowing if reconciling FOV numbers against other D4xx characterization papers that assume the 2 m convention.

**Physical (Table 3-54, p.69):** 42×42×23 mm nominal, 60 g, mounting via 1× ¼"-20 UNC + 2× M3. Depth start point (Table 4-16, p.90): camera's zero-depth reference is **−3.7 mm** behind the front cover glass (add |Z'| to a front-glass-referenced measurement to get true range). Power (Table 7-10, p.105 area): idle 35 mW, streaming 1.55 W. Uses USB 2.0 or USB 3.1 Gen1 via Micro-B (Table 3-44) — D405 is the only camera in the family without USB-C. No hardware multi-camera sync (p.~114: "D405 does not support the hardware sync signal for multi-camera configuration").

**Derived, not stated — invalid depth band from stereo occlusion (§4.5, p.83, formula given generically, values below are my own substitution of D405's numbers):**

```
DBR = B / (2·Z·tan(HFOV/2))              # ratio of invalid-depth strip to image width, one side
Invalid band [px] = HRES · DBR = B·F/Z    # F = focal length in pixel units
```
With B=18 mm, HFOV=84°, HRES=1280 px, at our stated 10–40 cm working range:

| Standoff Z | Invalid depth strip (one edge) |
|---|---|
| 10 cm | ≈ 10.0% of image width (~128 px) |
| 20 cm | ≈ 5.0% (~64 px) |
| 40 cm | ≈ 2.5% (~32 px) |

This is a **sensor-intrinsic, angle-independent** framing/visibility loss purely from the stereo baseline occluding one imager's view of a strip near the frame edge, and it scales as 1/Z — worst exactly where this project wants to operate (near the 10 cm end). It is distinct from surface self-occlusion and from the incidence-angle penalty; it belongs in the FOV-framing term, computable directly from B, Z, HFOV, HRES with no PLY or normal needed.

## Load-bearing assumptions in this document

1. All KPI numbers are factory/typical-condition, single-target (either fully textured for D405, or texture-less white for the active models) — not validated across surface finish, glossiness, or the specific primitives (box/cylinder) this project uses.
2. The ±2% Z-accuracy and ≥99.5% fill-rate spec applies **only** for range ≤0.5 m and within the central 80% of the FOV (ROI) — edge-of-frame and >0.5 m behavior is unspecified here.
3. D405's accuracy spec assumes ~250 lux ambient light and a **textured** target — the opposite failure mode from active sensors (which struggle on texture-less surfaces but don't care about ambient light or their own target's texture in the same way).
4. Zero angle-dependence data anywhere in the document — every number here is range-only or geometry-only.

## What transfers to our setting

- **Corrects a load-bearing sensor-model error already present elsewhere in this pass.** The `nguyen-izadi-kinect-noise-model.md` note in this same folder states *"The D405 is Intel RealSense active stereo (twin IR imagers + IR texture projector...)"* — this is factually wrong per the primary datasheet (Table 2-2, Table 3-15, footnote 4/Table 4-14). D405 is **passive stereo, visible light, IR-cut filtered, no projector**. This should be corrected wherever the branch's synthesis currently assumes "active stereo" for the D405 specifically (the branch's own framing text: *"an active-stereo sensor like the D405"* — also needs the same correction). This changes which literature is analogous: Kinect-v1/structured-light noise models (dot-pattern defocus, projector power falloff) are the wrong analogy; **passive/ambient stereo-matching noise models** (texture-dependent correspondence, ambient-light sensitivity, ambient specular blowout) are the closer fit.
- **Resolves the "hard minimum shell radius vs. sensor near-field claim" tension the task flagged — precisely, and unfavorably.** True Min-Z at full 1280×720 is 10 cm (Table 4-11), not the marketed 7 cm; only achievable at 7 cm by dropping to 848×480 or below. If the action-space shell's minimum radius needs to clear 10 cm (not 7 cm) at full resolution, that's a real, quantified, resolution/standoff trade-off to hand to whoever sets the shell's inner radius.
- **The invalid-depth-band formula** (derived above) is a concrete, computable, angle-independent framing/visibility term with named inputs (B, Z, HFOV, HRES) straight from stereo geometry — directly answers part of branch Q5 (term table: FOV framing / self-occlusion, PLY-adjacent/sensor-model rather than PLY-derived) with a closed form, and it's *worse* at the close range this project favors, which argues against always pushing standoff to the minimum even before considering incidence angle.
- **Native RGB/depth co-registration on D405** (shared left imager, Table 2-2 footnote 4) is a positive, load-bearing fact for the image-space defect-detectability track (RQ1/RQ from branch Q17–19): no extrinsic RGB-to-depth calibration to worry about when fusing an image-space defect score with a PLY-space geometric one, at least not from sensor design — whatever residual misalignment exists is from the ~4 cm fused-vs-single-view offset (a survey/registration problem), not from RGB-depth sensor extrinsics.
- **The "textured target, ambient light" dependency (footnote 4) surfaces a failure mode not otherwise flagged in this branch**: on a low-texture, matte primitive surface (a bare metal or plastic box face, exactly the object class in scope), D405 passive stereo can lose fill-rate/accuracy **independent of incidence angle** simply from lack of texture, since there is no projector to compensate. This is a candidate root cause worth checking against the "PLY dropout on glossy/dark surfaces" symptom already named in the project's Constraints & Givens — dropout may be as much a texture problem as a specular-angle problem for this specific sensor, which the branch's current framing (all dropout attributed to angle/specularity) doesn't distinguish.

## What does NOT transfer, and why

- **No incidence-angle-vs-error data at all.** This document cannot answer branch Q1 (σ(range,θ) curve) or Q2 (smooth vs. threshold) — it is silent on angle entirely. It was selected (per task framing) to bound the *standoff* term, not the angle term; it does that job and nothing more. A dedicated empirical D405 characterization source (if one exists) is still needed for Q1/Q2, and given point 1 above, that source must be found or run for *passive, visible-light, ambient-light-dependent* stereo — not extrapolated from any active/structured-light sensor's angle curve, because the failure mechanism at grazing angle is different (foreshortened effective baseline + reduced correspondence-window overlap for passive matching, vs. IR-pattern defocus/speckle-decorrelation for active/structured-light).
- **No specular-reflection-into-projector failure mode is possible on D405 at all** (branch Q3, as literally framed) — there is no projector to reflect back. The *symptom* the branch cares about (best view being off-normal on glossy surfaces) may still occur on D405, but through a different, undocumented-here mechanism: ambient-light specular highlight overexposing/saturating both left and right images locally, killing correspondence in that patch regardless of angle-to-projector. This document doesn't characterize that either — it's a hypothesis this reading raises, not something the datasheet confirms.
- **No defect-detectability, image-quality-vs-angle, or scoring-function content** (branch Q4–Q20) — this is pure hardware spec, zero algorithmic content. Contributes range/geometry bounds only.
- **No curve/table finer than the single ≤0.5 m band spec** — cannot be used to argue error is "much better at 10 cm than at 50 cm" quantitatively; that would need to be measured (the ±1.4%-at-20cm figure appearing in third-party summaries, e.g. QVIRO, is not present anywhere in the primary datasheet text and should not be cited to this document).

## Citable anchors

- No IR projector / passive stereo: Table 2-2, p.21 ("IR Projector" rows, D405 column = X); Table 3-49, p.67 ("IR Projector: N/A" for D405); footnote 4 to Table 4-14, p.86.
- IR-cut filter on stereo imagers (visible light only): Table 3-15, p.41 ("Filter Type: IR Cut Filter"); Table 3-49, p.67 ("Filter: IR-cut").
- Shared left-imager/RGB sensor: Table 2-2 footnote 4, p.22; Table 3-15 vs. Table 3-20 (identical OV9782 specs), p.41/44.
- Min-Z by resolution: Table 4-11, p.84.
- Depth quality spec (±2% Z-accuracy, ≥99.5% fill, ≤1% RMS, ≤0.5% temporal): Table 4-14, p.85–86, D401/D405 column.
- Textured-target/ambient-light test condition: footnote 4, Table 4-14, p.86.
- 7 cm–1 m range, lighting-dependent: feature bullet list, p.16.
- Baseline 18 mm, Depth FOV H84/V58/D92: Table 3-49, p.67.
- FOV measured at 20 cm (not 2 m) for D401/D405: footnote 2, Table 3-49, p.67.
- Invalid depth band formula (B, Z, F, HFOV, HRES): §4.5, p.83.
- Depth start point offset (−3.7 mm): Table 4-16, p.90.
- Physical dimensions/mounting: Table 3-54, p.69.
- No IMU: Table 3-49, p.67 ("IMU: None").
- No multi-camera hardware sync for D405: body text, §~4.11 area, p.~114 (verbatim: "D405 does not support the hardware sync signal for multi-camera configuration").

## Branch-question coverage map (this document only)

| Branch question | Answered? | What this document gives |
|---|---|---|
| Q1 (σ(range,θ) curve, D405 10–40 cm) | **No** | Range-only accuracy band (±2% ≤0.5 m); zero angle data |
| Q2 (smooth vs. threshold) | **No** | Not addressed |
| Q3 (orthogonality ever hurts — specular) | **No, but reframes it** | No projector exists to reflect — the branch's stated mechanism doesn't apply to D405; an ambient-light-specular-blowout mechanism is hypothesized, not confirmed here |
| Q4 (defect detectability vs. angle) | **Not addressed** | Out of scope for a spec sheet |
| Q5 (term table) | **Partial, concrete** | Standoff/range bound (Min-Z by resolution), FOV/framing bound (invalid depth band formula, sensor-intrinsic), both PLY-adjacent/sensor-model terms with exact formulas and inputs |
| Q9 (unit of scoring) | Not addressed | — |
| Q10 (which scores survive a defective fused PLY) | **Indirectly relevant** | Confirms depth accuracy/fill-rate specs assume a *textured, well-lit, ≤0.5 m* target — a defective/low-texture/glossy real object plausibly falls well outside the qualified envelope, independent of the fused-PLY defects named elsewhere |
| Q13 (4 cm fused-vs-single-view offset) | **Not addressed directly** | But confirms RGB/depth are natively co-registered on D405 (no sensor-side contribution to that offset — it must come from survey/registration, not sensor extrinsics) |

All other branch questions (viewpoint entropy, NBV information gain, local optimization, VLM-as-judge, ground truth for "best view," benchmark replication, ablations) are **not addressed** — this is a hardware datasheet, not a view-planning or algorithmic-scoring source.
