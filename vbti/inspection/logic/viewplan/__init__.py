"""View planning on the reconstructed surface — the alternative to `fit/`.

Where `vbti/inspection/logic/fit/` decomposes a cloud into primitives and would derive
views closed-form, this package plans directly on the geometry: accumulated
object cloud -> mesh -> surface samples -> normal-offset candidate cameras
-> measurability matrix -> orthogonality-weighted greedy set cover -> an
ordered list of camera poses.

Views are poses only. No IK, no reachability, no robot: the single physical
guard is the table (a camera may not sit under it). A planned pose is stored
exactly like a recorded one — 4x4 `T_base_cam`, OpenCV convention
(+Z boresight, +X right, +Y down), robot base frame, metres.

Peer of `vbti/inspection/logic/fit/`: same contract discipline (dataclasses with a
lossless JSON round trip, abstain as a first-class answer), same output
discipline (everything under `vbti/inspection/data/derived/<run>/viewplan/`).
"""
