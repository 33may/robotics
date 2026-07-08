"""humanoid.logic.oli.action — the Action layer (policy logic).

Owns everything policy-specific: obs-vector encoding, the policy's cross-step memory
(history ring + last_actions), the ONNX session, and resolution of raw actions to a
PD command. STAND is analytic; WALK is the LimX walk ONNX. Pure of isaacsim/limxsdk —
it consumes contracts and emits a resolved PolicyOut. Policy selection is by
`Intent.mode` (kept deliberately simple; a richer policy bank is a later design).
"""
