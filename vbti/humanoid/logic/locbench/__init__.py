"""humanoid.logic.locbench — the localization development & evaluation loop (MAY-173).

A TOOL, not part of the robot: sibling of `logic/oli/`, never imported by it (architecture
guard). The engine around the frozen `LocalizationModule` contract — frozen episode sets,
live in-the-loop evaluation over the brain service seam (W4 goals / W5 telemetry), raw
two-tier gated scoring, the self-validating reference candidate. First life: the red/green
harness an implementing agent iterates SLAM adapters against. Second life: the selector that
ranks working candidates for phase-2 deployment. See openspec/changes/may-173-locbench-harness.
"""
