#!/usr/bin/env python3
"""Single Recording contract — pydantic models. Run: p inspection/tests/test_record_schema.py

The schema IS the spec: docs and diagrams are generated from these models, so
every rule the disk must obey is asserted here first (TDD).
Covers schema v1 final: three-pass review fixes (docs/data-engine/schema-review.md).
"""
import json

import pytest
from pydantic import ValidationError

from inspection.record.schema import (
    SCHEMA_VERSION,
    AIRunRecord,
    AnswerRecord,
    ConfigSnapshot,
    DerivationMeta,
    GeometryStats,
    Manifest,
    MenuDef,
    MenuInput,
    OperatorEvent,
    RunRecord,
    Segmentation,
    SessionRecord,
    StepRecord,
    TranscriptRecord,
    ViewMethod,
    ViewState,
    validate_for_write,
)

T4 = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def _run_kwargs(**over):
    kw = dict(
        id="0309-boxA",
        name="boxA",
        object="box",
        source="live",
        rig="real",
        question="is there a logo?",
        status="running",
        created_at=1788180000.0,
        view_methods=[
            {"id": "vs1", "kind": "viewsphere",
             "params": {"h_bins": 12, "v_elevs": [10.0, 40.0, 70.0], "r": 0.24}},
        ],
        steps=[0, 1],
        q_survey=[0.0] * 6,
    )
    kw.update(over)
    return kw


def _step_kwargs(**over):
    kw = dict(
        step_id=1,
        view={"method": "vs1", "address": [3, 0]},
        t_arrived=1788180307.0,
        t_captured=1788180309.2,
        joints_rad=[0.1] * 6,
        T_base_flange=T4,
        T_base_cam=T4,
        rgb_rotation_deg=0,
        segmentation={"score": 0.97, "box": [218, 223, 387, 371], "px": 15592},
    )
    kw.update(over)
    return kw


# --- round trip ---------------------------------------------------------------

def test_run_record_round_trips_through_json():
    run = RunRecord(**_run_kwargs())
    back = RunRecord.model_validate_json(run.model_dump_json())
    assert back == run
    assert back.schema_version == SCHEMA_VERSION


def test_step_record_round_trips_through_json():
    step = StepRecord(**_step_kwargs())
    back = StepRecord.model_validate_json(step.model_dump_json())
    assert back == step


# --- version gate -------------------------------------------------------------

def test_future_schema_version_is_rejected_loudly():
    raw = json.loads(RunRecord(**_run_kwargs()).model_dump_json())
    raw["schema_version"] = SCHEMA_VERSION + 999
    with pytest.raises(ValidationError):
        RunRecord.model_validate(raw)


def test_version_gate_is_per_model():
    assert RunRecord.VERSION == 1
    assert StepRecord.VERSION == 1  # bumps independently later


# --- read-tolerant / write-strict (B6) ---------------------------------------

def test_read_path_tolerates_unknown_fields():
    raw = json.loads(RunRecord(**_run_kwargs()).model_dump_json())
    raw["field_from_the_future"] = 42
    run = RunRecord.model_validate(raw)  # forward-tolerant read
    assert run.name == "boxA"


def test_write_path_rejects_unknown_fields():
    raw = json.loads(RunRecord(**_run_kwargs()).model_dump_json())
    raw["typo_field"] = 1
    with pytest.raises(ValueError):
        validate_for_write(RunRecord, raw)


def test_write_path_accepts_clean_data():
    raw = json.loads(RunRecord(**_run_kwargs()).model_dump_json())
    assert validate_for_write(RunRecord, raw).name == "boxA"


# --- provenance for legacy adaptation (B8) ------------------------------------

def test_provenance_defaults_native():
    assert RunRecord(**_run_kwargs()).provenance == "native"


def test_legacy_records_declare_synthesized_fields():
    step = StepRecord(**_step_kwargs(provenance="legacy",
                                     synthesized=["t_arrived"]))
    back = StepRecord.model_validate_json(step.model_dump_json())
    assert back.synthesized == ["t_arrived"]


# --- view methods -------------------------------------------------------------

def test_viewsphere_params_are_validated():
    with pytest.raises(ValidationError):
        ViewMethod(id="vs1", kind="viewsphere", params={"h_bins": 12})  # no v_elevs


def test_unknown_method_kind_carries_free_params():
    m = ViewMethod(id="nbv1", kind="primitive-nbv", params={"faces": 6})
    assert m.params["faces"] == 6


def test_step_view_address_may_be_absent_for_free_poses():
    step = StepRecord(**_step_kwargs(view={"method": "free6dof", "address": None}))
    assert step.view.address is None


# --- physical sanity ----------------------------------------------------------

def test_transforms_must_be_4x4():
    with pytest.raises(ValidationError):
        StepRecord(**_step_kwargs(T_base_cam=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))


def test_joints_must_be_six():
    with pytest.raises(ValidationError):
        StepRecord(**_step_kwargs(joints_rad=[0.0] * 5))


def test_rgb_rotation_only_0_or_180():
    with pytest.raises(ValidationError):
        StepRecord(**_step_kwargs(rgb_rotation_deg=90))


# --- step lifecycle (B1/B2) ---------------------------------------------------

def test_rejected_step_needs_no_pose():
    step = StepRecord(step_id=5, view={"method": "vs1", "address": [4, 0]},
                      t_arrived=1788180400.0, outcome="rejected",
                      phase="captured", detail="plane gate: tilt 4.2deg")
    back = StepRecord.model_validate_json(step.model_dump_json())
    assert back.outcome == "rejected" and back.joints_rad is None


def test_captured_step_requires_full_pose():
    with pytest.raises(ValidationError):
        StepRecord(step_id=1, view={"method": "vs1", "address": [3, 0]},
                   t_arrived=1.0, outcome="captured", phase="fused")  # no pose


def test_phase_discriminates_crash_window():
    step = StepRecord(**_step_kwargs(phase="captured", geometry=None))
    assert step.phase == "captured"  # geometry=None here means "not yet", not "failed"


# --- segmentation -------------------------------------------------------------

def test_segmentation_is_optional():
    step = StepRecord(**_step_kwargs(segmentation=None))
    assert step.segmentation is None


def test_segmentation_score_bounded():
    with pytest.raises(ValidationError):
        Segmentation(score=1.5, box=[0, 0, 1, 1], px=1)


# --- geometry stats (A9/B3) ---------------------------------------------------

def _geo(**over):
    kw = dict(offered=3843, kept=3694, dropped=149, fused_points=2172,
              source="mask", extent_mm=[80.0, 82.0, 120.0],
              plane=[0.0, 0.0, 1.0, -0.012])
    kw.update(over)
    return kw


def test_step_carries_geometry_stats():
    step = StepRecord(**_step_kwargs(geometry=_geo()))
    back = StepRecord.model_validate_json(step.model_dump_json())
    assert back.geometry.kept == 3694
    assert back.geometry.fused_points == 2172


def test_geometry_counts_cannot_be_negative():
    with pytest.raises(ValidationError):
        GeometryStats(**_geo(kept=-1))


def test_geometry_records_cloud_branch():
    geo = GeometryStats(**_geo(source="depth", fallback_reason="mask lifted no depth"))
    assert geo.source == "depth"
    with pytest.raises(ValidationError):
        GeometryStats(**_geo(source="magic"))


# --- view state (B9) ----------------------------------------------------------

def test_view_state_round_trips():
    vs = ViewState(
        step_id=1,
        t=1788180310.0,
        candidates=[
            {"address": [3, 0], "pose": T4, "status": "available",
             "roll": 0.0, "scores": {"gain": 0.7}},
            {"address": [3, 1], "pose": T4, "status": "visited"},
            {"address": [5, 2], "status": "unreachable"},
        ],
        centroid=[0.3, 0.0, 0.05],
        extent=[0.08, 0.08, 0.12],
        r=0.24,
        chosen=[3, 0],
    )
    back = ViewState.model_validate_json(vs.model_dump_json())
    assert back == vs


def test_view_state_rejects_unknown_status():
    with pytest.raises(ValidationError):
        ViewState(step_id=1,
                  candidates=[{"address": [3, 0], "pose": T4, "status": "maybe"}])


# --- menu input snapshot (B9) -------------------------------------------------

def test_menu_input_round_trips():
    mi = MenuInput(turn=3, step_id=1, t=1788180311.0,
                   agent_context={"seen": [[3, 0]], "findings": 21})
    assert MenuInput.model_validate_json(mi.model_dump_json()) == mi


# --- manifest (A10/B7) --------------------------------------------------------

def test_manifest_entries_carry_hash_size_mtime():
    m = Manifest(files={"steps/001/rgb.png":
                        {"sha256": "a" * 64, "bytes": 498000,
                         "mtime": 1788180310.0}})
    assert m.files["steps/001/rgb.png"].bytes == 498000


def test_manifest_requires_sha256_hex():
    with pytest.raises(ValidationError):
        Manifest(files={"x.png": {"sha256": "zz", "bytes": 1, "mtime": 0.0}})


# --- run datasheet ------------------------------------------------------------

def test_source_and_status_are_closed_enums():
    with pytest.raises(ValidationError):
        RunRecord(**_run_kwargs(source="manual"))
    with pytest.raises(ValidationError):
        RunRecord(**_run_kwargs(status="paused"))


def test_rig_and_tags_and_instance(caplog=None):
    run = RunRecord(**_run_kwargs(rig="fake", tags=["debug"],
                                  object_instance="blue-cup-1"))
    assert run.rig == "fake" and run.tags == ["debug"]
    with pytest.raises(ValidationError):
        RunRecord(**_run_kwargs(rig="synthetic"))


def test_conventions_declared_in_band():
    run = RunRecord(**_run_kwargs())
    raw = json.loads(run.model_dump_json())
    assert "mask_polarity" in raw["conventions"]
    assert "box_frame" in raw["conventions"]


# --- session (A1) -------------------------------------------------------------

def _session_kwargs(**over):
    intr = {"width": 848, "height": 480, "fx": 431.0, "fy": 431.0,
            "ppx": 424.0, "ppy": 240.0, "model": "brown_conrady",
            "coeffs": [0.0] * 5}
    kw = dict(
        serial="123622270954",
        resolution=[848, 480],
        depth_scale_m_per_unit=9.999999747378752e-05,
        intrinsics={"color": intr, "depth": intr,
                    "ir_left": intr, "ir_right": intr},
        extrinsics_ir1_to_ir2={"rotation": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
                               "translation_m": [0.0179, 0.0, 0.0]},
    )
    kw.update(over)
    return kw


def test_session_record_round_trips():
    s = SessionRecord(**_session_kwargs())
    assert SessionRecord.model_validate_json(s.model_dump_json()) == s


def test_session_depth_scale_is_positive():
    with pytest.raises(ValidationError):
        SessionRecord(**_session_kwargs(depth_scale_m_per_unit=0.0))


# --- config snapshot ----------------------------------------------------------

def _config_kwargs(**over):
    kw = dict(
        git_sha="76fabc3",
        cell_yaml={"sha256": "b" * 64, "content": "frames:\n  table: {}\n"},
        calib={"file": "T_flange_cam_2026-08-19.npy", "sha256": "c" * 64},
        constants={"VOXEL_M": 0.003, "JUMP_GATE_M": 0.08},
        models={"segmentation": "sam3", "vlm": "gemini-robotics-er-2-preview"},
    )
    kw.update(over)
    return kw


def test_config_snapshot_round_trips():
    cfg = ConfigSnapshot(**_config_kwargs())
    assert ConfigSnapshot.model_validate_json(cfg.model_dump_json()) == cfg


def test_config_hashes_must_be_sha256():
    with pytest.raises(ValidationError):
        ConfigSnapshot(**_config_kwargs(cell_yaml={"sha256": "nope", "content": ""}))


# --- menu ---------------------------------------------------------------------

def test_menu_def_round_trips():
    menu = MenuDef(menu_id="moves-v1", version="1", content_hash="d" * 64,
                   code_sha="76fabc3",
                   verbs=["plan", "inspect", "move", "answer"],
                   renders_kinds=["viewsphere"],
                   template={"style": "ascii-grid"})
    assert MenuDef.model_validate_json(menu.model_dump_json()) == menu


def test_menu_needs_at_least_one_verb():
    with pytest.raises(ValidationError):
        MenuDef(menu_id="m", version="1", content_hash="d" * 64,
                code_sha="x", verbs=[], renders_kinds=["viewsphere"])


# --- AI run (A2/A3) -----------------------------------------------------------

def test_airun_mode_is_live_or_replay():
    ai = AIRunRecord(seq=0, mode="replay", orchestrator_model="claude-opus-5",
                     menu_id="moves-v1", menu_hash="d" * 64)
    assert AIRunRecord.model_validate_json(ai.model_dump_json()) == ai
    with pytest.raises(ValidationError):
        AIRunRecord(seq=0, mode="offline", orchestrator_model="x",
                    menu_id="m", menu_hash="d" * 64)


def test_airun_carries_usage_and_opening_prompt():
    ai = AIRunRecord(seq=0, mode="live", orchestrator_model="claude-opus-5",
                     menu_id="m", menu_hash="d" * 64,
                     opening_prompt="Question: is there a logo?\nSTATE...",
                     usage={"output_tokens": 4231, "cost_usd": 0.51})
    back = AIRunRecord.model_validate_json(ai.model_dump_json())
    assert back.usage["output_tokens"] == 4231


# --- AI subagent transcript ---------------------------------------------------

def _transcript_kwargs(**over):
    kw = dict(
        transcript_id="t001",
        kind="inspect",
        step_id=1,
        t=1788180312.0,
        model="gemini-robotics-er-2-preview",
        task="is there a logo on this side?",
        prompt="You are inspecting view [3,0]...",
        turns=[{"type": "tool_call", "tool": "crop",
                "args": {"box": [454, 401, 794, 654]},
                "artifact": "artifacts/t001_01_crop.png"},
               {"type": "text", "text": "I can see a printed logo."}],
        answer={"found": True, "evidence": ["white print on side"]},
        answer_schema={"type": "object"},
        artifacts=["artifacts/t001_01_crop.png"],
    )
    kw.update(over)
    return kw


def test_transcript_round_trips():
    tr = TranscriptRecord(**_transcript_kwargs())
    assert TranscriptRecord.model_validate_json(tr.model_dump_json()) == tr


def test_transcript_kind_is_closed():
    with pytest.raises(ValidationError):
        TranscriptRecord(**_transcript_kwargs(kind="dream"))


def test_transcript_requires_model_identity():
    kw = _transcript_kwargs()
    del kw["model"]
    with pytest.raises(ValidationError):
        TranscriptRecord(**kw)


def test_transcript_links_to_its_step():
    with pytest.raises(ValidationError):
        TranscriptRecord(**_transcript_kwargs(step_id=-1))


def test_transcript_answer_may_be_list():
    tr = TranscriptRecord(**_transcript_kwargs(answer=["logo", "handle"]))
    assert tr.answer == ["logo", "handle"]  # 2 real archive transcripts are lists


def test_transcript_turns_require_type_key():
    with pytest.raises(ValidationError):
        TranscriptRecord(**_transcript_kwargs(turns=[{"tool": "crop"}]))


# --- answer (A4) --------------------------------------------------------------

def test_answer_record_references_steps_and_transcripts():
    ans = AnswerRecord(verdict="yes", reasoning="logo seen on cell [3,0]",
                       evidence=["white logo print on side"],
                       step_ids=[1, 2], transcript_ids=["t001", "t003"])
    back = AnswerRecord.model_validate_json(ans.model_dump_json())
    assert back.step_ids == [1, 2]


def test_answer_evidence_images_typed():
    ans = AnswerRecord(verdict="yes", reasoning="r",
                       evidence_images=[{"step_id": 1, "transcript_id": "t001",
                                         "artifact": "artifacts/t001_01_crop.png",
                                         "note": "logo visible"}])
    assert ans.evidence_images[0].step_id == 1


def test_answer_carries_views_inspected_and_coverage():
    """brain/loop.py:_answer's payload — cells the agent actually looked at
    plus the ASCII coverage map in effect when it answered. Optional: legacy
    answers and non-AI runs never had these."""
    ans = AnswerRecord(verdict="yes", reasoning="r",
                       views_inspected=[[3, 0], [3, 1]],
                       coverage="2/48 cells seen (# seen · . unseen · @ current)")
    back = AnswerRecord.model_validate_json(ans.model_dump_json())
    assert back.views_inspected == [[3, 0], [3, 1]]
    assert back.coverage.startswith("2/48")


def test_answer_views_inspected_and_coverage_are_optional():
    ans = AnswerRecord(verdict="yes", reasoning="r")
    assert ans.views_inspected == [] and ans.coverage is None


# --- derivation provenance ----------------------------------------------------

def test_derivation_meta_round_trips():
    m = DerivationMeta(method="cyl", version="cyl/1.0+p:bf23b3aa",
                       params={"seed": 3},
                       source_hashes={"steps/000/cloud.ply": "a" * 64},
                       steps={0: "ok", 1: "failed: too few points"},
                       created_at=1788180100.0)
    back = DerivationMeta.model_validate_json(m.model_dump_json())
    assert back == m and back.steps[1].startswith("failed")


# --- operator events ----------------------------------------------------------

def test_operator_event_kinds_are_closed():
    ev = OperatorEvent(t=1788180310.0, kind="approved", step_id=1)
    assert OperatorEvent.model_validate_json(ev.model_dump_json()) == ev
    with pytest.raises(ValidationError):
        OperatorEvent(t=0.0, kind="shrugged")


def test_operator_event_blocked_kind_exists():
    ev = OperatorEvent(t=1.0, kind="blocked", step_id=None,
                       detail="cell [4,0]: no collision-free path")
    assert ev.kind == "blocked"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
