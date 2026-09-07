"""Single Recording contract — the schema as code (v1 final).

These pydantic models ARE the spec: schema-on-write validation for every JSON
file in a run directory, and the single legal source for generated docs and
diagrams. Design: docs/data-engine/schema-draft.md; three-pass review:
docs/data-engine/schema-review.md (decisions: Anton 2026-09-03).

Layout a run obeys:

    runs/<id>/
      run.json                     RunRecord     datasheet + status + methods + conventions
      config.json                  ConfigSnapshot  world-in-effect at start
      session.json                 SessionRecord   camera identity + intrinsics (absent on fake rig)
      events.jsonl                 OperatorEvent   run-level, append-only, one per line
      manifest.json                Manifest        binary artifacts hashed at close
      answer.json                  AnswerRecord    verdict (completed runs)
      steps/<NNN>/step.json        StepRecord      atomic step — THE join key
      steps/<NNN>/view_state.json  ViewState       candidate views at that moment
      steps/<NNN>/*.png|npy|ply    capture + geometry bytes (in manifest)
      ai/<seq>/airun.json          AIRunRecord     one orchestrator session
      ai/<seq>/menu.json           MenuDef         menu definition snapshot
      ai/<seq>/menu_input/<NNN>.json  MenuInput    per-turn AI-tier menu context
      ai/<seq>/trace.jsonl         (event stream — inner typing lands with wiring)
      ai/<seq>/transcripts/<tNNN>.json  TranscriptRecord  VLM subagent runs
      ai/<seq>/artifacts/<tNNN>_<turn>_<tool>.png  tool-use images

Versioning (review B6): per-model VERSION, bumped independently. Readers are
forward-tolerant (unknown fields ignored); writers go through
`validate_for_write`, which refuses unknown fields. Manifest (review B7)
hashes immutable binaries only — schema-managed JSON is validated by
re-validation, so the additive migration gate never contradicts it.
"""
from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1  # baseline; models bump their own VERSION independently

Row4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Mat4 = Annotated[list[Row4], Field(min_length=4, max_length=4)]
Joints6 = Annotated[list[float], Field(min_length=6, max_length=6)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class RecordModel(BaseModel):
    """Base for every on-disk record.

    - read-tolerant: unknown fields are ignored (forward compatibility);
      write-strictness comes from `validate_for_write`
    - version-gated per model: `VERSION` bumps independently per record kind
    - legacy-honest: adapted old runs mark `provenance="legacy"` and list
      the fields the adapter fabricated in `synthesized`
    """

    model_config = ConfigDict(extra="ignore")

    VERSION: ClassVar[int] = 1

    schema_version: int = 1
    provenance: Literal["native", "legacy"] = "native"
    synthesized: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if not 1 <= v <= cls.VERSION:
            raise ValueError(
                f"{cls.__name__} schema_version {v} is newer than this reader "
                f"(max {cls.VERSION}) — refusing to guess"
            )
        return v


def validate_for_write(model_cls: type[RecordModel], data: dict):
    """Write-path validation: strict about unknown fields (typo catcher).

    Read path stays tolerant (`model_validate`); this is the writer's gate.
    Top-level check; recursive nested checking tightens in the build phase.
    """
    unknown = set(data) - set(model_cls.model_fields)
    if unknown:
        raise ValueError(
            f"{model_cls.__name__}: unknown fields on write path: {sorted(unknown)}"
        )
    return model_cls.model_validate(data)


# --- view methods -------------------------------------------------------------

class ViewsphereParams(BaseModel):
    """Params for kind='viewsphere' (the current method, one among future ones)."""

    model_config = ConfigDict(extra="ignore")

    h_bins: int = Field(gt=0)
    v_elevs: list[float] = Field(min_length=1)
    r: float | None = None  # shell radius, derived at survey


_KNOWN_PARAMS = {"viewsphere": ViewsphereParams}


class ViewMethod(BaseModel):
    """How views are generated/addressed. Unknown kinds carry free params."""

    model_config = ConfigDict(extra="ignore")

    id: str
    kind: str  # "viewsphere" | "taught" | future: "primitive-nbv" | ...
    version: str = "1"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_known_kinds(self) -> "ViewMethod":
        model = _KNOWN_PARAMS.get(self.kind)
        if model is not None:
            model.model_validate(self.params)
        return self


class ViewRef(BaseModel):
    """A step's view identity: method + method-defined address.

    address is None for free 6-DoF views — T_base_cam is then the only identity.
    """

    model_config = ConfigDict(extra="ignore")

    method: str
    address: Any | None = None


# --- per-step records ---------------------------------------------------------

class Segmentation(BaseModel):
    """Per-step segmentation result (trimmed by decision: score/box/px only).

    `box` is in the RAW (unrotated) image frame — see RunRecord.conventions.
    """

    model_config = ConfigDict(extra="ignore")

    score: float = Field(ge=0.0, le=1.0)
    box: Annotated[list[int], Field(min_length=4, max_length=4)]  # x0,y0,x1,y1
    px: int = Field(ge=0)


class GeometryStats(BaseModel):
    """Per-step fusion accounting + table plane + cloud provenance.

    `fused_points` = merged cloud size AFTER this step (replay's second anchor;
    Σkept ≠ cloud size because voxel/outlier passes act on the merged cloud).
    `source` records which branch produced the points — mask-exact vs
    depth-growth clouds differ materially and benchmarks must filter on it.
    """

    model_config = ConfigDict(extra="ignore")

    offered: int = Field(ge=0)  # points this view contributed pre-gate
    kept: int = Field(ge=0)  # survived jump gate + filters
    dropped: int = Field(ge=0)  # jump-gate rejects
    fused_points: int = Field(ge=0)  # merged cloud size after this step
    source: Literal["mask", "depth"]  # which identity path produced the cloud
    fallback_reason: str | None = None  # why the mask path was abandoned
    extent_mm: Vec3 | None = None  # fused-cloud AABB extent after this step
    plane: Annotated[list[float], Field(min_length=4, max_length=4)] | None = None


class StepRecord(RecordModel):
    """One atomic step = one robot position. The universal join key.

    Lifecycle (review B1/B2): `outcome` says how the step ended; `phase` is the
    write-protocol discriminator — "captured" means geometry hasn't been written
    yet (crash window is unambiguous), "fused" means the record is complete.
    Pose/capture fields are required only when outcome == "captured".
    """

    step_id: int = Field(ge=0)  # survey = 0
    view: ViewRef
    t_arrived: float
    outcome: Literal["captured", "rejected", "failed"] = "captured"
    phase: Literal["captured", "fused"] = "fused"
    detail: str | None = None  # rejection/failure reason
    t_captured: float | None = None
    joints_rad: Joints6 | None = None
    T_base_flange: Mat4 | None = None
    T_base_cam: Mat4 | None = None  # camera pose in base — robot-free 3D view needs only this
    rgb_rotation_deg: Literal[0, 180] | None = None
    segmentation: Segmentation | None = None
    geometry: GeometryStats | None = None

    @model_validator(mode="after")
    def _captured_needs_pose(self) -> "StepRecord":
        if self.outcome == "captured":
            missing = [f for f in ("t_captured", "joints_rad", "T_base_flange",
                                   "T_base_cam", "rgb_rotation_deg")
                       if getattr(self, f) is None]
            if missing:
                raise ValueError(
                    f"captured step {self.step_id} missing {missing} — "
                    "a captured step carries its full pose"
                )
        return self


class ViewCandidate(BaseModel):
    """One candidate view in a step's ViewState snapshot.

    `unreachable` (hard constraint: no IK/collision-free branch) is distinct
    from `blocked` (soft: planner refused from here, transient).
    """

    model_config = ConfigDict(extra="ignore")

    address: Any | None = None
    pose: Mat4 | None = None
    status: Literal["available", "visited", "blocked", "unreachable", "current"]
    roll: float | None = None  # chosen roll for this candidate, if computed
    scores: dict[str, float] = Field(default_factory=dict)  # NBV features/gains


class ViewState(RecordModel):
    """Candidate view set as it existed at this step — PHYSICAL truth only.

    Timing contract: this is the state in effect when the NEXT action was
    chosen (computed at end-of-step after fusing/recentering). AI-tier context
    (seen cells, findings) lives in MenuInput under the AIRun — menus are pure
    functions of (ViewState, MenuInput).
    """

    step_id: int = Field(ge=0)
    t: float | None = None
    candidates: list[ViewCandidate]
    centroid: Vec3 | None = None  # object centroid, base frame, meters
    extent: Vec3 | None = None  # object AABB extent, meters
    r: float | None = None  # shell radius in effect, meters
    chosen: Any | None = None  # address picked next, if a choice was made
    decider: str | None = None  # who chose: "operator" | "brain" | method id


# --- run-level records --------------------------------------------------------

class Conventions(BaseModel):
    """Self-description written in-band (run.json) — the facts that today live
    only in code and would make the disk uninterpretable in a year."""

    model_config = ConfigDict(extra="ignore")

    transforms: str = ("T_A_B is 4x4 row-major, maps B-frame points into A; "
                       "base = UR controller base; meters")
    angles: str = "radians"
    clock: str = "unix wall seconds"
    depth_scale_ref: str = "session.json depth_scale_m_per_unit; 0 = invalid"
    mask_polarity: str = "255 = object, 0 = background"
    box_frame: str = "segmentation.box is in the RAW (unrotated) image frame"
    rotated_artifacts: list[str] = Field(
        default_factory=lambda: ["rgb.png", "depth_aligned.npy", "mask.png"])
    # rgb_rotation_deg applies to rotated_artifacts ONLY; depth_raw/ir_* stay raw


class RunRecord(RecordModel):
    """run.json — the datasheet: identity, motivation, status, view methods."""

    id: str
    name: str
    object: str | None = None  # free tag: cup, box, tube... (AI queries it)
    object_instance: str | None = None  # "same physical cup across runs"
    source: Literal["live", "data-engine"]  # stamped by the running code
    rig: Literal["real", "fake"] = "real"  # was the depth real hardware?
    tags: list[str] = Field(default_factory=list)  # debug/demo/... for curation
    question: str | None = None
    status: Literal["running", "completed", "aborted", "crashed"]
    created_at: float
    closed_at: float | None = None
    view_methods: list[ViewMethod] = Field(default_factory=list)
    steps: list[int] = Field(default_factory=list)  # ordered step_ids
    q_survey: Joints6 | None = None
    conventions: Conventions = Field(default_factory=Conventions)
    migrations: list[str] = Field(default_factory=list)  # applied migration ids


class Intrinsics(BaseModel):
    """Camera intrinsics for one stream, resolution-bound."""

    model_config = ConfigDict(extra="ignore")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str  # distortion model name — coefficients without it are noise
    coeffs: list[float] = Field(default_factory=list)


class StereoExtrinsics(BaseModel):
    """IR stereo pair mounting (rotation row-major 3x3 + translation)."""

    model_config = ConfigDict(extra="ignore")

    rotation: Annotated[list[float], Field(min_length=9, max_length=9)]
    translation_m: Vec3


class SessionRecord(RecordModel):
    """session.json — camera identity + calibration-level data, once per run.

    Absent on fake-rig runs (RunRecord.rig == "fake" declares why).
    The depth arrays' scale lives HERE — referenced, not repeated, per view.
    """

    serial: str
    resolution: Annotated[list[int], Field(min_length=2, max_length=2)]
    depth_scale_m_per_unit: float = Field(gt=0)
    intrinsics: dict[str, Intrinsics] = Field(default_factory=dict)
    extrinsics_ir1_to_ir2: StereoExtrinsics | None = None


class HashedFile(BaseModel):
    """A file identified by name + content hash (calib artifact, cell.yaml...)."""

    model_config = ConfigDict(extra="ignore")

    file: str | None = None
    sha256: Sha256Hex
    content: str | None = None  # embedded copy when small (cell.yaml)


class ConfigSnapshot(RecordModel):
    """config.json — the world-in-effect at run start.

    Makes "safe and doable" re-checkable and old runs renderable after
    cell.yaml/calib edits. Model identities live HERE, not per step.
    """

    git_sha: str
    cell_yaml: HashedFile
    calib: HashedFile  # resolved T_flange_cam_*.npy identity (not "latest")
    constants: dict[str, Any] = Field(default_factory=dict)  # tuning in effect
    models: dict[str, str] = Field(default_factory=dict)  # role -> model id


class MenuDef(RecordModel):
    """ai/<seq>/menu.json — the menu definition, snapshot + shared id.

    The menu is the pure function (ViewState, MenuInput) -> LLM text; this is
    its definition side. Provenance follows the scores.jsonl pattern:
    version (semver) + code_sha (renderer code identity).
    Rendered per-turn text lives in the AI trace.
    """

    menu_id: str
    version: str
    content_hash: Sha256Hex
    code_sha: str  # renderer code identity — semver alone binds nothing
    verbs: list[str] = Field(min_length=1)  # tool surface offered to the model
    renders_kinds: list[str] = Field(default_factory=list)  # view-method kinds
    template: dict[str, Any] = Field(default_factory=dict)


class MenuInput(RecordModel):
    """ai/<seq>/menu_input/<NNN>.json — AI-tier menu context, one per turn.

    ViewState is physical truth; THIS carries what the agent-side rendering
    additionally consumed (seen cells, findings count, recommendations).
    `agent_context` tightens when the eyes wiring lands; the envelope is the
    contract. Exact replay injects recorded text; menu experiments re-render
    from (ViewState[step_id], MenuInput[turn]).
    """

    turn: int = Field(ge=0)
    step_id: int = Field(ge=0)
    t: float
    agent_context: dict[str, Any] = Field(default_factory=dict)


class AIRunRecord(RecordModel):
    """ai/<seq>/airun.json — one orchestrator session over the run."""

    seq: int = Field(ge=0)
    mode: Literal["live", "replay"]
    orchestrator_model: str
    menu_id: str
    menu_hash: Sha256Hex  # must equal the MenuDef.content_hash it ran with
    opening_prompt: str | None = None  # verbatim first model input
    usage: dict[str, float] | None = None  # tokens/cost/duration — was discarded


class TranscriptRecord(RecordModel):
    """ai/<seq>/transcripts/<tNNN>.json — one VLM subagent run, stored ONCE.

    The image-tier agent's full reasoning: verbatim prompt, every turn (each
    carries a `type` discriminator; inner shapes tighten at wiring), final
    answer. `kind` says what invoked it — the filename never has to. `step_id`
    is the sync link to the atomic step whose capture it looked at.
    Image-producing tool uses save under artifacts/ keyed
    (transcript, turn, tool) — collisions impossible.
    """

    transcript_id: str
    kind: Literal["survey", "inspect", "move", "evidence"]
    step_id: int = Field(ge=0)
    t: float
    model: str  # VLM identity — unrecorded today, required here
    task: str
    prompt: str  # verbatim text sent to the model
    turns: list[dict[str, Any]] = Field(default_factory=list)
    answer: dict[str, Any] | list[Any] | str | None = None  # archive has list answers
    answer_schema: dict[str, Any] | None = None  # what the model was constrained to
    artifacts: list[str] = Field(default_factory=list)  # relative paths
    usage: dict[str, float] | None = None

    @field_validator("turns")
    @classmethod
    def _turns_carry_type(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, turn in enumerate(v):
            if "type" not in turn:
                raise ValueError(f"turn {i} lacks a 'type' discriminator")
        return v


class EvidenceImage(BaseModel):
    """One verdict->image citation in the answer."""

    model_config = ConfigDict(extra="ignore")

    step_id: int = Field(ge=0)
    transcript_id: str
    artifact: str  # relative path under the AIRun
    note: str | None = None


class AnswerRecord(RecordModel):
    """answer.json — the verdict, with explicit back-references."""

    verdict: str
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    evidence_images: list[EvidenceImage] = Field(default_factory=list)
    step_ids: list[int] = Field(default_factory=list)
    transcript_ids: list[str] = Field(default_factory=list)


class OperatorEvent(BaseModel):
    """One line of events.jsonl — run-level, append-only, both run kinds.

    No per-line schema_version: the run's version covers the file.
    """

    model_config = ConfigDict(extra="ignore")

    t: float
    kind: Literal["requested", "awaiting_approval", "approved", "redirected",
                  "cancelled", "captured", "blocked", "stopped", "fault"]
    step_id: int | None = None
    detail: str | None = None


class FileEntry(BaseModel):
    """One manifest entry: content hash + cheap-check fields."""

    model_config = ConfigDict(extra="ignore")

    sha256: Sha256Hex
    bytes: int = Field(ge=0)
    mtime: float


class Manifest(RecordModel):
    """manifest.json — immutable BINARY artifacts (png/npy/ply), at close.

    Schema-managed JSON is validated by re-validation, not byte hash — so the
    additive migration gate never contradicts the integrity check (review B7).
    bytes+mtime give an instant pre-pass; full re-hash is the deep check.
    Migrations extend this add-only; existing entries never change.
    """

    files: dict[str, FileEntry] = Field(default_factory=dict)


class DerivationMeta(RecordModel):
    """derived/<run>/<method>[/<variant>]/meta.json — derivation provenance.

    The scores.jsonl pattern made uniform: version = "<method>/<semver>+p:<code-hash>".
    `steps` records an explicit ok/failed entry per step — never a silent gap
    (the cup5 step-029 lesson). Staleness = recorded source_hashes vs current.
    """

    method: str
    variant: str | None = None
    version: str  # "<method>/<semver>+p:<code-hash>"
    params: dict[str, Any] = Field(default_factory=dict)
    source_hashes: dict[str, Sha256Hex] = Field(default_factory=dict)
    steps: dict[int, str] = Field(default_factory=dict)  # step_id -> ok | failed: why
    created_at: float


# --- directory-layout relations (for generated diagrams) ----------------------
# Composition INSIDE a file is derived from field types automatically; these are
# the cross-file relations the directory layout expresses. Mermaid cardinality.
RELATIONS = [
    ("RunRecord", "||--||", "ConfigSnapshot", "config.json"),
    ("RunRecord", "||--o|", "SessionRecord", "session.json (absent on fake rig)"),
    ("RunRecord", "||--||", "Manifest", "manifest.json"),
    ("RunRecord", "||--o{", "StepRecord", "steps/NNN/step.json"),
    ("RunRecord", "||--o{", "AIRunRecord", "ai/seq/"),
    ("RunRecord", "||--o|", "AnswerRecord", "answer.json"),
    ("RunRecord", "||--o{", "OperatorEvent", "events.jsonl"),
    ("StepRecord", "||--||", "ViewState", "steps/NNN/view_state.json"),
    ("AIRunRecord", "}o--||", "MenuDef", "ai/seq/menu.json"),
    ("AIRunRecord", "||--o{", "MenuInput", "ai/seq/menu_input/NNN.json"),
    ("AIRunRecord", "||--o{", "TranscriptRecord", "transcripts/tNNN.json"),
    ("TranscriptRecord", "}o--||", "StepRecord", "step_id"),
    ("RunRecord", "||--o{", "DerivationMeta", "derived/run/method/meta.json"),
]
