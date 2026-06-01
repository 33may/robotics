# Dynamic Inference Prompt Control Design

## Goal

Replace the current fixed-at-start inference prompt with a live prompt source that can change during robot execution. The first milestone is a deterministic red/black prompt toggle; the same interface will later accept voice-recognized text and GUI commands.

## Current Context

`vbti/logic/inference/run_real_inference.py` currently passes `task` into `_build_observation(...)` each control loop. `_build_observation(...)` writes `policy_input["task"] = task` before preprocessing. In the main `run(...)` path, observations are sent into `AsyncChunkRunner.step(obs)`, which predicts action chunks and refreshes them asynchronously.

This means the model can already receive a different prompt per observation, but prompt changes are gated by chunk timing unless the runner is reset.

## Design

### Runtime Prompt Source

Add a small runtime prompt state object with:

- current prompt string
- monotonically increasing version number
- last-update source label, initially `button`

The inference loop reads the current prompt every step and passes it into `_build_observation(...)`. The toggle prototype switches between:

- `Pick up the duck and place it in the red cup`
- `Pick up the duck and place it in the black cup`

### Prompt Update Modes

Expose two modes with a CLI option such as `--prompt-update-mode smooth|responsive`.

#### Smooth mode

When the prompt changes, only the runtime prompt state changes. The async runner naturally picks up the new prompt when it plans the next chunk.

Use this when motion smoothness matters more than instant language switching.

#### Responsive mode

When the prompt changes, update the runtime prompt state and reset the async runner. The next runner step plans a fresh chunk using the new prompt.

Use this for demos where the prompt must visibly affect behavior as soon as possible. This may produce a less smooth transition than RTC-guided chunk refresh.

### First Control Surface

Start with one local control input for red/black switching. The simplest integration point is the existing OpenCV display key handling in the live camera window:

- `t` toggles red/black prompt
- `q` keeps the existing quit behavior

The active prompt should be visible in terminal output and ideally in the camera HUD so the operator can confirm which command the model is receiving.

### Voice Integration Later

Voice recognition should not talk directly to the policy or runner. It should write recognized text into the same runtime prompt source used by the button toggle. This keeps the inference loop independent of the voice backend and lets us test prompt switching without audio first.

Initial voice behavior:

- background process/thread listens for a push-to-talk button
- when the held-button recording finishes, speech-to-text produces a prompt
- recognized prompt updates runtime prompt state
- selected update mode determines whether the runner resets

### Demo GUI Later

The GUI should also use the same prompt source. It can display:

- active prompt
- last recognized voice text
- prompt update mode
- model/run status
- red/black quick buttons

## Testing Plan

1. Run without voice using the keyboard toggle.
2. Confirm the displayed active prompt changes immediately.
3. In smooth mode, confirm no runner reset happens and prompt effect may lag until next chunk.
4. In responsive mode, confirm the runner resets on prompt change.
5. Add voice only after the prompt source and two modes are verified.

## Out of Scope For First Implementation

- Full GUI
- Speech recognition backend selection
- Natural-language command parsing beyond direct prompt text
- Multi-command history or arbitration
- Protocol/evaluation prompt changes
