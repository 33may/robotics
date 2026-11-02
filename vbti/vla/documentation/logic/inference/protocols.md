# Evaluation Protocols

## Location

```text
logic/inference/protocols/
```

Contains protocol JSON files, render/edit tools, and generators.

## Protocol Tools

```bash
python -m vbti.logic.inference.protocols.protocols render id_scale_60
python -m vbti.logic.inference.protocols.protocols verify id_scale_60
python -m vbti.logic.inference.protocols.protocols edit id_scale_60
python -m vbti.logic.inference.protocols.render_protocol dual_cup_60
python -m vbti.logic.inference.protocols.generators.make_checkpoint_sweep
python -m vbti.logic.inference.protocols.generators.make_dual_cup_60
```

Outputs:

```text
logic/inference/protocols/renders/<name>_overview.png
logic/inference/protocols/renders/<name>_verify.png
```

## Legacy Schema

Example shape:

```json
{
  "name": "id_scale_60",
  "version": "v1",
  "task": "pick up the duck and place it in the cup",
  "total_trials": 60,
  "trials": [
    {
      "trial_id": 0,
      "zone": "ID",
      "duck_px": [289, 245],
      "cup_px": [221, 117],
      "duck_dir_deg": 219.8,
      "cup_group": 0
    }
  ]
}
```

## Entity Schema

Used by newer protocols such as `checkpoint_sweep`, `dual_cup_15`, `dual_cup_30`, and `dual_cup_60`.

```json
{
  "name": "checkpoint_sweep",
  "schema": "entities",
  "task_template": "Pick up the duck and place it in the {color} cup",
  "total_trials": 20,
  "trials": [
    {
      "trial_id": 0,
      "entities": [
        {"name": "duck", "kind": "duck", "color": "yellow", "px": [391, 250]},
        {"name": "A", "kind": "cup", "color": "red", "px": [248, 297]}
      ],
      "target": "A",
      "task": "Pick up the duck and place it in the red cup",
      "tags": {"scene": "single_red", "target_color": "red"}
    }
  ]
}
```

## Design Rules

- Protocol defines trials before evaluation starts.
- Task text should be stored per trial when target changes.
- Tags should capture scene, target color, cup count, ID/OOD, or failure-relevant buckets.
- Do not mix protocols for quantitative comparison.

## Common Protocols

| Protocol | Use |
|---|---|
| `id_scale_60` | Older ID/OOD placement scaling evaluation. |
| `dual_cup_30` | 30-trial dual/single cup benchmark. |
| `dual_cup_60` | Larger 60-trial benchmark; v020 near-saturated it. |
| `checkpoint_sweep` | Compact checkpoint comparison protocol. |
