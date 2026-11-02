# Runbook: Evaluation Analysis

Use after one or more protocol sessions are complete.

## 1. List Sessions

```bash
python -m vbti.logic.inference.eval_helpers ls --detailed
```

## 2. Inspect Latest

```bash
python -m vbti.logic.inference.eval_helpers info latest
```

## 3. Inspect A Specific Session

```bash
python -m vbti.logic.inference.eval_helpers info /absolute/path/to/session
```

Or shorthand if supported:

```bash
python -m vbti.logic.inference.eval_helpers info v020-150k
```

## 4. Group By Tags

```bash
python -m vbti.logic.inference.eval_helpers info latest --group_by=scene
python -m vbti.logic.inference.eval_helpers info latest --group_by=target_color
```

Available group keys depend on protocol trial tags.

## 5. Open Trial Video

```bash
python -m vbti.logic.inference.eval_helpers play latest 3
```

## 6. Render Heatmap

```bash
python -m vbti.logic.inference.eval_render heatmap /absolute/path/to/session --output=/tmp/eval_heatmap.png
```

## 7. Write Result Summary

Template:

```markdown
## Evaluation Summary

- Experiment/version:
- Checkpoint:
- Protocol:
- Action horizon:
- Camera preset:
- Depth/detection:
- Result: X/Y = Z%
- Session path:
- Main failure patterns:
- Valid comparison against:
- Next action:
```

## 8. Decide Next Action

| Pattern | Likely next action |
|---|---|
| Spatial failures clustered in one region | Collect targeted data or evaluate camera placement. |
| Language/color target failures | Check task text, dual-cup data, visual ambiguity. |
| Drop/placement failures after good grasp | Adjust data for placement precision or action horizon. |
| Early reach failures | Check camera/state schema and action distribution. |
| Random failures with camera glitches | Fix hardware before changing model. |
