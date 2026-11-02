# Evaluation Session Analysis

## List Sessions

```bash
python -m vbti.logic.inference.eval_helpers ls
python -m vbti.logic.inference.eval_helpers ls --detailed
```

## Inspect Session

```bash
python -m vbti.logic.inference.eval_helpers info latest
python -m vbti.logic.inference.eval_helpers info v020-150k
python -m vbti.logic.inference.eval_helpers info v020-150k --group_by=scene
```

## Play Trial Video

```bash
python -m vbti.logic.inference.eval_helpers play latest 3
```

## Render Heatmap

```bash
python -m vbti.logic.inference.eval_render heatmap /path/to/session --output=/tmp/heatmap.png
```

Other render modes include `grid` and `pooled` if available in current CLI.

## What To Report

For every evaluation claim include:

- experiment/version;
- checkpoint step/path;
- protocol name;
- action horizon;
- camera preset;
- depth/detection flags;
- success count and denominator;
- session path;
- any failed/cancelled trials;
- relevant breakdowns by tags;
- video/heatmap paths if generated.

## Known Result Anchors

| Session | Result |
|---|---|
| `v017` step 20k, `dual_cup_30` | 11/30 |
| `v018` step 20k, `dual_cup_30` | 14/30 |
| `v019` step 20k, `dual_cup_30` | 20/30 |
| `v020` step 150k, `dual_cup_30` | 30/30 |
| `v020` step 150k, `dual_cup_60` | 56/60 |
| `v020` step 336940, `checkpoint_sweep` | 19/20 |
| `v024` step 167940, `checkpoint_sweep` | 14/20 |

## Valid Comparison Checklist

- Same protocol.
- Same physical setup or documented setup difference.
- Same camera schema.
- Same depth/detection modality expectation.
- Same or intentionally varied action horizon.
- Enough trials to support the claim.

If these are not true, present the result as anecdotal/debugging evidence, not a clean comparison.
