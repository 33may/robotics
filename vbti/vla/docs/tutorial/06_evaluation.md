# 6. Evaluation

**What you have:** checkpoints pulled into `experiments/<exp>/<vNNN>/<run_name>/checkpoints/<NNNNNN>/pretrained_model/` (block 5). `eval_engine` runs them on the real robot through the same inference path as block 7, but against a protocol: a JSON in `logic/inference/protocols/` that fixes every trial up front (object placements in top-camera pixels, target, prompt, tags). Each session writes `session.json` (per-trial results), `session.md` (summary) and `videos/` to `experiments/<exp>/<vNNN>/eval_sessions/chkpt_<step>_ah_<horizon>_pr_<protocol>_<timestamp>/`.

**What you need to do:** pick or generate a protocol, run it per checkpoint with the camera preset, action horizon and depth flag the comparison will keep fixed, then read the sessions. A result is only comparable to sessions of the same protocol under the same settings.

```bash
# protocol: generate, render, hand-adjust placements
python logic/inference/protocols/generators/make_<protocol>.py
python -m vla.logic.inference.protocols.render_protocol <protocol>
python -m vla.logic.inference.protocols.protocols edit <protocol>

# session: one per checkpoint, the version must be active
python -m vla.logic.train.experiment_utils use <exp> <vNNN>
python -m vla.logic.inference.eval_engine \
  --checkpoint=<step> \
  --protocol=<protocol> \
  --port=<follower_port> \
  --cameras=<preset> \
  --depth=<true|false> \
  --action_horizon=10 \
  --max_steps=10000 \
  --record=True
# interrupted session: same command + --resume=True

# analysis
python -m vla.logic.inference.eval_helpers ls
python -m vla.logic.inference.eval_helpers info latest --group_by=<tag>
python -m vla.logic.inference.eval_helpers play latest <trial_id>
python -m vla.logic.inference.eval_render heatmap <session_dir>
```

Controls: `SPACE` start trial · `S` success · `F` failure · `V` save video · `N` add note · `SPACE` next trial · `Q` quit.

Checkpoint sweep: `experiment_utils ckpts` lists the saved steps; run `checkpoint_sweep` once per step with identical flags; `eval_helpers ls` puts the success rates side by side.

**Done when:**
- the session dir holds `session.json`, `session.md` and one `videos/trial_NN_<success|failure>.mp4` per trial
- `eval_helpers info` shows N/N trials recorded with Wilson CI and per-tag breakdowns
- the version's `notes.md` records checkpoint, protocol, action horizon, camera preset, depth flag, result as X/N, session path

## Evaluation commands

| Command | What it does |
|---|---|
| `python logic/inference/protocols/generators/make_<protocol>.py` | Write `<protocol>.json` from a deterministic placement design and render it |
| `python -m vla.logic.inference.protocols.render_protocol <protocol>` | `renders/<protocol>_overview.png` (per-trial grid) + `_positions.png`; both protocol schemas |
| `python -m vla.logic.inference.protocols.protocols edit <protocol> [--base_image]` | Placement editor on the top-camera frame `base_frame.png`; click to place, arrows to navigate/rotate, `S` save |
| `python -m vla.logic.inference.protocols.protocols verify <protocol>` | Distribution check of a legacy-schema protocol: heading, coverage, per-cup scatter |
| `python -m vla.logic.inference.eval_engine --checkpoint=<spec> --protocol=<p> [flags]` | Protocol session. `<spec>`: `<step>`, `<N>k`, `last`, `v<NN>-<step>` or a `pretrained_model` path |
| `  --task="<text>"` | One prompt for every trial; default is the protocol's per-trial or global prompt |
| `  --resume=True` · `--start_trial=<i>` | Continue the latest matching session · start at trial `i` |
| `  --enable_rtc` · `--fps` · `--max_relative_target` · `--robot_id` | Same meaning as in block 7 |
| `python -m vla.logic.inference.eval_helpers ls [--detailed]` | Sessions of a version: checkpoint, protocol, success rate with CI, date |
| `python -m vla.logic.inference.eval_helpers info <session> [--group_by=<tag>]` | Breakdown by zone and protocol tags, step counts, failures, notes. `<session>`: `latest`, `<step>`, `v<NN>-<step>`, session id or path |
| `python -m vla.logic.inference.eval_helpers play <session> <trial_id>` | Open a trial video |
| `python -m vla.logic.inference.eval_render heatmap <session> [--output]` | Duck-position success heatmap, all trials + per target color |
| `python -m vla.logic.inference.eval_render grid \| pooled <session> <session> … [--output]` | Heatmaps of several sessions side by side · pooled into one |

Protocols shipped: `checkpoint_sweep`, `checkpoint_sweep_no_ood`, `dual_cup_15/30/60`, `id_scale_60`, `id_smoke_15`, `id_single_cup_30`, `ood_smoke_15`, `ood_single_cup_30`. Full reference: `docs/modules/inference/`.
