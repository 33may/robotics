"""
Training engine — the shared training loop for all model backends.

Usage:
    from vbti.logic.train.engine import train
    from vbti.logic.train.config_utils import TrainConfig

    config = TrainConfig.load("path/to/config.yaml")
    train(config, experiment="lift_cube_smolvla", notes="baseline run")
"""

import time
import torch
from pathlib import Path
from tqdm import tqdm

from vbti.logic.train.config_utils import TrainConfig, ModelType
from vbti.logic.train.experiment_utils import (
    create_version, complete_version, log_metrics, save_summary,
    get_version_dir, active, use, load_config,
)
from vbti.logic.train.monitor import update_status


def _get_backend(model_type: ModelType):
    """Lazy-load the correct backend."""
    if model_type == ModelType.SMOLVLA:
        from vbti.logic.train.backends.smolvla import SmolVLABackend
        return SmolVLABackend()
    elif model_type == ModelType.GROOT:
        from vbti.logic.train.backends.groot import GR00TBackend
        return GR00TBackend()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(config: TrainConfig | str | None = None, experiment: str | None = None,
          version: str | None = None, notes: str = ""):
    """Run a training experiment.

    Two flows:
        1) New version: pass config + experiment + notes → creates version, then trains
        2) Existing version: pass experiment + version (or have them active) → loads
           config from that version's config.yaml, then trains

    Args:
        config: TrainConfig, path to YAML, or None to use active version's config
        experiment: experiment name (uses active if None)
        version: version to resume/use (creates new if None and config is given)
        notes: notes for new version (ignored if using existing)
    """
    # ── Resolve config + version ──────────────────────────────────
    if config is not None:
        if isinstance(config, (str, Path)):
            config = TrainConfig.load(config)

        if version:
            # Config provided + existing version → check for diff
            use(experiment or active()[0], version)
            existing = TrainConfig.from_dict(load_config())
            diff = config.diff(existing)
            if diff:
                print("Config differs from version's stored config:")
                for section, fields in diff.items():
                    for field, vals in fields.items():
                        print(f"  {section}.{field}: {vals['old']} → {vals['new']}")
                resp = input("Overwrite version config? [y/N] ").strip().lower()
                if resp not in ("y", "yes"):
                    print("Aborted.")
                    return None
                # Overwrite the version's config
                import yaml
                config_path = get_version_dir() / "config.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
            version_id = version
        else:
            # New version from config
            version_id = create_version(
                config=config.to_dict(),
                notes=notes,
                experiment=experiment,
            )
    else:
        # No config → use existing version's config
        if version:
            use(experiment or active()[0], version)
        config = TrainConfig.from_dict(load_config())
        _, version_id = active()
        if not version_id:
            raise ValueError("No active version and no config provided.")

    cfg_t = config.training
    cfg_l = config.logging

    # ── Setup ─────────────────────────────────────────────────────
    backend = _get_backend(config.model_type)
    version_dir = get_version_dir()
    checkpoints_dir = version_dir / "checkpoints"

    # ── Resolve experiment name ───────────────────────────────────
    exp_name, _ = active()

    print("=" * 60)
    print(f"TRAINING — {config.model_type.value.upper()}")
    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Version: {version_id}")
    print(f"Output:  {version_dir}")

    # ── W&B init ──────────────────────────────────────────────────
    wandb_run = None
    if cfg_l.wandb_enabled:
        import wandb

        # Check for existing run ID (resume support)
        run_id_file = version_dir / "wandb_run_id.txt"
        resume = None
        run_id = None
        if run_id_file.exists():
            run_id = run_id_file.read_text().strip()
            resume = "allow"

        wandb_run = wandb.init(
            project=cfg_l.wandb_project,
            entity=cfg_l.wandb_entity,
            group=exp_name,
            name=version_id,
            config=config.to_dict(),
            tags=[config.model_type.value],
            id=run_id,
            resume=resume,
            mode=cfg_l.wandb_mode,
        )
        run_id_file.write_text(wandb_run.id)
        print(f"W&B: {wandb_run.url or 'offline mode'}")

    # ── Load model ────────────────────────────────────────────────
    print("\nLoading model...")
    bundle = backend.load_model(config)

    # ── Dataloaders ───────────────────────────────────────────────
    print("Loading datasets...")
    train_loader, val_loader = backend.make_dataloaders(config, bundle.dataset_meta)
    print(f"Train batches: {len(train_loader)}, Val: {'yes' if val_loader else 'no'}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer, scheduler = backend.get_optimizer(bundle, config)

    # ── Training loop ─────────────────────────────────────────────
    print(f"\nTraining for {cfg_t.steps} steps, batch_size={cfg_t.batch_size}")
    print("=" * 60)

    step = 0
    done = False
    losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    while not done:
        for batch in tqdm(train_loader, desc=f"Step {step}", leave=False):
            # Train step
            loss, loss_dict = backend.train_step(bundle, batch, optimizer)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                bundle.model.parameters(), cfg_t.grad_clip_norm
            )
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            step += 1

            # ── Logging ───────────────────────────────────────────
            if step % cfg_l.log_freq == 0:
                avg_loss = sum(losses[-cfg_l.log_freq:]) / min(len(losses), cfg_l.log_freq)
                lr = scheduler.get_last_lr()[0]
                # Extract loggable metrics from loss_dict.
                # SmolVLA returns debug tensors (losses_after_forward etc) alongside
                # scalar "loss". We log scalars and reduce tensors to summary stats.
                clean_dict = {}
                for k, v in loss_dict.items():
                    if isinstance(v, (int, float)):
                        clean_dict[k] = v
                    elif hasattr(v, 'numel'):
                        if v.numel() == 1:
                            clean_dict[k] = v.item()
                        else:
                            # Summarize multi-element tensors (e.g. per-joint losses)
                            clean_dict[f"{k}_mean"] = v.mean().item()
                            clean_dict[f"{k}_max"] = v.max().item()
                metrics = {"train_loss": avg_loss, "lr": lr, **clean_dict}
                log_metrics(step=step, metrics=metrics)
                update_status(step=step, metrics=metrics)
                if wandb_run:
                    wandb_run.log({"train/loss": avg_loss, "train/lr": lr, **{f"train/{k}": v for k, v in loss_dict.items() if isinstance(v, (int, float))}}, step=step)
                print(f"Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # ── Validation ────────────────────────────────────────
            if val_loader and step % cfg_l.val_freq == 0:
                val_metrics = backend.validate(bundle, val_loader, cfg_l.val_size)
                val_loss = val_metrics["val_loss"]
                log_metrics(step=step, metrics=val_metrics)
                update_status(step=step, metrics=val_metrics)
                if wandb_run:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}, step=step)
                print(f"Step {step:5d} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    backend.save_checkpoint(bundle, checkpoints_dir / "best", is_best=True)

            # ── Checkpoint ────────────────────────────────────────
            if step % cfg_l.save_freq == 0:
                backend.save_checkpoint(bundle, checkpoints_dir / f"step_{step:06d}")

            if step >= cfg_t.steps:
                done = True
                break

    # ── Final save ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    duration_str = _format_duration(elapsed)

    backend.save_checkpoint(bundle, checkpoints_dir / "final")

    summary = {
        "final_train_loss": losses[-1] if losses else None,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "total_steps": step,
        "duration": duration_str,
        "duration_seconds": elapsed,
    }
    save_summary(summary)
    complete_version(status="completed", duration_str=duration_str)

    # ── W&B finish ────────────────────────────────────────────────
    if wandb_run:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Duration: {duration_str}")
    print(f"Final loss: {losses[-1]:.4f}" if losses else "No losses recorded")
    print(f"Best val loss: {best_val_loss:.4f}" if best_val_loss < float("inf") else "No validation")
    print(f"Checkpoints: {checkpoints_dir}")

    return summary


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _train_cli(config: str = None, experiment: str = None, version: str = None, notes: str = ""):
    """Train from config YAML (new version) or from existing version.

    Examples:
        # New version from config:
        python -m vbti.logic.train.engine train --config=path/to/config.yaml --experiment=lift_cube

        # Run active version (already created, config already there):
        python -m vbti.logic.train.engine train

        # Run specific existing version:
        python -m vbti.logic.train.engine train --experiment=lift_cube --version=v001
    """
    cfg = TrainConfig.load(config) if config else None
    return train(cfg, experiment=experiment, version=version, notes=notes)


def _status():
    """Show active experiment and version."""
    exp, ver = active()
    print(f"Active: {exp}" + (f" / {ver}" if ver else ""))


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "train":  _train_cli,
        "status": _status,
    })
