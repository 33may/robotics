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

from vbti.logic.train.config_utils import TrainConfig, ModelType, DatasetSource
from vbti.logic.train.experiment_utils import (
    create_version, complete_version, log_metrics, save_summary,
    get_version_dir, active, use, load_config, list_checkpoints,
)
from vbti.logic.train.monitor import update_status
from vbti.logic.dataset.loading_utils import resolve_dataset_source


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


def _find_latest_step_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Find the highest step_XXXXXX checkpoint that has training_state.pt."""
    candidates = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("step_") and d.name[5:].isdigit():
            if (d / "training_state.pt").exists():
                candidates.append((int(d.name[5:]), d))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _check_existing_checkpoints(checkpoints_dir: Path, resume: bool) -> bool:
    """Check for existing checkpoints and prompt user if not resuming.

    Returns True if training should proceed, False to abort.
    """
    ckpts = [d for d in checkpoints_dir.iterdir()
             if d.is_dir() and (d / "model.safetensors").exists()]
    if not ckpts:
        return True

    step_ckpts = [d.name for d in ckpts if d.name.startswith("step_")]
    named_ckpts = [d.name for d in ckpts if not d.name.startswith("step_")]
    step_ckpts.sort()

    print(f"\n⚠  Found existing checkpoints:")
    for name in step_ckpts + named_ckpts:
        has_state = (checkpoints_dir / name / "training_state.pt").exists()
        print(f"    {name}" + (" (resumable)" if has_state else ""))

    if resume:
        latest = _find_latest_step_checkpoint(checkpoints_dir)
        if latest:
            print(f"\n  Will resume from: {latest.name}")
            return True
        else:
            print(f"\n  No resumable checkpoint found (missing training_state.pt).")
            print(f"  Starting from scratch will overwrite step checkpoints.")
            resp = input("  Continue anyway? [y/N] ").strip().lower()
            return resp in ("y", "yes")
    else:
        print(f"\n  Starting fresh will train from pretrained base, not from these checkpoints.")
        print(f"  Use --resume to continue from the latest checkpoint instead.")
        resp = input("  Start fresh? [y/N] ").strip().lower()
        return resp in ("y", "yes")


def resolve_config_and_version(
    config: "TrainConfig | str | None" = None,
    experiment: str | None = None,
    version: str | None = None,
    notes: str = "",
) -> tuple["TrainConfig", str, Path]:
    """Shared helper: resolve config + experiment version → (cfg, version_id, version_dir).

    Two flows:
        1) config provided, no version  → create new version
        2) config provided, version given → use existing version (prompt if config differs)
        3) no config                    → load config from active/specified version

    Used by both train() and remote.train() so resolution logic stays in one place.
    """
    import yaml as _yaml

    if config is not None:
        if isinstance(config, (str, Path)):
            config = TrainConfig.load(config)

        if version:
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
                    raise SystemExit("Aborted.")
                config_path = get_version_dir() / "config.yaml"
                with open(config_path, "w") as f:
                    _yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
            version_id = version
        else:
            version_id = create_version(
                config=config.to_dict(),
                notes=notes,
                experiment=experiment,
            )
    else:
        if version:
            use(experiment or active()[0], version)
        config = TrainConfig.from_dict(load_config())
        _, version_id = active()
        if not version_id:
            raise ValueError("No active version and no config provided.")

    return config, version_id, get_version_dir()


def train(config: TrainConfig | str | None = None, experiment: str | None = None,
          version: str | None = None, notes: str = "", resume: bool = False,
          reset_lr: bool = False):
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
        resume: if True, resume from latest checkpoint (restores model + optimizer + step)
        reset_lr: if True (requires --resume), skip scheduler restoration and rebuild
                  a fresh cosine schedule over the remaining steps.
    """
    if reset_lr and not resume:
        raise ValueError("--reset-lr requires --resume")
    # ── Resolve config + version ──────────────────────────────────
    config, version_id, _ = resolve_config_and_version(config, experiment, version, notes)

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

    # ── Resolve datasets (auto-aggregate if multiple sources) ─────
    if len(config.dataset.sources) > 1:
        repo_id, root = resolve_dataset_source(
            config.dataset.sources, experiment=exp_name, version=version_id
        )
        # Replace sources with the single aggregated dataset
        config.dataset.sources = [DatasetSource(repo_id=repo_id, root=root, source="mixed")]

    # ── Check existing checkpoints ────────────────────────────────
    if not _check_existing_checkpoints(checkpoints_dir, resume):
        print("Aborted.")
        return None

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

    # ── Resume from checkpoint ─────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")

    if resume:
        latest_ckpt = _find_latest_step_checkpoint(checkpoints_dir)
        if latest_ckpt:
            start_step = backend.load_checkpoint_for_resume(
                bundle, latest_ckpt, optimizer, scheduler, reset_lr=reset_lr
            )
            if reset_lr and start_step > 0:
                remaining = cfg_t.steps - start_step
                if cfg_t.lr_schedule == "wsd":
                    from vbti.logic.train.backends.smolvla import SmolVLABackend
                    scheduler = SmolVLABackend._build_wsd_scheduler(optimizer, config)
                    # Advance scheduler to match resume position
                    for _ in range(start_step):
                        scheduler.step()
                    print(f"  Reset WSD schedule at step {start_step}, "
                          f"{remaining} steps remaining")
                else:
                    scheduler_preset = bundle.model.config.get_scheduler_preset()
                    scheduler = scheduler_preset.build(optimizer, remaining)
                    print(f"  Fresh cosine schedule: {remaining} steps, "
                          f"peak={cfg_t.lr:.1e} → floor={cfg_t.decay_lr:.1e}")

    # ── Training loop ─────────────────────────────────────────────
    remaining = cfg_t.steps - start_step
    batches_per_epoch = len(train_loader)
    total_epochs = cfg_t.steps / batches_per_epoch if batches_per_epoch > 0 else 0
    print(f"\nTraining for {cfg_t.steps} steps, batch_size={cfg_t.batch_size}")
    print(f"  {batches_per_epoch} batches/epoch, ~{total_epochs:.1f} epochs total")
    if start_step > 0:
        print(f"  Resuming from step {start_step} ({remaining} remaining)")
    print("=" * 60)

    step = start_step
    done = False
    losses = []
    epoch = step // batches_per_epoch if batches_per_epoch > 0 else 0
    start_time = time.time()

    while not done:
        epoch_step = 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}",
                    bar_format="{desc} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                    leave=False)
        for batch in pbar:
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
            epoch_step += 1
            pbar.set_description(f"Epoch {epoch} | Step {step}/{cfg_t.steps}")
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

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
                    backend.save_checkpoint(bundle, checkpoints_dir / "best", is_best=True,
                                            optimizer=optimizer, scheduler=scheduler, step=step)

            # ── Checkpoint ────────────────────────────────────────
            if step % cfg_l.save_freq == 0:
                backend.save_checkpoint(bundle, checkpoints_dir / f"step_{step:06d}",
                                        optimizer=optimizer, scheduler=scheduler, step=step)

            if step >= cfg_t.steps:
                done = True
                break

        pbar.close()
        epoch += 1
        if not done:
            elapsed_so_far = time.time() - start_time
            steps_this_session = step - start_step
            eta = elapsed_so_far / steps_this_session * (cfg_t.steps - step) if steps_this_session > 0 else 0
            print(f"  Epoch {epoch - 1} done | Step {step}/{cfg_t.steps} | ETA: {_format_duration(eta)}")

    # ── Final save ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    duration_str = _format_duration(elapsed)

    backend.save_checkpoint(bundle, checkpoints_dir / "final")

    summary = {
        "final_train_loss": losses[-1] if losses else None,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "total_steps": step,
        "epochs": round(step / batches_per_epoch, 1) if batches_per_epoch > 0 else 0,
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

def _train_cli(config: str = None, experiment: str = None, version: str = None,
               notes: str = "", resume: bool = False, reset_lr: bool = False):
    """Train from config YAML (new version) or from existing version.

    Examples:
        # New version from config:
        python -m vbti.logic.train.engine train --config=path/to/config.yaml --experiment=lift_cube

        # Run active version (already created, config already there):
        python -m vbti.logic.train.engine train

        # Resume from latest checkpoint:
        python -m vbti.logic.train.engine train --resume

        # Resume with fresh LR schedule over remaining steps:
        python -m vbti.logic.train.engine train --resume --reset_lr

        # Run specific existing version:
        python -m vbti.logic.train.engine train --experiment=lift_cube --version=v001
    """
    cfg = TrainConfig.load(config) if config else None
    return train(cfg, experiment=experiment, version=version, notes=notes,
                 resume=resume, reset_lr=reset_lr)


def _status():
    """Show active experiment and version."""
    exp, ver = active()
    print(f"Active: {exp}" + (f" / {ver}" if ver else ""))


def _build_lerobot_command(config: TrainConfig, output_dir: str, job_name: str) -> list[str]:
    """Translate our TrainConfig into a lerobot-train CLI command."""
    import json

    model_cfg = config.model
    dataset_cfg = config.dataset
    training_cfg = config.training
    logging_cfg = config.logging

    args = ["lerobot-train"]

    # Policy / model
    args.append(f"--policy.path={model_cfg.pretrained}")

    # Dataset
    repo_id = dataset_cfg.sources[0].repo_id
    args.append(f"--dataset.repo_id={repo_id}")
    root = dataset_cfg.sources[0].root
    if root:
        args.append(f"--dataset.root={root}")

    # Training
    args.append(f"--batch_size={training_cfg.batch_size}")
    args.append(f"--steps={training_cfg.steps}")
    args.append(f"--num_workers={training_cfg.num_workers}")

    # Optimizer
    args.append(f"--optimizer.lr={training_cfg.lr}")
    args.append(f"--optimizer.weight_decay={training_cfg.weight_decay}")
    args.append(f"--optimizer.grad_clip_norm={training_cfg.grad_clip_norm}")

    # Scheduler — map to lerobot's cosine_decay_with_warmup
    args.append("--scheduler.type=cosine_decay_with_warmup")
    args.append(f"--scheduler.num_warmup_steps={training_cfg.warmup_steps}")
    args.append(f"--scheduler.num_decay_steps={training_cfg.steps}")
    args.append(f"--scheduler.peak_lr={training_cfg.lr}")
    args.append(f"--scheduler.decay_lr={training_cfg.decay_lr}")

    # Policy config
    args.append(f"--policy.chunk_size={model_cfg.chunk_size}")
    args.append(f"--policy.n_obs_steps={model_cfg.n_obs_steps}")
    args.append(f"--policy.freeze_vision_encoder={str(model_cfg.freeze_vision_encoder).lower()}")
    args.append(f"--policy.train_expert_only={str(model_cfg.train_expert_only).lower()}")
    args.append(f"--policy.train_state_proj={str(model_cfg.train_state_proj).lower()}")
    args.append(f"--policy.tokenizer_max_length={model_cfg.tokenizer_max_length}")

    # Camera alignment — build rename_map from config camera order
    pretrained_slots = ["camera1", "camera2", "camera3"]
    camera_names = dataset_cfg.cameras.names or []
    rename_map = {}
    n_empty = 0
    for i, cam in enumerate(camera_names):
        src = f"observation.images.{cam}"
        if i < len(pretrained_slots):
            dst = f"observation.images.{pretrained_slots[i]}"
        else:
            dst = f"observation.images.empty_camera_{n_empty}"
            n_empty += 1
        if src != dst:
            rename_map[src] = dst

    if rename_map:
        args.append(f"--rename_map={json.dumps(rename_map)}")

    empty_total = max(model_cfg.empty_cameras, n_empty)
    args.append(f"--policy.empty_cameras={empty_total}")

    # Output
    args.append(f"--output_dir={output_dir}")
    args.append(f"--job_name={job_name}")
    args.append(f"--policy.repo_id={job_name}")

    # Logging
    args.append(f"--log_freq={logging_cfg.log_freq}")
    args.append(f"--save_freq={logging_cfg.save_freq}")

    # W&B
    args.append(f"--wandb.enable={str(logging_cfg.wandb_enabled).lower()}")
    if logging_cfg.wandb_enabled:
        args.append(f"--wandb.project={logging_cfg.wandb_project}")
        if logging_cfg.wandb_entity:
            args.append(f"--wandb.entity={logging_cfg.wandb_entity}")
        if logging_cfg.wandb_mode:
            args.append(f"--wandb.mode={logging_cfg.wandb_mode}")

    return args


def _train_lerobot_cli(config: str = None, experiment: str = None,
                       version: str = None, notes: str = "", dry_run: bool = False):
    """Generate and run lerobot-train from our config.

    Examples:
        # From config file:
        python -m vbti.logic.train.engine train-lerobot --config=path/to/config.yaml --experiment=duck_cup

        # From active version:
        python -m vbti.logic.train.engine train-lerobot

        # Dry run — just print the command:
        python -m vbti.logic.train.engine train-lerobot --dry_run
    """
    import subprocess, shlex

    # Resolve config (same logic as train)
    if config is not None:
        cfg = TrainConfig.load(config) if isinstance(config, str) else config
        if version:
            use(experiment or active()[0], version)
        else:
            version_id = create_version(
                config=cfg.to_dict(), notes=notes, experiment=experiment,
            )
    else:
        if version:
            use(experiment or active()[0], version)
        cfg = TrainConfig.from_dict(load_config())

    exp_name, version_id = active()
    if not version_id:
        raise ValueError("No active version and no config provided.")

    version_dir = get_version_dir()
    output_dir = str(version_dir / "lerobot_output")
    job_name = f"{exp_name}_{version_id}"

    args = _build_lerobot_command(cfg, output_dir, job_name)
    cmd_str = " \\\n    ".join(args)

    print(f"Experiment: {exp_name} / {version_id}")
    print(f"Output: {output_dir}\n")
    print(cmd_str)

    if dry_run:
        print("\n(dry run — not executing)")
        return cmd_str

    print("\n" + "=" * 60)
    subprocess.run(args, check=True)


if __name__ == "__main__":
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        "train":  _train_cli,
        "train-lerobot": _train_lerobot_cli,
        "status": _status,
    })
