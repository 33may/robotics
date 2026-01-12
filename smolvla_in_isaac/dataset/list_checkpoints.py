"""Script to list available checkpoints."""

from pathlib import Path

def list_checkpoints():
    train_dir = Path("outputs/train")

    if not train_dir.exists():
        print(f"Training directory not found: {train_dir}")
        return

    print("="*60)
    print("AVAILABLE CHECKPOINTS")
    print("="*60)

    for run_dir in sorted(train_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            continue

        print(f"\nRun: {run_dir.name}")
        print("-" * 60)

        checkpoints = []
        for ckpt in sorted(checkpoint_dir.iterdir()):
            if ckpt.is_dir() and ckpt.name.isdigit():
                checkpoints.append(ckpt.name)

        if checkpoints:
            for ckpt_name in checkpoints:
                ckpt_path = checkpoint_dir / ckpt_name
                pretrained_path = ckpt_path / "pretrained_model"

                if pretrained_path.exists():
                    print(f"  ✓ Step {ckpt_name}: {pretrained_path}")
                else:
                    print(f"  ✗ Step {ckpt_name}: pretrained_model not found")

            # Check for 'last' symlink
            last_link = checkpoint_dir / "last"
            if last_link.exists():
                target = last_link.resolve().name
                print(f"\n  → 'last' points to: Step {target}")
                print(f"    Path: {last_link / 'pretrained_model'}")
        else:
            print("  No checkpoints found")

    print("\n" + "="*60)
    print("To test a checkpoint, run:")
    print("python smolvla_in_isaac/simulation_learning/test_act_policy.py \\")
    print("  --checkpoint outputs/train/act_so101_test/checkpoints/last/pretrained_model \\")
    print("  --num_episodes 10")
    print("="*60)

if __name__ == "__main__":
    list_checkpoints()
