#!/usr/bin/env python3
"""Is this machine ready to run the local verb models?

Every check here corresponds to a failure that has actually happened to
somebody (researched 2026-08-21), not to a theoretical one:

- `torch.cuda.is_available()` is NOT sufficient on Blackwell: CUDA 12.8/12.9
  shipped without `libnvptxcompiler.so`, so runtime PTX JIT can fail while
  ahead-of-time kernels work. We run a real bf16 matmul on the device.
- `Sam3Model` needs transformers >= 5.0.
- `facebook/sam3` is a GATED repo: the weights only resolve after the license
  has been accepted by the logged-in account, and the failure at load time is
  a 401 deep inside a download, which is easy to misread as a network fault.

Run: p inspection/eyes/env_check.py
"""
SM_BLACKWELL = (12, 0)
MIN_TRANSFORMERS = (5, 0)
SAM3_REPO = "facebook/sam3"


def check_env():
    """Returns a dict of check -> (ok, detail). Never raises."""
    out = {}

    try:
        import torch
        out["torch"] = (True, torch.__version__)
        out["cuda"] = (torch.cuda.is_available(),
                       f"build {torch.version.cuda}, "
                       f"{torch.cuda.device_count()} device(s)")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            out["capability"] = (cap == SM_BLACKWELL,
                                 f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
            try:
                a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
                v = float((a @ a).float().abs().sum())
                out["matmul"] = (v > 0, f"bf16 512x512 on device -> {v:.3g}")
            except Exception as e:                       # PTX JIT / driver
                out["matmul"] = (False, f"{type(e).__name__}: {e}")
        else:
            out["capability"] = (False, "no cuda device")
            out["matmul"] = (False, "skipped")
    except ImportError as e:
        out["torch"] = (False, str(e))
        out["cuda"] = out["capability"] = out["matmul"] = (False, "no torch")

    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        out["transformers"] = (v >= MIN_TRANSFORMERS,
                               f"{transformers.__version__} "
                               f"(need >= {MIN_TRANSFORMERS[0]}.{MIN_TRANSFORMERS[1]})")
    except ImportError as e:
        out["transformers"] = (False, str(e))

    try:
        from huggingface_hub import get_token
        tok = get_token()
        out["hf_token"] = (bool(tok), "present" if tok else "run: hf auth login")
    except ImportError as e:
        out["hf_token"] = (False, str(e))

    # Must FETCH A FILE, not just read metadata: facebook/sam3 is
    # `gated: manual`, and its metadata resolves for anyone — only the
    # download 401s. Checking model_info() alone reports a green gate on a
    # machine that cannot pull a single weight.
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(SAM3_REPO, "config.json")
        out["sam3_gate"] = (True, f"{SAM3_REPO} config fetched")
    except Exception as e:
        hint = ("accept terms at huggingface.co/" + SAM3_REPO
                if "Gated" in type(e).__name__ else str(e)[:60])
        out["sam3_gate"] = (False, f"{type(e).__name__} — {hint}")

    return out


def main():
    res = check_env()
    width = max(len(k) for k in res)
    for name, (ok, detail) in res.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    ready = all(ok for ok, _ in res.values())
    print(f"\n{'READY' if ready else 'NOT READY'} for the local verb stack")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
