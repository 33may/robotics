"""The eyes-tier agents: one loop (`vlm_agent.VlmAgent`), several specs.

Each module is a variant — its rules text, its appendix vocabulary, a spec,
and a thin entry function. The loop itself lives in `vlm_agent.py` and no
variant may rebuild it (Anton 2026-08-27). AGENTS.md in this package is the
recipe for adding a variant.
"""
