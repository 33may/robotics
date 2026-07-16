"""Warehouse dressing pipeline (MAY-173 locdev) — anti-aliasing visual landmarks.

Generates unique number plates + VBTI posters as textured quads in a wrapper USD
that references the untouched NVIDIA Simple_Warehouse scene. Pure-python parts
(texture gen, layout math) run in the `hum` env; USD bake + QA renders run in
`isaac`.
"""
