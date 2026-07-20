"""deploybench — interactive, per-run end-to-end localization deploy evaluation.

Sibling of `locbench` (the frozen accuracy oracle). Where locbench freezes a versioned
episode set and scores localization ACCURACY under GT-steered navigation, deploybench answers
the DEPLOY question for *any* localization algorithm, on whatever partial map a demo happened
to bake:

    spawn at an operator-chosen start (KNOWN or KIDNAPPED) → guide the robot to an
    operator-chosen goal → did it ARRIVE, and what are the localization statistics?

The eval course is chosen per run (an interactive `Scenario`, not a frozen episode set) —
demos cover only part of a scene, so the start/goal are picked on the baked map each time.
deploybench REUSES locbench's scoring spine (`pairs`, `stats`) and the `LocalizationModule`
seam; it never edits the locbench oracle. Pure numpy/stdlib at the core → `brain` env.
"""
