---
name: feedback-close-validation-loop
description: Always target closing the validation loop — make "working" a computable number — in any task we build
metadata:
  type: feedback
---

**Whatever we build, target closing the validation loop: make "working" a number the system can compute automatically, then build toward passing it.** Don't ship a feature/module/pipeline whose success is only eyeballed — give it a measurable pass/fail gate and a way to run it against ground truth.

**Why:** Anton's principle, crystallized 2026-07-10 during the localization eval design. The whole reason that plan was strong is it reframed "wire up RTAB-Map and hope" into "measure estimate-vs-GT meter error until it clears the threshold" — an agent (or a human) can only *iterate until working* when *working* is a computable score. A validation loop is what turns an open-ended build into something that converges.

**How to apply:** at the start of any build task, ask "what's the ground truth here and how do we score against it?" and design that loop in early — not as an afterthought. Isaac GT pose, a held-out dataset, a golden output, a threshold on a metric — whatever the domain's truth source is. Prefer building the harness/scorer *before* (or alongside) the thing being validated, and freeze the target so iteration optimizes a fixed goal. Applies broadly: nav/localization ([[project-localization-eval-harness]] — the eval-harness/GT-scorer loop), training/eval runs, sim pipelines, any autonomous agent loop. Related: [[feedback-show-dont-tell]].
