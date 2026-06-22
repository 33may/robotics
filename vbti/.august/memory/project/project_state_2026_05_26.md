---
name: project-state-2026-05-26
description: Current graduation-table framing of the robotics project state as of 2026-05-26
metadata: 
  node_type: memory
  type: project
  originSessionId: 70f909db-3205-4b3f-9a48-3300eaeef3e7
---

The project is currently framed as a robotics research trajectory inside a company context, not just a company delivery task. It started with building a complete robotics pipeline: 3D scene reconstruction, composition into a physics engine, real teleoperation data collection, training in simulation, and inference in simulation.

After that, the work moved toward evaluating model performance on top of this pipeline, but the available baseline model was not strong enough for the task. This shifted the work into building a stronger baseline using modern VLA architectures with a more classical imitation-learning setup.

As of 2026-05-26, the model has reached 100% success rate on the lab evaluation task after several engineering and training decisions. The core task is therefore basically solved, but this creates a new research problem: harder evaluation settings, downsampled datasets, sim-data comparisons, and new architectures like UVA are needed to measure real improvement.

**Why:** This is the story to use for the graduation table meeting and assignment framing: infrastructure first, then baseline establishment, then research evidence around data efficiency, evaluation, and scalability.

**How to apply:** When discussing the project with university assessors or preparing submission material, present the current state as: full robotics pipeline built → weak baseline discovered → stronger VLA/IL baseline built → 100% lab success achieved → now validating generalization/data efficiency and preparing UVA/simulation/humanoid next steps. Link with [[project-robotics-reframe-hbo-to-research]] and [[project-smolvla-uva]].
