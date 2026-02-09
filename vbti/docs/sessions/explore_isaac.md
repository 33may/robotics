
Task: Explore codebase structure and draw architecture diagrams
Goal: Understand manager-based vs agent-based approaches in Isaac Lab

---
Questions to Answer for Task Completion

You need to answer these questions by exploring the codebase yourself. When you can answer all of them confidently, you've earned the
completion.

---
Part 1: Directory Structure (Foundation)

1. What are the main directories under /home/may33/isaac/IsaacLab/source/? What does each appear to contain?

tree -L 1
.
├── isaaclab
├── isaaclab_assets
├── isaaclab_mimic
├── isaaclab_rl
└── isaaclab_tasks

**isaaclab** contains the main Isaac Lab codebase.

**isaaclab_assets** contains various assets used in Isaac Lab, like objects and robots. 

**isaaclab_mimic** don't know

**isaaclab_rl** contains RL algorithms to train models

**isaaclab_tasks** contains the task definitions for both manager and direct workflows, including configs and mdps

2. Where do environment definitions (tasks) live? What's the path?

in **isaaclab_tasks** there are definitions for both manager based and direct

3. Where are robot configurations (articulations) defined?

in **isaaclab_assets**

---
Part 2: Manager-Based Architecture

4. Find a manager-based environment (hint: look for ManagerBasedEnv or ManagerBasedRLEnv). What file did you find?

We examine stack manager env, in there we see the 

5. What are the different managers in a manager-based env? (List at least 5 types of managers)
6. How does a manager-based env define observations? What class/config handles this?
7. How does a manager-based env define rewards? Show me the pattern.
8. What is the scene configuration and what does it contain?

---
Part 3: Direct/Agent-Based Architecture

9. Find a direct-style environment (hint: look for DirectRLEnv). What file did you find?
10. How does a direct env differ from manager-based? What methods must you implement yourself?
11. What are the trade-offs between manager-based and direct approaches? (Think: flexibility vs convenience)

---
Part 4: Training Integration

12. Where are the RL training scripts located? What algorithms are supported?
13. How does an environment get registered so training scripts can find it?

---
Part 5: Diagrams (Deliverable)

14. Draw a diagram showing the manager-based architecture with all manager types and their relationships
15. Draw a diagram showing the direct env architecture and what you implement yourself
