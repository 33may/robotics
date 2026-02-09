# Project

Simulation-Driven Robotics Learning Framework

---

## 1. Introduction

- What is the one-sentence description of this project?

    **Develop framework for simulation driven robotics learning**

- What problem are we solving?

    **We already have the solution using the classic BC that achieve 80% accuracy, scaling above these accuracy takes more and more effort. So the goal is to solve this by designing the environment where we could employ computations to allow models to continiously learn new skills and solve tasks in simulated physical worlds.**

- What is the expected outcome/deliverable?
    - **The main outcome is the working model as proof of concept, that allows to evaluate the efficiency and potential of this method.**
    - **The secondary outcome is to develop the complete pipeline that allows to generate environments, create sim-ready assets and collect data, to use it for both BC and RL training phases.**

---

## 2. Background and Motivation

- Why are we doing this project now?

    **I believe that scaling intellegent robotics systems in the traditional way or in the way we approached other systems like text and vision is not feasible, due to the absence of the data required to train them. The concept of foundational models for robotics, that are moving in open source allows us to achieve the generalist robots. Then once wee have the robot that understand how to move, but might be bit clumsy or immature in the task, then we could use RL to enable generalist freshmen robot to become PhD level in the task we designed in simulation. This way in the bright future we don't need to record data with BC, since the pre-training of the robotics model is already done by the foundation model provider, and the goal is to actually train the robot to move in our own created environment,**

    **Once we have covered motivation for RL, the motivation for this approach came from exploration of the so called World Models that seems to be the research direction of all the labs that involved in the embodied AI. The idea here that we eventually don't have data that might be directly used to train the robot (like web text or images). However, we have enough text and video information to come up with the model that understand what the world is (including physics) and could generate the next frame for given input, this way we could generate data and use it for training, solving the problem of less amounts of data, by making it potentially infinite.**

    **The other approach to the simulated environment (which is going to be used more in this project) is the idea of representing 3D scenes using Gaussian Splatting. This is technology that allows to reconstruct realistic 3D scene from photos or video, also there is the MarbleLabs that have the generative model, that could generate this kind of worlds, also well known MetaAI SAM model got and updated and presented SAM3D, so looks like both the industry and academia is looking towards these tools.**

    **So with all this set, the motivation is to become the early adopters of the technologies that might become the industry standard in the short future.**

- What is the current state (the 80% BC accuracy baseline)?
- What are the limitations of the current approach?
- What makes simulation-based training the right solution?

---

## 3. Goals

- What is the primary goal of this project?

    **Verify the proposed solution works and improve the model performance while being feasible to implement.**

- What are the secondary goals?
    - **Develop the pipelines to create Digital Twins and Sim Ready assets**
    - **Explore how we could inject randomness inside the simulation**
- How will we measure success? (quantitative metrics)

    **We record the task success rate with just BC part, then apply the RL strategy and evaluate after it is trained on the real robot. The difference in success rates is seen as the pure method gains.**

- What performance improvement are we targeting?

    **Firstly we will focus on the task success rate, then we explore how we could improve variability in unknown scenarios.**

---

## 4. Technical Approach

- What is our overall technical strategy?

    **The goal is to go the whole cycle of development, starting from the BC using lerobot, then moving to the 3D envs and assets creation with GS tools, after that merging everything together inside the NVIDIA IsaacSim. so we could run the pre-trained model inside the simulation with the same 3D twin env that it was trained on. At this moment we expect that the model perform roughly the same as in real world. After that we use this scene with created reward function inside the NVIDIA IsaacLab to duplicate this env and run parallel training. In the process of training, maximizing the reward function, the model will execute the task better, hence the task success rate should increase. After that we try to use this model in real world, maybe it will require some sort of alignment to compensate sim2real transfer.**

- Why NVIDIA IsaacSim/IsaacLab specifically?

    **NVIDIA is seen as the leading commercial and research institution working towards the embodied AI, the set of tools they develop cover all of the tasks of this project, keeping NVIDIA native stack ensures that we won't have too many inconsistencies during the phase shifts, where we move from one project state to another**

- What are the key technical phases? (digital twin creation, training, sim2real, etc.)
    - BC pre-training

**At this state we collect the data using leader arm, record the dataset in real world and apply imitation learning to get decent results for the model to perform task in real world. At this phase we do our first evaluation, counting what is task success rate for the BC trained policy.**

- Digital Twin Creation

**This phase focuses on exploring tooling, software and methods to create simulation environment, this includes using real images for 3D reconstruction and also creating physically accurate assets that need to be interactable inside simulation.**

- Training inside simulation

**At this step we already have the digital world that behave the very similar to the real, we start the parallel training with weights updates based on the reward from the function. As the train goes we expect to see reward increasing, and if reward is designed properly, this will also improve the task success rate.**

- Sim2Real

**After we have the mode that works perfect inside the simulated worlds, we want to deploy it in real world, for that we need to explore how the inference will be affected by transferring to the real world. At this stage we might get back to the previous state, train models with different world parameters, gravity, joint configurations, camera settings etc., to cover larger distribution of worlds and increase the probability that real world is contained inside these.**

- What pre-trained model will we start with?
- How will we design the reward function?
- What sim2real transfer techniques will we use?

---

## 5. Scope

- What is in scope for this project?
    - Training and evaluating models with BC (Phase 1)
    - Collecting data for BC (Phase 1)
    - Creating 3D reconstructions (Phase 2)
    - Creating Sim Ready assets in CAD (Phase 2) +=
- What is explicitly out of scope?
- What hardware/software do we need?
- What data/models do we need access to?
- What are the technical requirements for the digital twin?

---

## 6. Risks

- What could go wrong technically?
    - We might not achieve the visually accurate 3D simulation
    - The action space might be too large for RL to train effecttively
    - The model might perform dramatically different in real world compared to simulation
    - Even after whole pipeline executed without problems, the results might be indifferent or worse then just BC setup
- What are the risks with sim2real transfer?
    - Camera inputs might be different quality for Sim vs Real
    - The world physics including object mass, softness might be different from Real world
    - The motion dynamics of the robot might be different in simulation
- What if the simulation doesn't match reality well enough?
    - Either improve simulation quality or run RL training with existing quality and verify if it is possible to compensate it with some additional alignment in Sim2Real Phase.
- What are the computational/time constraints?
    - ?
- What are our backup plans?
    - No backup all this is possible, so just moving forward towards it.

---

## 7. Timeline & Milestones

**Timeline:** February 1, 2026 – June 30, 2026

---

### 1. Phase 1: Behavior Cloning (BC) & Baseline Establishment

**Dates:** February 3 – February 20

**Goal:** Establish a functional hardware baseline and record initial performance metrics.

- **Hardware Setup:** Configure the SO-101 robot arm and ensure it is operational for data collection.
- **Data Collection:** Collect real-world dataset for the specific task using a leader arm.
- **Model Training:**
    - Train the initial model using ACT or VLA architectures.
    - Utilize `lerobot` for the Behavior Cloning process.
- **Evaluation:** Evaluate the model in the real world to establish the "80% accuracy" baseline and record the specific success rate.

### 2. Phase 2: Digital Twin Creation (The "Easy Scene")

**Dates:** February 21 – March 15

**Goal:** Construct a high-fidelity Digital Twin using Gaussian Splatting and import it into the simulation engine.

- **Scene Analysis:** Classify scene components into environment, assets, lighting, and physics properties.
- **Environment Reconstruction:**
    - Create the Gaussian Splatting (GS) environment from real images/video.
    - Post-process and clean the environment data.
    - Generate meshes for the GS environment to allow for collision and interaction.
- **Asset Integration:**
    - Create or generate "Sim Ready" assets (using CAD or generative tools like MarbleLabs/SAM3D).
    - Set up lighting and physics properties (mass, softness, friction) to match reality.
- **Simulation Setup:**
    - Load the robot model and configure joints within NVIDIA IsaacSim.
    - Configure virtual cameras to match real-world inputs.
    - **Milestone:** Verify the Digital Twin behaves reasonably similar to the real world.

### 3. Phase 3: Simulation Training (RL)

**Dates:** March 16 – March 30

**Goal:** Use Reinforcement Learning to improve the pre-trained model inside the simulated environment.

- **Reward Engineering:** Define a reward function that encourages task completion and efficiency.
- **Parallel Training:** Run training in NVIDIA IsaacLab, iterating until the train results looks good.
- **Metric Recording:** Record the task success rate metrics within the simulation to compare against the BC baseline.

### 4. Phase 4: Sim2Real Transfer & Validation

**Dates:** April 1 – April 15

**Goal:** Deploy the simulation-improved model to the physical robot and validate performance gains.

- **Deployment:** Run the optimized model on the real physical robot.
- **Visual Validation:** Verify that camera inputs in the real world are reasonably similar to the simulation inputs.
- **Final Evaluation:**
    - Record final task success rate metrics.
    - Calculate the "pure method gains" (difference between Phase 1 BC baseline and Phase 4 RL results).
- **Pipeline Review:** Evaluate the entire workflow. If successful, proceed to Phase 5. If not, analyze failures (e.g., physics mismatch, lighting issues) and fix.

### 5. Phase 5: Infrastructure Scaling (Real Data)

**Dates:** April 16 – June 30

**Goal:** Solidify the pipeline into a reusable infrastructure for future projects driven by the real tomato project.

- **Infrastructure Build:** Reuse scripts and tools from the exploration phases (1-4) to build a robust robotics solution infrastructure.
- **Optimization:** Shift from training scratch BC models to utilizing already pre-trained models from the collected data or foundation models as the starting point.
- **Deliverable:** A working model (Proof of Concept) and a complete pipeline for generating environments and training assets.

---

## 8. Technical Stack & Resources

- **Simulation Engine:** NVIDIA IsaacSim & IsaacLab.
- **3D Reconstruction:** Gaussian Splatting (GS) tools, potential use of MarbleLabs or SAM3D.
- **Training Framework:** LeRobot for BC; Custom RL loop in IsaacLab or LeRobot IsaacArena.
- **Hardware:** SO-101 Robot Arm, Camera setups, later the tomato Robot.

---

## 10. Research & Publication Plan

Answer these questions to plan the research contribution:

### Key Questions

- What is the research contribution/novelty?
- What results do we need to achieve for publication?
- What conferences/journals are we targeting?
- What is the publication timeline?
- Who will be the authors?
