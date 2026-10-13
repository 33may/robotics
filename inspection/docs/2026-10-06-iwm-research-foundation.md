# IWM: research foundation

IWM is the working name for the idea and the data structure described here.

## 1. Introduction

We are developing an inspection system for any object, built as an agentic harness
with three parts: perception, decision making and execution. In phase 1 the decision
maker chose camera views from a fixed grid of 36 directions around the object's
centroid, because a language model can understand such a grid natively.

The model that decides is blind. It knows the object only through text reports written
by a vision agent, and the system keeps no internal state of the world it has observed.
Each capture is read once, described in words, and the rest is left on disk.

The goal of this research is the missing piece: a data structure, created and
maintained from images and 3D data, that holds what the robot has seen and that the
model can work with directly. In short, the context a robot maintains and uses to make
decisions.

The need is not specific to inspection. A manipulator and a mobile robot face the same
question, so the structure is developed as its own module and is also judged by whether
it would serve them.

## 2. What phase 1 showed

**What the planner knows about space today.** Per view: a prose finding, one line of at
most 72 characters on what was seen, and a movement phrase. Per turn: which grid cells
are visited, the status of the box faces, and two counts.

**What is captured and never used for decisions.** Camera pose, depth, object mask, the
accumulated coloured point cloud, the fitted box, and every pixel box the vision agent
detected or cropped.

**What the planner asked.** In 25 recorded runs the planner put 100 questions to the
vision agent. Every one was answered from a single image.

| Count | Runs | Kind of question | Example, verbatim |
|---|---|---|---|
| 52 | 23 | Read what is there | "Read all text visible on the top sticker verbatim, including any small print, model numbers, or words below 'AI KIT'." |
| 23 | 11 | Re-read or check a reading | "Is the vertical text 'VATI' or something else (e.g. 'YAHBOOM', 'WAVESHARE')?" |
| 8 | 7 | Is there anything else on this surface | "Is there any other printing, text, arrows, or handwriting on the top face or the visible edges besides the AI KIT sticker?" |
| 6 | 6 | Where is it, and which way to orbit | "Which direction should the camera orbit (slightly/half left or right) to view that barcode label head-on?" |
| 6 | 3 | Which captured view shows it | "Is a white rectangular barcode/product label visible on any side panel in this view? If yes, which side of the frame is it on?" |
| 3 | 1 | Several objects | "Describe both objects: the red sphere and the blue box." |
| 1 | 1 | Is this view good enough | "Is the VBTI sticker fully inside the frame, unclipped, and sharp enough that each character is unambiguous?" |
| 1 | 1 | Is it the same thing | "If what you took for a sticker is actually the printed logo on the top edge of the chassis, say so." |

These questions are the starting point. The idea behind the IWM is that they are better
answered from a model built over all the views than from one image at a time.

The planner's plans and final answers also show it working out, without help, which way
each face points, whether the object is covered, and whether two sightings are the same
thing. That is read from its own text; it did not ask these as questions.

## 3. Main question

How do we build a data structure that compresses 3D information and lets the agent
recall what it has seen of the physical world?

## 4. Research questions

**A. What a robot needs to ask**

1. What exact questions does an inspection robot need to ask to plan its next action?
2. What general questions does a universal manipulator need answered by the world?
3. What questions does a mobile robot ask?

**B. The structure**

4. What is the atomic block of information?
5. How can hierarchy let a general model reason at different granularities?
6. How is the state updated at runtime without drifting?
7. Who is responsible for managing the representation and resolving conflicts and
   drift?
8. What is the contract of the structure: what information flows into it, and what
   flows out of it?
9. When raw observations (images, depth, point clouds) are compressed into the
   structure, what must be preserved so that the robot can still answer the questions
   it needs for its next action, and what can be discarded or left outside as raw data
   to be fetched on demand?
10. How should the structure represent how well something is known (glimpsed from far
    away versus examined up close) and what has not been observed at all, so that the
    robot can tell "it is not there" apart from "I have not looked"?
11. When the same physical thing is observed several times, from different viewpoints
    and with imperfect localisation, how does the structure recognise it as one entry
    instead of storing it twice, and how can a wrong merge be detected and undone?
12. How does one structure link the different kinds of information about the same thing
    (its geometry, its appearance in images, its meaning in words), so that a query
    made in one kind can be answered in another: asking in words and receiving a place,
    or pointing at a place and receiving what is known about it?
13. In which frame of reference should information be stored (the world, the object,
    the robot, the viewpoint it was seen from), and how is it translated into the point
    of view of whoever asks, given that the robot and its sensors keep moving?

These questions concern the data structure. How the robot acts on what it recalls is a
question about the system and is not part of this list.

## 5. Scope

- **Perception and decision making only.** Execution, such as whether the arm can reach
  a view, is outside the IWM.
- **Own module.** Independent of the inspection loop.
- **One object for now.** Every decision is checked against a scene with several
  objects, which comes later.
- **Research first.** Nothing is wired into the inspection loop yet. Tools and the
  structure come first; the agent on top is a description of how to use the tools.
- **The model refers, code computes.** The model names things; geometry stays in code.
- **No camera images for the planner.** Derived visuals such as diagrams, top views and
  feature maps may be added later.
- **Static world.**
- **Unchanged from phase 1:** the robot and camera, classical motion planning, no model
  training, the agent loop.
- **Judged by eye** while it is developed. An offline benchmark follows once a design
  has settled.
- **Time:** one to two months.

## 6. Working picture

Where the idea stands after the literature review of 2026-10-06. A direction, not a design.

- **An object graph.** The scene-graph idea turned inward: the object plays the scene,
  and its details play the objects.
- **One tree, with the voxels inside it.** A tree of voxel cells on an octree base
  (OctoMap is the likely base). At the top the object is a coarse blob. Opening a region
  goes one level deeper, where the voxels are denser and carry richer detail. Each cell
  node carries meaning, not only occupancy.
- **Where the levels come from.** From segmentation. All masks of a view are taken, and
  code or the agent places them in the tree, using which mask contains which and the
  tree that already exists. This step needs no training (OmniSeg3D, section 3.1).
  Whether it works on our objects is checked by experiment.
- **Who works on it.** An IWM agent that is a vision-language model and sees the frames.
  It reasons over the structure and maintains it. The orchestrator stays blind and asks
  the IWM agent.
- **How it is read.** Through tool calls, in the manner of GraphPad: look at the
  structure, find what is missing, look closer, write the result back.

## 7. What we know going in

The papers are kept in `docs/research/iwm/papers/`, the literature map in
`docs/research/iwm/passes/pass1/`.

**From the literature.**

- We found no published structure whose nodes are the surface regions or features of a
  single object, maintained across views and read by a language model without
  training. Existing work is at room scale or needs training.
- How well each element was observed (angle, resolution, from which view) is never
  exposed to a language model. Classical inspection planning computes it and keeps it
  in code.
- One direct comparison of how to hand a structure to a model (MIT, arXiv 2510.16643):
  on a small graph, pasting it into the context scores 0.75 against 0.85 for query
  tools. On a large graph the paste falls to 0.33 and queries hold at 0.77.

**From our own recordings.**

- The same physical point seen from two views lands about 10 mm apart (19 mm at the
  90th percentile, over 22 view pairs). Each view is shifted as a whole: once that
  shift is removed, 1.6 mm remains.
- The fused point cloud hides this, because averaging blends the shifted copies.
- With today's poses a voxel means the same place across views only at 15 to 20 mm, and
  position alone separates two features only when they are about 3 cm apart.
- The code to go from a pixel to a 3D point and back already exists.

## 8. Open

- **A yardstick.** Candidate: the structure is right when it answers the questions of
  group A, and compressed when it does so with less than the raw data.
- How parts that have not been seen are represented, when their shape is unknown.
- How a view is offered to the model.
- Whether relations between things are stored or derived on demand.
- What exactly makes a child node, and what a level means.
- How a mask in a new view is recognised as an existing node or a new one.
- Who names what the segmentation model looks for.
- How well a node was seen, in a form the agent can read.
