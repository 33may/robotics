# Masson, Ranchod, Konidaris — "Reinforcement Learning with Parameterized Actions"

AAAI 2016. arXiv:1509.01644 (v4). Full text obtained via arXiv HTML rendering, read in full (all sections: Introduction, Background/2.1 Parameterized Tasks, §3 Parameterized Action MDPs, §4 Theoretical Results, §5 Experiments, §6 Related Work, §7 Conclusion). Cross-checked quotes across three separate fetches for verbatim accuracy on definitions, theorem numbers, equations, and figure captions.

This is a **pure RL-theory paper with two toy physics-sim domains (2D robot-soccer goal-scoring, a "Platformer" game)**. It contains no vision, no meshes, no point clouds, no camera geometry, and no mention of view planning, aspect graphs, or object addressing of any kind. Its relevance to the branch is entirely at the level of **formal vocabulary**: it names and proves convergence properties for exactly the type signature the branch's contract already assumes (`A = {(k, x_k)}`), and nothing else. That scope-boundary is itself the main finding to carry forward.

## Core idea

A **Parameterized Action MDP (PAMDP)** generalizes standard MDPs by making each discrete action carry its own continuous parameter vector, rather than forcing a choice between a purely discrete action set (can't fine-tune) or a purely continuous action space (can't express "differences in kind"). Opening paragraph of the Introduction states the tradeoff directly:

> "If we use a continuous action space, we lose the ability to consider differences in kind: all actions must be expressible as a single vector. If we use only discrete actions, we lose the ability to finely tune action selection based on the current state." (§1, Introduction, opening paragraph)

## Formal definitions (§3, Parameterized Action MDPs)

- **Action space**: `A = ⋃_{a∈A_d} {(a,x) | x∈X_a}` — the discrete action set is `A_d = {a_1, a_2, ..., a_k}` (a finite, fixed, flat, unordered set), and each discrete action `a` has its own parameter space `X_a` (dimensionality can differ per action).
- **Policy factorization**: `π(a,x|s) = π^d(a|s)·π^a(x|s)` — a single agent policy samples the discrete action first, then samples parameters conditioned on state (and, in the algorithm, conditioned on the chosen discrete action). Both `π^d` and `π^a` belong to the *same* learned policy — there is no architectural split between "who decides the discrete part" and "who decides the continuous part."
- **Objective**: `J(θ,ω) = E_{s0~D}[V^{π_θ,ω}(s0)]`, jointly parameterized by discrete-policy parameters `ω` and parameter-policy parameters `θ`.
- **Induced discrete MDP**: `M_θ = ⟨S, A_d, P_θ, R_θ, γ⟩`, with transition/reward marginalized over the parameter policy: `P_θ(s'|s,a) = ∫_{x∈X_a} π^a_θ(x|s) P(s'|s,a,x) dx`, and likewise for `R_θ`. This is the "fix the continuous policy, get an ordinary discrete MDP to learn `ω` in" move that the Q-PAMDP algorithm exploits.
- **`H(θ) = J(θ, W(θ))`**, where `W(θ) = arg max_ω J(θ,ω) = ω*_θ` — "the performance of the best discrete policy for a fixed θ" (§3).

## Q-PAMDP algorithm (§4/Algorithm 1)

> "Algorithm 1 Q-PAMDP(k) — Input: initial θ₀, ω₀; P-UPDATE; Q-LEARN. ω ← Q-LEARN^∞(M_θ, ω₀); repeat: θ ← P-UPDATE^k(J_ω, θ); ω ← Q-LEARN^∞(M_θ, ω); until θ converges."

Two variants: **Q-PAMDP(1)** updates the parameter policy once per outer loop then fully relearns discrete Q-values to convergence; **Q-PAMDP(∞)** fully optimizes the parameter policy at each outer step. Both alternate between a continuous-parameter policy-search step and a discrete Q-learning step on the marginalized MDP `M_θ`.

## Theorems (§4, Theoretical Results — numbered exactly as printed)

- **Theorem 4.1** (Convergence to a Local Optimum): if `P-UPDATE` converges to a local optimum w.r.t. `H`, Q-PAMDP(1) converges to a local optimum w.r.t. the true objective `J`.
- **Theorem 4.2** (Bounded Updates to ω): continuity + step-size conditions bound how much `ω` moves per outer iteration.
- **Theorem 4.3** (Gradient of `H(θ)`): if `J` and `W` are differentiable, `∇_θ H(θ) = ∇_θ J(θ, ω*)` at `ω* = W(θ)` — licenses using policy-gradient methods for `P-UPDATE`.
- **Theorem 4.4** (Local Convergence of Q-PAMDP(∞)): with a global-optimization `P-UPDATE` step, Q-PAMDP(∞) converges to a local optimum.

All four theorems are about **learning dynamics** (does the alternating-optimization *procedure* converge), not about the *correctness* of a hand-specified, non-learned parameter function. This matters for the branch — see "What does NOT transfer" below.

## Experiments (§5) and numbers (Figures 2, 5)

- **Goal-scoring domain**: 14 continuous state variables (player/keeper/ball positions, velocities, orientations); two parameterized actions — `kick-to(x,y)` ("kicks the ball towards position (x,y)", 2D parameter) and `shoot-goal(h)` ("shoots the ball towards a position h along the goal line", 1D parameter). Reward: 0 non-terminal, 50 for a goal, `-d` (distance to goal) on failure. 20 runs each. Result (Fig. 2, "goal scoring probability, averaged over 20 runs... intervals show standard error"): Q-PAMDP(1) and Q-PAMDP(∞) converge to **roughly 35%** goal probability, "learning policies that score goals roughly 35% of the time" (§7 Conclusion); fixed-parameter SARSA reaches ~20%; direct policy search (eNAC) reaches ~15%.
- **Platform domain**: state `(x, ẋ, e_x, ė_x)` (agent position/velocity, enemy position/velocity); actions `run(dx)`, `hop(dx)`, `leap(dx)`, each with a 1D horizontal-distance parameter. Reward is fraction of total platform/gap length traversed. 20 runs each. Result (Fig. 5): Q-PAMDP(∞) reaches ~65% distance covered, Q-PAMDP(1) ~50%, fixed SARSA ~40%, eNAC (direct optimization) ~10%.

Both domains are hand-designed 2D physics toys with the parameterized action set **given a priori by the domain author** — `kick-to`, `shoot-goal`, `run`, `hop`, `leap` are not derived from anything, they are specified as part of the domain definition before any learning happens.

## Related Work (§6) — the "what/how" language, attributed correctly

The paper's related-work section, discussing Hoey et al.'s Bayesian affect-control-theory work (not the PAMDP authors' own framing), contains the line most relevant to the branch's discrete/continuous split:

> "the discrete and continuous components of the action space reflect different control aspects: the discrete control provides the 'what', while the continuous control describes the 'how'." (§6, on Hoey et al.)

**This is Hoey et al.'s phrase, cited approvingly, not a claim original to Masson et al.** Important for citation hygiene if the "what/how" language gets used elsewhere in the branch (e.g. against HACMan++) — it did not originate in the PAMDP paper itself, though the PAMDP paper's own formalism is a special case of the same discrete/continuous split.

§6 also explicitly contrasts PAMDP's action space against Hauskrecht et al.'s factored hybrid-action MDPs: "their formalism has an action space with a mixed set of discrete and continuous components, whereas our domain has distinct actions with **a different number of continuous components for each action**." This per-action variable parameter-dimensionality is a load-bearing structural feature of PAMDP (see below).

## Load-bearing assumptions

1. **A single, fixed MDP.** PAMDP is defined for one environment with one fixed `A_d`. There is no notion of a *family* of PAMDPs indexed by an external variable (e.g., object identity/shape) with shared or corresponding discrete labels across instances.
2. **`A_d` and each `X_a` are given, not derived.** The paper is entirely about learning policies over an already-specified parameterized action set; it gives zero method for constructing that set from environment/object structure. Domains are hand-authored (soccer kick/shoot, platformer run/hop/leap).
3. **The parameter policy `π^a_θ(x|s)` is learned via policy search** (P-UPDATE = eNAC or similar), not computed by a closed-form/deterministic geometric rule. The convergence theorems (4.1–4.4) are proven for an *iterative optimization* `P-UPDATE`, not for an arbitrary fixed function `x = f(s)`.
4. **`X_a` is a fixed continuous domain per action**, not state-dependent, not restricted by a feasibility/availability predicate. The integral `∫_{x∈X_a}` in `P_θ`/`R_θ` (§3) assumes every `x∈X_a` is always a legal choice.
5. Both experimental domains are low-dimensional (4–14 state variables), fully simulated, single-agent, no vision/perception step of any kind.

## What transfers to our setting

- **The type signature is validated and given a name.** `A = ⋃_{a∈A_d}{(a,x)|x∈X_a}` is exactly the branch's `{(k, x_k)}` schema — a discrete label plus a per-label continuous parameter of possibly different dimensionality per label. This is category (b)-adjacent in the branch's Q1 taxonomy (an index into a discrete set, paired with continuous offset data), formalized with a name (PAMDP), a name for the general class (parameterized action), and a convergence theory for optimizing within it.
- **Per-action variable parameter dimensionality is native, not a workaround.** §6's explicit contrast with Hauskrecht et al. ("a different number of continuous components for each action") directly supports Q12's cross-shape variable-arity question at the level of a single object: a box face-address could carry a 2D in-plane offset parameter, a cylinder-orbit address could carry a 1D angle parameter, within one formalism, with no forcing of a shared parameter dimensionality across addresses. This is the right level of "yes it composes" — but see below for what it does *not* buy across objects.
- **Theorem 4.3 (`∇_θH(θ) = ∇_θJ(θ,ω*)`) gives a principled reason the discrete/continuous split doesn't lose information relative to a monolithic policy**, provided the parameter policy is optimized to (local) convergence before/while the discrete policy is learned — i.e., splitting `k` and `x_k` into separately-optimized pieces is not inherently lossy versus jointly optimizing a flat action vector, as long as each piece is actually optimized. This is a mild, generic piece of evidence for the branch's premise that a VLM-owns-`k` / geometry-owns-`x_k` split is not architecturally handicapped relative to some hypothetical joint optimizer — *conditional* on the geometry-side optimizer for `x_k` actually being good (a hand-coded deterministic rule is not automatically covered by this guarantee; see negative findings).
- **Confirms the RL/robot-learning-side vocabulary the branch's framing doc gestures at ("discrete skill vocabularies, parameterized actions") is this exact paper's namesake concept**, giving the branch a citable formal anchor for describing our own action space as "the discrete address is `A_d`, the deterministic-or-optimized pose-resolution is the parameter policy `X_{a}`" without inventing new terminology.

## What does NOT transfer, and why (negative findings)

- **No adjacency, topology, or ordering over the discrete action set anywhere in the formalism.** `A_d = {a_1,...,a_k}` is stated and used as a flat, unordered, finite set throughout (§3, confirmed by direct query — no ordering, adjacency, or relational structure between discrete actions appears anywhere in the paper). PAMDP contributes **zero** evidence either for or against needing view-space adjacency (branch Q4–Q6) beyond a structural existence proof that a flat menu-of-typed-actions works fine for RL when the actions really are just a flat menu — i.e., it is a genuine (if incidental) data point *for* the "flat menu, no topology needed" side of Q6, since the paper's own two domains never needed any relation between `kick-to`/`shoot-goal` or `run`/`hop`/`leap` to learn effective policies.
- **No initiation sets, no feasibility predicate, no state-dependent action availability.** Directly checked and confirmed absent: "the paper does not discuss infeasibility of parameters x, parameter ranges/constraints, or adaptation to state-dependent action availability." The **options framework** (Sutton/Precup/Singh) is the formalism with initiation sets, not PAMDP — these are different, related-but-distinct formalisms, and the branch question that asks whether "the options initiation set and the PAMDP formalism" jointly support region-then-refine addressing is conflating two papers' apparatus. PAMDP alone gives **no precedent** for an address persisting-with-degraded-pose / vanishing / being-flagged when infeasible (branch Q16) — that gap should be handed to Branch 4 as the branch's own question list anticipates, with the explicit note that PAMDP itself is silent here (not merely "doesn't emphasize it" — the concept doesn't appear).
- **No account of how to construct `A_d` or `X_a` from geometry, sensor data, or any environment structure.** This is the single biggest scope mismatch with our RQ2/RQ3: the paper is 100% about *learning to act well given an already-fixed parameterized action set*, never about *deriving* that set. `kick-to`/`shoot-goal`/`run`/`hop`/`leap` are domain-author choices baked in before any algorithm runs. For our build-cost question (Q17): **not applicable** — there is no "build the address space" step in this paper's pipeline to time or characterize as offline/online/amortized.
- **No cross-object / cross-MDP consistency treatment.** PAMDP is defined per single fixed MDP; nothing in the formalism addresses keeping a discrete label's *meaning* stable across two different PAMDPs (e.g., "face_3" meaning the same relative thing on a box vs. a different assembly). This bears on Q12/Q13/Q19: PAMDP gives no traction on cross-shape addressing consistency, reinforcing (does not contradict) the branch's separate finding that cross-shape *view*-addressing consistency looks like a genuine literature gap — PAMDP is simply orthogonal to that question, not evidence against the gap existing.
- **The parameter policy is learned, not deterministic — the convergence theorems do not certify a hand-coded geometric `x_k` function.** If our contract's "geometry emits `x_k` via a deterministic rule or optimizer" is read as `π^a_θ(x|s)` collapsed to a fixed (non-learned) function, none of Theorems 4.1–4.4 directly apply (they're statements about an *iterative* `P-UPDATE` converging), though nothing in the definitions *forbids* a deterministic parameter function either — it's simply an un-discussed special case, and the paper offers no evidence about whether such a substitution is safe or how it interacts with the discrete policy's learning (`M_θ`'s transition/reward marginalization `∫_{x∈X_a}π^a_θ(x|s)P(s'|s,a,x)dx` assumes a *distribution* over parameters, degenerate-at-a-point being a permissible but untested edge case).
- **Zero vision/geometry content — nothing liftable as code or geometric method.** No point-cloud, mesh, CAD, or camera-pose content anywhere; no relation to healpy/H3/MVTorch/Open3D-style tooling. For Q18 (what's liftable): only the *formal vocabulary and type signature* is liftable, not any implementation.
- **No benchmark comparing a geometry-conditioned vs. uniform action space on any task.** The two experimental domains are unrelated to view planning; this paper cannot be cited either for or against Q19's benchmark-absence claim except as one more paper (of the ~20 read across this pass) in which the comparison the branch wants simply does not exist.
- **No bearing on RQ3's cross-shape view-naming novelty claim (Q20).** The paper does not discuss viewpoints, view spaces, or object shape at all; it neither strengthens nor weakens the "nobody carries cross-shape part-consistency through to naming *views*" gap claim — it is silent on the entire domain.

## Answering the branch questions this document actually serves

- **Q1 (address taxonomy)**: PAMDP formalizes a fourth axis alongside the branch's (a)/(b)/(c) — a discrete label with a *typed, variable-dimensionality* continuous parameter attached, and names the pairing `(a,x)`. It is the closest formal validation found so far for "index + local continuous data" as a first-class RL action-space category, distinct from a pure grid index (a) and distinct from a purely relational node ID (c).
- **Q3 (does the address need pose, can it be symbolic with pose resolved downstream; test the PAMDP split)**: Directly tested by reading the source. The PAMDP split is `π^d(a|s)` (discrete) and `π^a(x|s)` (continuous), but **both live in the same agent/policy** — the paper does not model "one agent picks `k`, a *different*, non-learning geometric module picks `x_k`" as the branch's contract does. Structurally compatible (nothing forbids `π^a` being replaced by a deterministic function), but **not the same architecture**, and the paper offers no evidence either way about correctness/safety of that substitution. This is the paper's single most load-bearing answer for the branch and should be stated carefully: PAMDP licenses the *type signature*, not the *heterogeneous-agent* architecture the contract actually proposes.
- **Q15 (discrete names a region, continuous refines inside it — supported by initiation sets + PAMDP?)**: PAMDP alone supports the discrete-names/continuous-refines *shape* of the answer via `(a,x)` with `x∈X_a`, but **has no initiation-set concept** (that's the options framework, a different paper family) — the branch question's premise conflates the two formalisms and should be corrected in the synthesis.
- **Q16 (infeasibility precedent)**: none found; explicitly absent.
- **Q17 (build cost)**: not applicable; the paper never constructs an action space, only learns within a given one.
- **Q18 (liftability)**: vocabulary/formalism only, no code, no geometric method, no data-type requirement (paper is data-agnostic — pure RL theory).
- **Q6 (does the agent need adjacency at all)**: one incidental data point *for* "no" — two working RL domains never used any relation between discrete actions.

## Citations (for checkability)

- Abstract, full text: top of paper.
- Discrete-vs-continuous tradeoff motivating parameterized actions: §1 Introduction, opening paragraph.
- Parameterized-task motivating example (`kick-to(τ)` from Kober et al./da Silva et al./Deisenroth et al. task-parameterized-policy work): §2.1 Parameterized Tasks.
- Action space `A = ⋃_{a∈A_d}{(a,x)|x∈X_a}`, policy factorization `π(a,x|s)=π^d(a|s)π^a(x|s)`, `M_θ=⟨S,A_d,P_θ,R_θ,γ⟩`, `P_θ`/`R_θ` integral definitions, `H(θ)=J(θ,W(θ))`, `W(θ)=argmax_ω J(θ,ω)`: §3 Parameterized Action MDPs.
- Algorithm 1 (Q-PAMDP(k) pseudocode): §4, Algorithm 1.
- Theorems 4.1–4.4 (convergence to local optimum, bounded updates, gradient of H, Q-PAMDP(∞) convergence): §4 Theoretical Results.
- Goal-scoring domain definition (14 state vars, `kick-to(x,y)`, `shoot-goal(h)`, reward 0/50/−d): §5 Experiments.
- Goal-scoring results (~35% Q-PAMDP(1)/(∞), ~20% fixed SARSA, ~15% eNAC, 20 runs, standard-error bands): §5, Figure 2 caption; restated in §7 Conclusion ("roughly 35% of the time").
- Platform domain definition (state `(x,ẋ,e_x,ė_x)`, actions `run(dx)`/`hop(dx)`/`leap(dx)`): §5 Experiments.
- Platform results (~65% Q-PAMDP(∞), ~50% Q-PAMDP(1), ~40% fixed SARSA, ~10% eNAC, 20 runs): §5, Figure 5 caption.
- "what"/"how" quote, attributed to Hoey et al.'s Bayesian affect-control-theory work, not original to this paper: §6 Related Work.
- Contrast with Hauskrecht et al. on per-action variable parameter dimensionality: §6 Related Work.
- Related-work survey of prior hybrid/parameterized-action formalisms (Rachelson's XMDPs, Hoey et al.'s POMCP-based mixed actions, Zamani et al.'s symbolic dynamic programming, hierarchical MDPs/MAXQ): §6 Related Work, full paragraphs.
- Conclusion, verbatim summary of contributions and results: §7 Conclusion.

## Assessment for the branch

This paper is a **formal-vocabulary source, not a geometry or view-planning source** — it should be cited narrowly, for the type signature `A={(k,x_k)}` and its name (PAMDP/parameterized action), and for the honest caveat that its convergence theorems certify an *iterative-optimization* parameter policy, not a hand-coded deterministic geometric rule of the kind our contract actually wants. Its most useful contribution to the branch is negative-by-omission: it has no topology, no initiation sets (that's a different formalism the branch's own question conflates with this one), no address-construction method, and no cross-object consistency treatment — confirming that everything the branch needs beyond the bare `(k,x_k)` type signature has to come from elsewhere (aspect graphs, view graphs, C-space/primitive-frame papers, or original work). It is also incidental evidence, from two working RL domains, that a flat unordered discrete action menu needs no adjacency structure to be learnable and usable — mildly supporting the "adjacency may be unnecessary decoration" side of Q6.
