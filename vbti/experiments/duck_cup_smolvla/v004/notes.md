# v004 — Notes

test 08-merged_trimmed to verify data distribution hypothesis. Same hyperparams as v003, only dataset changes. If this shows meaningful motion → problem is in so101_real data, not training.


Okay I have stopped the training so now we need to evaluate how the model performes, we hope to see meaningfull actions that are not stuck in resting position

![[eval_v004_step_005000_realsense.mp4]]

Okay the model still has a hard time escaping the resting posion, however when it esxapes it, it starts to move somewhat reasonable. lets verify also how it will move with the opencv cameras, since that is the dataset was recorded

![[eval_v004_step_005000.mp4]]

Okay switching to the opencv didn't saved the performance, we still see the same problem of escaping the starting position and then doing somewhar random actions. Now when I think it miight be also the trimming so the model rarely see the starting state and thats why can't learn eough demonstartaions to exape it

![[eval_v004_step_005000_ah10_opencv.mp4]]

Okay the model looks clearly undertrained, however it shos reaonable behaviour once esccapes the resting state.

Lets also eval the previous v003 model and check if it will perform fine in the non resting position

No the v003 still returns to the resting position, even starting from non-resting state

Where I think we are:
  1. The so101_real data has a structural rest bias — confirmed, v003 proves it
  2. The 08-merged_trimmed data is healthier — confirmed, v004 shows different (better) failure mode
  3. v004 is undertrained at 3.8 epochs — the reasonable-once-moving behavior supports this
  4. The trimming may be cutting too much of the rest→motion transition, making it hard to learn "how to start moving"


v005 dataset

- 08-merged_trimmed — healthy distribution, proven to produce motion
- so101_real aggressively trimmed — more data variety, but with the rest spikes cut harder (thresh 40-50 instead of 15)
- sim data — no rest bias at all, adds diversity + the model already learned from it in v001

The mix covers each other's weaknesses. Sim fills gaps in real data coverage, 08-merged provides the backbone, and aggressively trimmed so101 adds more real-world variety without the rest poison.

The problem here is that it might actually corrupt the training if even with more agressive trimming the rest bias still in the model

lets examine the so101_real_50_eps better

