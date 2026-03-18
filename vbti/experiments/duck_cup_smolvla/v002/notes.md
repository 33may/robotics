# v002 — Notes

baseline real data


Okay the model trained on real data performed way worse then the sim data.

lets check what is wrong

The model always goes to the resting position. This might be also because there are really a lot of frames in the dataset that are like that. This is datasets problem.

However even starting from the not-resting position gives the same result, lets also try to increase the horizon, so that way we should execute the whole sequence of actions, however I douubt it will help

Since even when I place the robot is the mid-action position it still returns to the initial resting pose.

indeed the larger action horizon still have the same problem