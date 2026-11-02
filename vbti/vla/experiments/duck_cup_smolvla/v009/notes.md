

Okay here we collected the new dataset for front duck red cup and fine-tuned the v008 ckpt 20k to see if the pre-trained model could execute the target action from dataset with good precision

![[Pasted image 20260401095916.png]]

Again, here we could see that model settles on platoeu around step 20-25k, so we expect these models to perform best out of the sample, we will eval on the same table, here actually we expect the good performance only on the front mode, since the data in dataset focused mostly on the front, however we will also eval the other duck positions.


Okay actually even the first model completed the front task twice quite confidentially, so we will eval for 5 times each to find what was the best performing model

| ckpt   | area  | note                                                                                                             |
| ------ | ----- | ---------------------------------------------------------------------------------------------------------------- |
| 003000 | 11001 | Really fails to generalize, overfitted on the front positioniong, can't even reach the duck on the left properly |
| 006000 | 01011 | Really fails to generalize, overfitted on the front positioniong, can't even reach the duck on the left properly |
| 009000 | 10011 |                                                                                                                  |
| 012000 |       |                                                                                                                  |
| 015000 |       |                                                                                                                  |
| 018000 |       |                                                                                                                  |
| 021000 |       |                                                                                                                  |
| 024000 |       |                                                                                                                  |
| 027000 |       |                                                                                                                  |
| 030000 |       |                                                                                                                  |
Okay I don't want to complete the task, I want to collect more data already. the model covers fine the top part of the middle area

I believ this week we could achieve the 80-100% sucess rate on The whole green area with any orientation, we will collect more data

So first of all the forgetting problem is real, so we need to extend the dataset, lets collect 150 episodes today, basically by the 4:15 I will collect teleop.

