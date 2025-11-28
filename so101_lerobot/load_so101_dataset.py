import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pprint

repo_id = "lerobot/svla_so101_pickplace"

dataset = LeRobotDataset(repo_id)

sample = dataset[10]

# print(sample)

delta_timestamps = {
    "observation.images.up" : [-0.2, -0.1, 0.0] #what is this?
}

dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)

sample = dataset[100]

# pprint(sample["observation.images.up"].shape)


batch_size = 16

data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)