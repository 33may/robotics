from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "lerobot/svla_so101_pickplace"

ds = LeRobotDataset(repo_id)


print(ds.num_episodes)

print(ds.num_frames)

print(ds.fps)

print(ds.features)

print(ds.meta.episodes[0])

print(ds.root)


item = ds[0]

# image = item["observation.image.top"]

# print(f"Image shape: {image.shape}")
# print(f"Image data type: {image.dtype}")
# print(f"Image values range: {image.min()} to {image.max()}")
