import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path


dataset_repo_id = "eternalmay33/pick_place_test"
dataset_root = Path.home() / ".cache/huggingface/lerobot"


full_dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)


print("==================================================================================")
print("=== Lerobot Dataset ===")

print(full_dataset)

data = full_dataset[0]

print("Keys: ")

print(data.keys())

action = data["action"]

image = data["observation.images.front"]

print("Action: ")
print(action)


print("Image: ")
print(image.shape)

print(image.dtype)


print("=== HDF5 Dataset ===========================================================================")
with h5py.File('/home/may33/projects/robotics/leisaac/datasets/lift_cube.hdf5', 'r') as f:
    print('=== Dataset Structure ===')
    print(f'Episodes: {list(f["data"].keys())}')
    print()

    
    ep_name = list(f['data'].keys())[0]
    ep = f['data'][ep_name]
    print(f'=== Episode: {ep_name} ===')
    print(f'Keys в эпизоде: {list(ep.keys())}')
    print()

    # Actions
    if 'actions' in ep:
        print(f'Action: {ep["actions"][[0]]}')
        

    # Observations
    if 'obs' in ep:
        print(f'\nObservations keys: {list(ep["obs"].keys())}')
        for key in ep['obs'].keys():
            print(f'  obs/{key} shape: {ep["obs"][key].shape}')


    for i in range(10,20):        
        action = ep["actions"][[i]]
        print(action)