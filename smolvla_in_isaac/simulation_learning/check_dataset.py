import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_hdf5_step_images(obs_dict, step_num=0):
    """Visualize camera images from HDF5 observation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Front camera
    axes[0].imshow(obs_dict['front'][step_num])
    axes[0].set_title(f'Front - Step {step_num}')
    axes[0].axis('off')

    # Third person (front_cam_cfg)
    axes[1].imshow(obs_dict['front_cam_cfg'][step_num])
    axes[1].set_title(f'Third Person - Step {step_num}')
    axes[1].axis('off')

    # Gripper
    axes[2].imshow(obs_dict['gripper_cam_cfg'][step_num])
    axes[2].set_title(f'Gripper - Step {step_num}')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


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


    # print('\n=== Actions for steps 10-20 ===')
    # for i in range(10, 20):
    #     action = ep["actions"][i]
    #     joint_pos = ep["obs"]["joint_pos"][i]

    #     print(f'Step {i}: action={action}, joint_pos={joint_pos}')

    #     # Uncomment to visualize images at this step
    #     fig = plot_hdf5_step_images(ep["obs"], step_num=i)
    #     plt.show()
    #     plt.close()

    #     # Uncomment to visualize images at this step
    #     fig = plot_hdf5_step_images(ep["obs"], step_num=i)
    #     plt.show()
    #     plt.close()


    #     # Uncomment to visualize images at this step
    #     fig = plot_hdf5_step_images(ep["obs"], step_num=i)
    #     plt.show()
    #     plt.close()

    #     # break  # Uncomment for debug