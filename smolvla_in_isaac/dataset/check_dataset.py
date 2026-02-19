import fire
import h5py
from pathlib import Path
import matplotlib.pyplot as plt


def plot_hdf5_step_images(obs_dict, step_num=0):
    """Visualize camera images from HDF5 observation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(obs_dict['front'][step_num])
    axes[0].set_title(f'Front - Step {step_num}')
    axes[0].axis('off')

    axes[1].imshow(obs_dict['front_cam_cfg'][step_num])
    axes[1].set_title(f'Third Person - Step {step_num}')
    axes[1].axis('off')

    axes[2].imshow(obs_dict['gripper_cam_cfg'][step_num])
    axes[2].set_title(f'Gripper - Step {step_num}')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def lerobot(repo_id: str = "eternalmay33/pick_place_test", step: int = 0):
    """Inspect a LeRobot dataset: print metadata, keys, first action & image shape."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path.home() / ".cache/huggingface/lerobot"
    ds = LeRobotDataset(repo_id, root=root)

    print("=== LeRobot Dataset ===")
    print(ds)
    print()

    data = ds[step]
    print(f"Keys: {list(data.keys())}")
    print(f"Action: {data['action']}")
    print(f"Image shape: {data['observation.images.front'].shape}")
    print(f"Image dtype: {data['observation.images.front'].dtype}")


def hdf5(path: str, episode: int = 0):
    """Inspect an HDF5 dataset: print structure, actions, and observation shapes."""
    with h5py.File(path, 'r') as f:
        episodes = list(f['data'].keys())
        print("=== HDF5 Dataset ===")
        print(f"Episodes ({len(episodes)}): {episodes}")
        print()

        ep_name = episodes[episode]
        ep = f['data'][ep_name]
        print(f"=== Episode: {ep_name} ===")
        print(f"Keys: {list(ep.keys())}")
        print()

        if 'actions' in ep:
            print(f"Action shape: {ep['actions'].shape}")
            print(f"First action: {ep['actions'][0]}")

        if 'obs' in ep:
            print(f"\nObservation keys:")
            for key in ep['obs'].keys():
                print(f"  obs/{key}: {ep['obs'][key].shape}")


if __name__ == '__main__':
    fire.Fire({
        'lerobot': lerobot,
        'hdf5': hdf5,
    })
