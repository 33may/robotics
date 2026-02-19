from scipy.spatial.transforms import Rotation

def xyz_to_quat_isaac(x, y, z):
    r = Rotation.from_euler('xyz', [x, y, z], degree=True)

    return (r[3], r[0], r[1], r[2])


def plot_observation_images(observation: dict, step_num: int = 0):
    """
    Create a matplotlib figure showing all camera images from an observation.

    Args:
        observation: Dictionary with LeRobot-formatted observations
        step_num: Step number for title

    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Front camera
    front_img = observation["observation.images.front"].cpu().numpy()[0].transpose(1, 2, 0)
    axes[0].imshow(front_img)
    axes[0].set_title(f'Front Camera - Step {step_num}')
    axes[0].axis('off')

    # Third person camera
    third_img = observation["observation.images.third_person"].cpu().numpy()[0].transpose(1, 2, 0)
    axes[1].imshow(third_img)
    axes[1].set_title(f'Third Person - Step {step_num}')
    axes[1].axis('off')

    # Gripper camera
    gripper_img = observation["observation.images.gripper"].cpu().numpy()[0].transpose(1, 2, 0)
    axes[2].imshow(gripper_img)
    axes[2].set_title(f'Gripper - Step {step_num}')
    axes[2].axis('off')

    plt.tight_layout()
    return fig