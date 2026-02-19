from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

def load_dataset_meta(repo_id):
    dataset_meta = LeRobotDatasetMetadata(repo_id=repo_id)

    return dataset_meta


print(load_dataset_meta("younghwan-chae/so101-v2"))