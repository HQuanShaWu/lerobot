import logging
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 在这里列出你想下载的所有数据集 ID
DATASET_IDS = [
    "trantor2nd/rheovla_dataset",
    "trantor2nd/dynvla_dataset_lerobot"
    # "lerobot/ucsd_pick_and_place_dataset",
    # "lerobot/stanford_kuka_multimodal_dataset",
    # "lerobot/jaco_play",
    # "lerobot/taco_play",
    # "lerobot/toto",
    # "lerobot/stanford_robocook",
    # "lerobot/utaustin_mutex",
    # "lerobot/stanford_hydra_dataset",
    # "lerobot/berkeley_autolab_ur5",
]

def download_only():
    # 设置日志，这样你能看到下载进度条
    logging.basicConfig(level=logging.INFO)
    
    for repo_id in DATASET_IDS:
        print(f"\n[INFO] 开始处理: {repo_id} ...")
        try:
            # 核心步骤：初始化即触发下载
            # 这会自动拉取 parquet 文件、元数据和视频/图片文件到缓存
            dataset = LeRobotDataset(repo_id=repo_id)
            
            # 为了确保视频数据也被下载（防止懒加载），我们可以简单尝试读取第一帧
            # 如果这里不报错，说明数据已经完整在本地了
            _ = dataset[0] 
            
            print(f"   - 成功加载! 数据集包含 {len(dataset)} 帧数据。")
            print(f"[SUCCESS] {repo_id} 下载并缓存完成！")
            
        except Exception as e:
            print(f"[ERROR] 下载 {repo_id} 时依然出错: {e}")
            # 如果 Python 方式依然有问题，建议打印出来，后续可以用 CLI 方式补救

if __name__ == "__main__":
    download_only()