#!/usr/bin/env python3
# This is a script to convert TWIST dataset to LeRobot dataset format.

import os
import cv2
import json
import shutil
import argparse
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

IMAGE_SHAPE = (480, 640, 3)


def create_empty_lerobot_dataset(repo_id, root, use_videos: bool = False):
    if os.path.exists(root + "/meta/info.json"):
        print(f"LeRobot dataset already exists at {root}. Skipping creation.")
        return LeRobotDataset(repo_id, root)

    vision_dtype = "video" if use_videos else "image"
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="arx",
        fps=50,
        features={
            "observation.images.head_image": {
                "dtype": vision_dtype,
                "shape": IMAGE_SHAPE,
                "names": [
                    "height", 
                    "width", 
                    "channel"
                ],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (50,),
                "names": [i for i in range(50)],
            },
            "action": {
                "dtype": "float32",
                "shape": (51,),
                "names": {
                    "motors": [i for i in range(51)],
                }
            },
        },
        image_writer_threads=5,
        image_writer_processes=10,
        use_videos=use_videos,
        video_backend="pyav",
    )


def get_episode_dirs(dataset_path):
    episode_dirs = [f"{dataset_path}/{d}" for d in os.listdir(
        dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"Episode directories: {episode_dirs}")
    return episode_dirs


def read_data_in_episode(episode_dir):
    with open(os.path.join(episode_dir, "data.json"), "r") as f:
        data = json.load(f)
    print(f"Read {len(data)} data points from {episode_dir}")
    return data


def main(args):
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset path {args.dataset_path} does not exist.")

    # check how many episodes
    episode_dirs = get_episode_dirs(args.dataset_path)
    num_episodes = len(episode_dirs)
    if num_episodes == 0:
        raise ValueError(
            f"No episodes found in dataset path {args.dataset_path}.")
    print(f"Found {num_episodes} episodes in the dataset.")

    shutil.rmtree(args.output_path, ignore_errors=True)
    dataset = create_empty_lerobot_dataset(
        repo_id="lerobot/twist-dataset",
        root=args.output_path,
        use_videos=False,
    )

    for ep, episode_dir in enumerate(episode_dirs):
        twist_data = read_data_in_episode(episode_dir)
        task = twist_data['text']['goal']
        for i, data in enumerate(twist_data['data']):
            print(
                f"[EP {ep}] Converting data point {i} / {len(twist_data['data'])}")
            # print(f"Sample data point: {data}")

            # Read jpg image as numpy array using cv2
            image_path = os.path.join(episode_dir, data['rgb'])
            image = cv2.imread(image_path)
            assert image.shape == IMAGE_SHAPE, f"Image shape mismatch: expected {IMAGE_SHAPE}, got {image.shape}"

            # State body (34), state_hand_left (7), state_hand_right (7), state_neck (2) = 50
            state = data['state_body'] + data['state_hand_left'] + \
                data['state_hand_right'] + data['state_neck']
            state = np.array(state, dtype=np.float32)

            # Action body (35), action_hand_left (7), action_hand_right (7), action_neck (2) = 50
            action = data['action_body'] + data['action_hand_left'] + \
                data['action_hand_right'] + data['action_neck']
            action = np.array(action, dtype=np.float32)

            frame = {
                "observation.images.head_image": image,
                "observation.state": state,
                "action": action,
                "task": task,
            }
            print(f"Frame: {frame}")

            dataset.add_frame(
                frame
            )
        dataset.save_episode()

    dataset.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TWIST dataset to LeRobot dataset format.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the TWIST dataset directory."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the converted LeRobot dataset.",
        default="./lerobot_twist_dataset"
    )
    args = parser.parse_args()

    # temp test
    main(args)
