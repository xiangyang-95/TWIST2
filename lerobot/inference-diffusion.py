import cv2
import json
import time
import redis
import threading
import argparse
import os
import numpy as np
from pathlib import Path
from multiprocessing import shared_memory
from scipy.spatial.transform import Rotation as R

try:
    import mujoco
    from mujoco.viewer import launch_passive
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not found. Visualization disabled.")

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_pre_post_processors

from utils.image_client import ImageClient
from utils.modeling_diffusion import DiffusionPolicy

SEND_REDIS = True


class TeleimagerClient:
    """
    A simple Teleimager client to receive images from the server.
    """

    def __init__(self, host: str, img_shape: tuple, img_shm_name: str):
        self.client = ImageClient(host=host)
        self.cam_config = self.client.get_cam_config()
        self.img_shape = img_shape
        self.img_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray(
            img_shape, dtype=np.uint8, buffer=self.img_shm.buf)

    def receive_process(self):
        while True:
            head_img, head_img_fps = self.client.get_head_frame()
            if head_img is not None:
                np.copyto(self.img_array, head_img)
            time.sleep(0.001)  # slight delay to prevent busy waiting


def get_robot_state(redis_client: redis.Redis, suffix: str) -> dict:
    keys = [
        f"state_body{suffix}",
        f"state_hand_left{suffix}",
        f"state_hand_right{suffix}",
        f"state_neck{suffix}",
        f"t_state{suffix}"
    ]
    pipeline = redis_client.pipeline()
    for key in keys:
        pipeline.get(key)
    values = pipeline.execute()
    state = {
        "state_body": json.loads(values[0]),
        "state_hand_left": json.loads(values[1]),
        "state_hand_right": json.loads(values[2]),
        "state_neck": json.loads(values[3]),
        "t_state": int(time.time() * 1000)
    }
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", help="Robot name")
    parser.add_argument("--vis", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    # Setup visualization if requested
    sim_model = None
    sim_data = None
    viewer = None
    
    if args.vis:
        if not MUJOCO_AVAILABLE:
            print("Error: Cannot visualize because MuJoCo is not available.")
            return

        # Locate XML file
        HERE = os.path.dirname(os.path.abspath(__file__))
        # Assuming we are in TWIST2/lerobot/
        xml_file = os.path.join(HERE, "../assets/g1/g1_mocap_29dof.xml")
            
        if not os.path.exists(xml_file):
            print(f"Error: XML file not found at {xml_file}")
            # Try to find it via absolute path if relative fails, largely depends on where script is run from
            # But here we use __file__ so it should be fine.
            return
            
        print(f"Loading MuJoCo model from {xml_file}...")
        try:
            sim_model = mujoco.MjModel.from_xml_path(xml_file)
            sim_data = mujoco.MjData(sim_model)
            viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
            viewer.cam.distance = 2.0
            print("MuJoCo viewer initialized.")
        except Exception as e:
            print(f"Error loading MuJoCo: {e}")
            return

    print("Starting diffusion inference script ...")
    output_directory = Path("outputs/diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    if SEND_REDIS:
        print(f"Connecting to Redis at localhost:6379...")
        redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()

        print("Connecting to image client")
        image_shape = (480, 640, 3)
        image_shared_memory = shared_memory.SharedMemory(
            create=True,
            size=int(np.prod(image_shape) * np.uint8().itemsize)
        )
        image_array = np.ndarray(
            image_shape,
            dtype=np.uint8,
            buffer=image_shared_memory.buf
        )
        teleimager_client = TeleimagerClient(
            host="localhost",
            img_shape=image_shape,
            img_shm_name=image_shared_memory.name,
        )
        teleimager_thread = threading.Thread(
            target=teleimager_client.receive_process, daemon=True)
        teleimager_thread.daemon = True
        teleimager_thread.start()

    device = torch.device("cuda")
    dataset_id = "lerobot/twist-dataset"
    dataset_root = "/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/lerobot/lerobot_twist_dataset"

    dataset_metadata = LeRobotDatasetMetadata(
        dataset_id,
        root=dataset_root
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key,
                       ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key,
                      ft in features.items() if key not in output_features}
    
    model_id = "/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/lerobot/outputs/diffusion/final"
    model = DiffusionPolicy.from_pretrained(model_id)
    model.n_action_steps = 8  # set number of action steps to predict
    print(model)
    print(f"Loading policy to device: {device} ...")
    model.to(device)
    model.eval()

    preprocess, postprocess = make_pre_post_processors(
        model.config, model_id, dataset_stats=dataset_metadata.stats
    )

    while True:
        try:
            if SEND_REDIS:
                if image_array is None:
                    print("Waiting for image data...")
                    time.sleep(0.1)
                    continue

                cv2.imshow("Head Image", image_array)
                cv2.waitKey(1)

                # Convert BGR to RGB and normalize to [0, 1] float
                img_copy = image_array.copy()
                img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(
                    img_rgb).permute(2, 0, 1).float() / 255.0
                
                robot_state = get_robot_state(redis_client, suffix="_unitree_g1_with_hands")
                # State body (34), state_hand_left (7), state_hand_right (7), state_neck (2) = 50
                print("Robot state:", robot_state)

                state = robot_state["state_body"] + robot_state["state_hand_left"] + \
                    robot_state["state_hand_right"] + robot_state["state_neck"]
                state = np.array(state, dtype=np.float32)

                obs_frame = {
                    "observation.images.head_image": img_tensor.unsqueeze(0).to(device),
                    "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
                }
            else:
                obs_frame = {
                    "observation.images.head_image": torch.randn(1, 3, 480, 640).to(device),
                    "observation.state": torch.randn(1, 50).to(device),
                }

            obs = preprocess(obs_frame)
            action = model.select_action(obs)
            print(action.shape)
            print(f"Predicted action: {action[0]}, shape: {action[0].shape}")

            # Visualization
            if viewer:
                try:
                    # action_body structure (mimic_obs): 
                    # 0-1: root_vel_xy (2)
                    # 2: root_pos_z (1)
                    # 3-4: roll, pitch (2)
                    # 5: ang_vel_yaw (1)
                    # 6...: dof_pos
                    
                    # Extract action_body (first 35 elements)
                    # action[0] can be (T, D) or (D,) depending on config
                    act_tensor = action[0]
                    if act_tensor.ndim == 2:
                        # Take the first step if sequence
                        val_tensor = act_tensor[0]
                    else:
                        val_tensor = act_tensor
                        
                    val = val_tensor[:35].detach().cpu().numpy()
                    
                    if val is not None and len(val) > 6:
                        root_z = val[2]
                        roll = val[3]
                        pitch = val[4]
                        dof_pos = val[6:]
                        
                        # Set root position (fixed x,y=0)
                        sim_data.qpos[0] = 0
                        sim_data.qpos[1] = 0
                        sim_data.qpos[2] = root_z
                        
                        # Set root orientation
                        # scipy Rotation uses scalar-last (x,y,z,w), MuJoCo uses scalar-first (w,x,y,z)
                        r = R.from_euler('xyz', [roll, pitch, 0], degrees=False)
                        quat = r.as_quat()
                        sim_data.qpos[3] = quat[3] # w
                        sim_data.qpos[4] = quat[0] # x
                        sim_data.qpos[5] = quat[1] # y
                        sim_data.qpos[6] = quat[2] # z
                        
                        # Set joint positions
                        dof_offset = 7
                        # Ensure we don't overflow
                        n_dof = min(len(dof_pos), len(sim_data.qpos) - dof_offset)
                        sim_data.qpos[dof_offset:dof_offset+n_dof] = dof_pos[:n_dof]
                        
                        mujoco.mj_forward(sim_model, sim_data)
                        viewer.sync()
                except Exception as e:
                    # Don't crash playback if vis fails
                    print(f"Visualization error: {e}")
                    pass

            if SEND_REDIS:
                # Action body (35), action_hand_left (7), action_hand_right (7), action_neck (2) = 50
                action_body = action[0][:35].cpu().numpy().flatten().tolist()
                action_hand_left = action[0][35:42].cpu(
                ).numpy().flatten().tolist()
                action_hand_right = action[0][42:49].cpu(
                ).numpy().flatten().tolist()
                action_neck = action[0][49:51].cpu().numpy().flatten().tolist()
                t_action = time.time()
                state_body = robot_state["state_body"]
                state_hand_left = robot_state["state_hand_left"]
                state_hand_right = robot_state["state_hand_right"]
                state_neck = robot_state["state_neck"]
                t_state = robot_state["t_state"]

                suffix = "_unitree_g1_with_hands"
                action_keys_map = {
                    "action_body": f"action_body{suffix}",
                    "action_hand_left": f"action_hand_left{suffix}",
                    "action_hand_right": f"action_hand_right{suffix}",
                    "action_neck": f"action_neck{suffix}",
                    "state_body": f"state_body{suffix}",
                    "state_hand_left": f"state_hand_left{suffix}",
                    "state_hand_right": f"state_hand_right{suffix}",
                    "state_neck": f"state_neck{suffix}",
                    "t_state": f"t_state{suffix}",
                    "t_action": f"t_action{suffix}"
                }
                pipeline = redis_client.pipeline()
                pipeline.set(action_keys_map["action_body"], str(action_body))
                pipeline.set(action_keys_map["action_hand_left"], str(action_hand_left))
                pipeline.set(action_keys_map["action_hand_right"], str(action_hand_right))
                pipeline.set(action_keys_map["action_neck"], str(action_neck))
                pipeline.set(action_keys_map["t_action"], str(t_action))
                # pipeline.set(action_keys_map["state_body"], str(state_body))
                # pipeline.set(action_keys_map["state_hand_left"], str(state_hand_left))
                # pipeline.set(action_keys_map["state_hand_right"], str(state_hand_right))
                # pipeline.set(action_keys_map["state_neck"], str(state_neck))
                # pipeline.set(action_keys_map["t_state"], str(t_state))
                pipeline.execute()
                print("Sent action to Redis.")

        except KeyboardInterrupt:
            print("Exiting inference loop.")
            break


if __name__ == "__main__":
    main()
