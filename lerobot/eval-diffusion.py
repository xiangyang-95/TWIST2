import argparse
import time
import torch
import numpy as np
import os
import redis
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

try:
    import mujoco
    from mujoco.viewer import launch_passive
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    # print("Warning: MuJoCo not found. Visualization disabled.") # printed in main

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/diffusion/final", help="Path to the trained model directory")
    parser.add_argument("--dataset_root", type=str, default="/mnt/2eb9e109-0bb6-41db-a49a-483d3806fe10/xy-ws/unitree-g1-ws/TWIST2/lerobot/lerobot_twist_dataset", help="Root directory of the dataset")
    parser.add_argument("--dataset_id", type=str, default="lerobot/twist-dataset", help="Dataset ID")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vis", action="store_true", help="Enable visualization")
    parser.add_argument("--send_result", action="store_false", help="Send results to Redis")
    args = parser.parse_args()

    # Setup visualization if requested
    sim_model = None
    sim_data = None
    viewer = None
    
    if args.send_result:
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

    print(f"Starting evaluation with model from: {args.model_path}")
    
    device = torch.device(args.device)
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
        
    dataset_root = Path(args.dataset_root)
    # if not dataset_root.is_absolute():
    #     dataset_root = Path.cwd() / dataset_root

    # Load Model
    print(f"Loading policy from {model_path}...")
    try:
        policy = DiffusionPolicy.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    policy.to(device)
    policy.eval()
    cfg = policy.config
    
    # Load Dataset Metadata
    print(f"Loading dataset metadata from {dataset_root}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_id, root=dataset_root)

    # Setup Pre/Post Processors
    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_path, dataset_stats=dataset_metadata.stats
    )

    # Setup Data Loading
    # Use [0] to force single frame evaluation to avoid 5D tensor issues in policy
    obs_delta_indices = [0] 

    delta_timestamps = {
        "observation.state": make_delta_timestamps(obs_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    if cfg.image_features:
        delta_timestamps |= {
            k: make_delta_timestamps(obs_delta_indices, dataset_metadata.fps)
            for k in cfg.image_features
        }

    dataset = LeRobotDataset(
        args.dataset_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    all_mses = []
    all_maes = []
    
    print("Starting inference loop...")
    print(f"Total batches: {len(dataloader)}")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if "action" not in batch:
                continue
                
            gt_action = batch["action"].to(device)
            obs = preprocess(batch)            
            
            # Squeeze time dimension to match inference expectation (4D tensors)
            for k in obs:
                 if isinstance(obs[k], torch.Tensor) and obs[k].ndim == 5:
                      obs[k] = obs[k].squeeze(1) # [B, 1, C, H, W] -> [B, C, H, W]
                 if isinstance(obs[k], torch.Tensor) and obs[k].ndim == 3 and k == "observation.state":
                      obs[k] = obs[k].squeeze(1) # [B, 1, D] -> [B, D]
            
            # Inference - generating normalized actions
            # Adjust input if necessary for specific policy implementation
            output_action = policy.select_action(obs)
            
            # Unnormalize predictions to physical space
            try:
                unnormalized_generated_action = postprocess(output_action)
                if isinstance(unnormalized_generated_action, dict):
                    unnormalized_generated_action = unnormalized_generated_action["action"]
            except Exception as e:
                print(f"Warning: postprocess failed with tensor: {e}")
                unnormalized_generated_action = output_action

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
            # Action body (35), action_hand_left (7), action_hand_right (7), action_neck (2) = 50
            
            if args.send_result:
                
                action_body = output_action[0][:35].cpu().numpy().flatten().tolist()
                action_hand_left = output_action[0][35:42].cpu().numpy().flatten().tolist()
                action_hand_right = output_action[0][42:49].cpu().numpy().flatten().tolist()
                action_neck = output_action[0][49:51].cpu().numpy().flatten().tolist()
                t_action = int(time.time() * 1000)
                pipeline.set(action_keys_map["action_body"], str(action_body))
                pipeline.set(action_keys_map["action_hand_left"], str(action_hand_left))
                pipeline.set(action_keys_map["action_hand_right"], str(action_hand_right))
                pipeline.set(action_keys_map["action_neck"], str(action_neck))
                pipeline.set(action_keys_map["t_action"], str(t_action))
                pipeline.execute()
            
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
                    # unnormalized_generated_action can be (B, T, D) or (B, D)
                    act_tensor = unnormalized_generated_action
                    if act_tensor.ndim == 3: # Batch, Time, Dim
                        # Take the first batch and first step
                        val_tensor = act_tensor[0, 0]
                    elif act_tensor.ndim == 2: # Batch, Dim
                        val_tensor = act_tensor[0]
                    else:
                         val_tensor = act_tensor[0]

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
                    # print(f"Visualization error: {e}")
                    pass

            unnormalized_generated_action = unnormalized_generated_action.to(gt_action.device)
            
            # Match dimensions
            if unnormalized_generated_action.shape != gt_action.shape:
                # Assuming unnormalized is [B, D] and gt is [B, Horizon, D]
                # We probably want to compare against the first action in sequence?
                # Or maybe gt is [B, D] and unnormalized is [B, D]
                if gt_action.ndim == 3 and unnormalized_generated_action.ndim == 2:
                     gt_action_step = gt_action[:, 0, :]
                elif gt_action.ndim == 3 and unnormalized_generated_action.ndim == 3:
                     # Compare relevant steps
                     min_steps = min(gt_action.shape[1], unnormalized_generated_action.shape[1])
                     gt_action_step = gt_action[:, :min_steps, :]
                     unnormalized_generated_action = unnormalized_generated_action[:, :min_steps, :]
                else:
                     gt_action_step = gt_action
                
                mse = torch.nn.functional.mse_loss(unnormalized_generated_action, gt_action_step)
                mae = torch.nn.functional.l1_loss(unnormalized_generated_action, gt_action_step)
            else:
                mse = torch.nn.functional.mse_loss(unnormalized_generated_action, gt_action)
                mae = torch.nn.functional.l1_loss(unnormalized_generated_action, gt_action)
            
            all_mses.append(mse.item())
            all_maes.append(mae.item())

    print("-" * 30)
    print(f"Evaluation Results over {len(dataset)} samples:")
    print(f"Average MSE: {np.mean(all_mses):.4f}")
    print(f"Average MAE: {np.mean(all_maes):.4f}")
    print("-" * 30)
    
if __name__ == "__main__":
    evaluate()
