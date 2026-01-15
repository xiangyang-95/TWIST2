#!/usr/bin/env python3

"""
Data playback script.
Load data from a JSON file and send action data to Redis.
The data includes:
- action_body
- action_hand_left
- action_hand_right
- action_neck
"""

import os
import json
import time
import redis
import argparse
import numpy as np
from rich import print
from scipy.spatial.transform import Rotation as R

try:
    import mujoco
    from mujoco.viewer import launch_passive
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not found. Visualization disabled.")

def main(args):
    # Setup visualization if requested
    sim_model = None
    sim_data = None
    viewer = None
    
    if args.vis:
        if not MUJOCO_AVAILABLE:
            print("Error: Cannot visualize because MuJoCo is not available.")
            return

        # Locate XML file (mimicking server_motion_lib.py logic)
        HERE = os.path.dirname(os.path.abspath(__file__))
        if "unitree_g1" in args.robot:
            xml_file = os.path.join(HERE, "../assets/g1/g1_mocap_29dof.xml")
        else:
            print(f"Warning: Unknown robot type {args.robot}, trying default XML.")
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
        except Exception as e:
            print(f"Error loading MuJoCo: {e}")
            return

    # Connect to Redis
    try:
        print(f"Connecting to Redis at {args.robot_ip}:6379...")
        redis_pool = redis.ConnectionPool(
            host=args.robot_ip,
            port=6379,
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1
        )
        redis_client = redis.Redis(connection_pool=redis_pool)
        redis_client.ping()
        print("Connected to Redis successfully.")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    # Check if data file exists
    json_path = os.path.join(args.data_path, "data.json")
    if not os.path.exists(json_path):
        # Try checking if args.data_path is the episode folder or the data.json itself
        if os.path.isfile(args.data_path) and args.data_path.endswith('.json'):
            json_path = args.data_path
        else:
            print(f"Error: {json_path} does not exist.")
            return

    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'data' not in data:
            print("Error: Invalid JSON format. 'data' key missing.")
            return
            
        records = data['data']
        print(f"Loaded {len(records)} frames.")
        
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return

    frequency = args.frequency
    period = 1.0 / frequency
    
    print(f"Starting playback at {frequency}Hz...")
    
    # Map JSON keys to Redis keys
    # Note: Using unitree_g1_with_hands suffix as default for now
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

    try:
        for i, record in enumerate(records):
            start_time = time.time()
            
            pipeline = redis_client.pipeline()
            
            for json_key, redis_key in action_keys_map.items():
                if json_key in record and record[json_key] is not None:
                    # Convert to JSON string as expected by the consumer (if following server_data_record pattern)
                    value_str = json.dumps(record[json_key])
                    pipeline.set(redis_key, value_str)
            
            pipeline.execute()
            print(record)
            break
            # Visualization
            if viewer and f"action_body_{args.robot}" in record:
                try:
                    # action_body structure (mimic_obs): 
                    # 0-1: root_vel_xy (2)
                    # 2: root_pos_z (1)
                    # 3-4: roll, pitch (2)
                    # 5: ang_vel_yaw (1)
                    # 6...: dof_pos
                    val = record[f"action_body_{args.robot}"]
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
                    pass

            print(f"Playing frame {i+1}/{len(records)}", end='\r')
            
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)
                
    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    except Exception as e:
        print(f"\nError during playback: {e}")
    
    print("\nPlayback finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play back recorded actions to Redis.")
    parser.add_argument("--data_path", required=True, help="Path to the episode folder (containing data.json) or the json file itself.")
    parser.add_argument("--robot_ip", default="localhost", help="Robot IP address")
    parser.add_argument("--frequency", type=int, default=30, help="Playback frequency in Hz")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion in MuJoCo")
    parser.add_argument("--robot", default="unitree_g1_with_hands", help="Robot name (for asset loading)")    
    args = parser.parse_args()
    main(args)
