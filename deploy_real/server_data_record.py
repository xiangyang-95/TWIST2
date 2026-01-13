#!/usr/bin/env python3

"""
Data collection script.
Collect data from Redis and save to a file.
The data includes:
- vision data
- body and hand state
- body and hand action


"""
import os
import cv2
import json
import time
import redis
import argparse
import threading
import numpy as np
from rich import print
from datetime import datetime
from multiprocessing import shared_memory

from robot_control.speaker import Speaker
from data_utils.episode_writer import EpisodeWriter
from data_utils.image_client import ImageClient


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


def main(args):
    # Connect to Redis with connection pool for better performance
    try:
        print(
            f"Connecting to Redis at {args.robot_ip}:6379, DB=0 with connection pool...")
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
        redis_pipeline = redis_client.pipeline()
        # Test connection
        redis_client.ping()
        print(f"Connected to Redis at localhost:6379, DB=0 with connection pool")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")
        return

    num_cameras = 1
    image_show = True
    recording = False
    goal = "walk ahead and pick a box."
    desc = "a humanoid robot walk head and pick a box from the table."
    steps = "step1: walk ahead 1 meter. step2: pick a box from the table."

    print("Task description:")
    print(f"Goal: {goal}\nDescription: {desc}\nSteps: {steps}")

    # Initialize OpenCV window
    # Create shared memory for single camera - 640x480 image
    # Height, Width, Channels for OpenCV format
    image_shape = (480, 640*num_cameras, 3)
    image_shared_memory = shared_memory.SharedMemory(create=True, size=int(
        np.prod(image_shape) * np.uint8().itemsize * num_cameras))
    image_array = np.ndarray(image_shape, dtype=np.uint8,
                             buffer=image_shared_memory.buf)

    print(f"Creating teleimager client. Connecting to {args.robot_ip}")
    teleimager_client = TeleimagerClient(
        host=args.robot_ip,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
    )
    teleimager_thread = threading.Thread(
        target=teleimager_client.receive_process, daemon=True)
    teleimager_thread.daemon = True
    teleimager_thread.start()

    # create recorder
    save_data_keys = ['rgb']
    if not os.path.exists(args.data_folder):
        print(f"Creating data folder at {args.data_folder}...")
        os.makedirs(args.data_folder)

    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=image_shape,
        data_keys=save_data_keys
    )
    recorder.text_desc(
        goal=goal,
        desc=desc,
        steps=steps
    )

    control_dt = 1 / args.frequency
    step_count = 0
    running = True
    print("Recorded control frequency: ", args.frequency)

    speaker = Speaker()
    prev_button_pressed = False
    print("Press Y button on the left controller to start/save recording.")
    
    try:
        while running:
            start_time = time.time()
            controller_data = json.loads(redis_client.get(f"controller_data"))
            button_pressed = controller_data['LeftController']['key_two']
            print(f"==> button_pressed: {button_pressed}", end="\r")

            quit_key = controller_data['LeftController']['axis_click']
            if quit_key:
                running = False
                speaker.speak("Recording stopped.")
                print("\nQuitting...")
                break

            # Detect button press (rising edge detection)
            if button_pressed and not prev_button_pressed:
                print("button pressed")
                recording = not recording
                if recording:
                    speaker.speak("episode recording started.")
                    if not recorder.create_episode():
                        recording = False
                    step_count = 0
                    print("episode recording started...")
                else:
                    recorder.save_episode()
                    speaker.speak("episode saved.")

            # Update previous button state
            prev_button_pressed = button_pressed

            if recording:
                # Attempt to retrieve "action_mimic" from Redis
                data_dict = {'idx': step_count}
                # receive vision data
                data_dict["rgb"] = image_array.copy()  # type: ignore
                # current timestamp in ms
                data_dict["t_img"] = int(time.time() * 1000)

                # Pipeline Redis operations for better performance
                redis_keys = [
                    "state_body_unitree_g1_with_hands",
                    "state_hand_left_unitree_g1_with_hands",
                    "state_hand_right_unitree_g1_with_hands",
                    "state_neck_unitree_g1_with_hands",
                    "t_state",
                    "action_body_unitree_g1_with_hands",
                    "action_hand_left_unitree_g1_with_hands",
                    "action_hand_right_unitree_g1_with_hands",
                    "action_neck_unitree_g1_with_hands",
                    "t_action",
                ]

                data_dict_keys = [
                    "state_body",
                    "state_hand_left",
                    "state_hand_right",
                    "state_neck",
                    "t_state",
                    "action_body",
                    "action_hand_left",
                    "action_hand_right",
                    "action_neck",
                    "t_action",
                ]

                try:
                    # Use Redis pipeline to batch all GET operations (1 network round-trip instead of 10)
                    for key in redis_keys:
                        redis_pipeline.get(key)
                    redis_results = redis_pipeline.execute()

                    # Process results with error handling
                    for i, (result, dict_key) in enumerate(zip(redis_results, data_dict_keys)):
                        if result is not None:
                            try:
                                data_dict[dict_key] = json.loads(result)
                            except json.JSONDecodeError:
                                print(
                                    f"Warning: Failed to decode JSON for key {redis_keys[i]}")
                                data_dict[dict_key] = None
                        else:
                            print(
                                f"Warning: No data found for key {redis_keys[i]}")
                            data_dict[dict_key] = None

                except Exception as e:
                    print(f"Error in Redis pipeline operation: {e}")
                    # Fallback: skip this recording cycle
                    continue

                # write data to recorder
                recorder.add_item(data_dict)

                if image_show:
                    # Check if image array has valid data
                    if image_array is not None and image_array.size > 0:
                        # resize image for display (stereo image is already concatenated)
                        # image_display = cv2.resize(image_array, (image_array.shape[1]//2, image_array.shape[0]//2))
                        image_display = image_array
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(
                            window_name, image_display.shape[1], image_display.shape[0])
                        # Position window on left side
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)

                step_count += 1
                elapsed = time.time() - start_time
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)
            else:
                if image_show:
                    # Check if image array has valid data
                    if image_array is not None and image_array.size > 0:
                        # resize stereo image for display
                        # image_display = cv2.resize(image_array, (image_array.shape[1]//2, image_array.shape[0]//2))
                        image_display = image_array
                        # Create window with size matching image
                        window_name = "Press controller button to start/stop recording"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(
                            window_name, image_display.shape[1], image_display.shape[0])
                        # Position window on left side
                        cv2.moveWindow(window_name, 50, 50)
                        cv2.imshow(window_name, image_display)
                        cv2.waitKey(1)
                else:
                    # For keyboard mode, just sleep to avoid busy waiting
                    time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, exiting...")
        running = False
    finally:
        print(
            f"\nDone! Recorded {recorder.episode_id + 1} episodes to {task_dir}")

        # unlink and release shared memory
        image_shared_memory.unlink()
        image_shared_memory.close()
        recorder.close()
        cv2.destroyAllWindows()  # Close OpenCV window
        print("Exiting the recording...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Record 'mimic_obs' from Redis.")
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument(
        "--data_folder", default=f"./deploy_real/twist2_demonstration", help="data folder")
    parser.add_argument("--task_name", default=f"{cur_time}", help="task name")
    parser.add_argument("--frequency", default=30, type=int)
    parser.add_argument("--robot", default="unitree_g1",
                        choices=["unitree_g1"], help="robot name")
    parser.add_argument(
        "--robot_ip", default="192.168.123.164", help="robot ip")

    args = parser.parse_args()

    main(args)
