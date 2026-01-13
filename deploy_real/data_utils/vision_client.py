#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import numpy as np
import time
import cv2
import io
from multiprocessing import shared_memory
from collections import deque
from rich import print
import struct
import pickle

class VisionClient:
    def __init__(
        self,
        server_address="127.0.0.1",
        port=5555,
        
        img_shape=None,
        img_shm_name=None,
        depth_shape=None,
        depth_shm_name=None,
        
        unit_test=False,
        image_show=False,
        depth_show=False,
    ):
        """
        A ZeroMQ subscriber client for receiving JPEG-compressed RGB and depth data from a single camera.

        Parameters:
        -----------
        server_address : str
            The IP address or hostname where the server is running.
        port : int
            The ZeroMQ PUB socket port used by the server.
        img_shape : tuple or None
            Shape (H, W, C) for the RGB image shared memory array (optional).
            Default resolution is 480x640 (Height x Width).
        img_shm_name : str or None
            Shared memory name for RGB image (optional).
        depth_shape : tuple or None
            Shape (H, W) for depth image shared memory array (optional).
            Default resolution is 480x640.
        depth_shm_name : str or None
            Shared memory name for depth image (optional).
        unit_test : bool
            If True, enables a simple FPS/performance print-out.
        image_show : bool
            If True, displays the RGB image in a pop-up window.
        depth_show : bool
            If True, uses OpenCV to visualize the depth image in real-time.
            
        Note:
        ----
        This client expects JPEG-compressed image data with the format:
        [4-byte width][4-byte height][4-byte JPEG length][JPEG data]
        """
        self.server_address = server_address
        self.port = port
        self.running = True
        
        self.image_show = image_show
        self.depth_show = depth_show
        
        # Optional shared memory for RGB image (single camera)
        self.img_shape = img_shape
        self.img_shm_name = img_shm_name
        self.img_shm_enabled = False
        if (self.img_shape is not None) and (self.img_shm_name is not None):
            self.img_shm = shared_memory.SharedMemory(name=self.img_shm_name)
            self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img_shm.buf)
            self.img_shm_enabled = True

        # Optional shared memory for depth image (single camera)
        self.depth_shape = depth_shape
        self.depth_shm_name = depth_shm_name
        self.depth_shm_enabled = False
        if (self.depth_shape is not None) and (self.depth_shm_name is not None):
            self.depth_shm = shared_memory.SharedMemory(name=self.depth_shm_name)
            self.depth_array = np.ndarray(self.depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
            self.depth_shm_enabled = True

        # Simple performance metrics if needed
        self.unit_test = unit_test
        if self.unit_test:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        """Initialize a simple set of performance metrics (FPS, etc.)."""
        self.frame_count = 0
        self.start_time = time.time()
        self.time_window = 1.0
        self.frame_times = deque()

    def _update_performance_metrics(self, print_info, verbose=False):
        """Update and print performance metrics (like FPS)."""
        if not self.unit_test:
            return

        now = time.time()
        self.frame_times.append(now)
        while self.frame_times and self.frame_times[0] < now - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

        # Print every 30 frames
        if self.frame_count % 30 == 0:
            real_time_fps = len(self.frame_times) / self.time_window
            elapsed = now - self.start_time
            if verbose:
                print(f"[VisionClient] FPS: {real_time_fps:.2f}, Frames: {self.frame_count}, Elapsed: {elapsed:.2f}s")
                print(print_info, "\n")

    def handle_color_image(self, color_img):
        """
        Handle the RGB image from single camera.
        - Optionally display it
        - Optionally copy to shared memory
        """
        if color_img is None:
            return

        # If shared memory is enabled for RGB images, copy it over
        if self.img_shm_enabled and self.img_shape is not None:
            try:
                if color_img.shape == self.img_shape:
                    np.copyto(self.img_array, color_img)
                else:
                    # Resize if shape mismatch
                    h, w = self.img_shape[0], self.img_shape[1]
                    resized = cv2.resize(color_img, (w, h))
                    np.copyto(self.img_array, resized)
            except Exception as e:
                print(f"[VisionClient] Error copying to SHM: {e}")

        # If you want to display the image
        if self.image_show:
            # Convert RGB to BGR for OpenCV display (similar to example_vision_client.py)
            display_img = color_img.copy()
            if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                # Assume RGB format and convert to BGR for OpenCV
                # display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                pass
            
            cv2.imshow("VisionClient - RGB", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

    def handle_depth_image(self, depth_img):
        """
        Handle the depth image from single camera.
        - Optionally copy to a shared memory region
        - Optionally visualize with OpenCV if depth_show is True.
        """
        if depth_img is None:
            return

        # If shared memory is enabled for depth images, copy it
        if self.depth_shm_enabled and self.depth_shape is not None:
            if depth_img.shape == self.depth_shape:
                np.copyto(self.depth_array, depth_img)
            # else: adapt cropping/resizing as needed

        # Real-time depth visualization using simplified approach from example_vision_client.py
        if self.depth_show:
            # Handle different depth formats
            temp = depth_img.copy()
            
            # If the image has 3 channels, extract first channel
            if temp.ndim == 3:
                if temp.shape[2] == 1:
                    temp = np.squeeze(temp, axis=2)
                else:
                    temp = temp[:, :, 0]
            
            # Use simplified depth visualization approach (similar to example_vision_client.py)
            max_depth = 5000  # Maximum depth value for normalization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(temp, alpha=255.0/max_depth), 
                cv2.COLORMAP_JET
            )
            
            cv2.imshow("VisionClient - Depth", depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
    
    def _close(self):
        """Close the ZeroMQ socket and any windows."""
        self.socket.close()
        self.context.term()
        if self.image_show or self.depth_show:
            cv2.destroyAllWindows()
        print("[VisionClient] Closed.")

    def receive_process(self):
        """
        Main loop for single camera:
        - Connect to the server (PUB socket)
        - Continuously receive JPEG-compressed data
        - Decode JPEG and handle RGB and depth images
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 1)  # Only keep latest message
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.server_address}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        print(f"[VisionClient] Subscribed to tcp://{self.server_address}:{self.port}. Waiting for data...")
        try:
            while self.running:
                events = dict(poller.poll(timeout=100))
                if self.socket in events:
                    try:
                        start_time = time.time()

                        # 接收消息
                        message = self.socket.recv()
                        # print(f"[VisionClient] Received message of size {len(message)} bytes.")
                        
                        if message is not None:
                            np_img = np.frombuffer(message, dtype=np.uint8)
                            bgr_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                            if bgr_img is not None:
                                # Convert BGR to RGB
                                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                                self.handle_color_image(rgb_img)
                            else:
                                print(f"[VisionClient] Failed to decode image. Message len: {len(message)}")
                    
                        end_time = time.time()
                        loop_fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                        
                        print_info = f"fps: {loop_fps:.2f}"
                        self._update_performance_metrics(print_info)

                    except Exception as e:
                        print(f"[VisionClient] Error receiving/decoding: {e}")
                else:
                     # Timeout, continue or sleep slightly?
                     # poller.poll already waited up to 100ms.
                     pass

        except KeyboardInterrupt:
            print("[VisionClient] Interrupted by user.")
        except Exception as e:
            print(f"[VisionClient] Error: {e}")
        finally:
            self._close()
