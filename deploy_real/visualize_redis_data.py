#!/usr/bin/env python3

import redis
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Visualize Redis data in real-time.")
    parser.add_argument("--host", default="localhost", help="Redis host")
    parser.add_argument("--port", type=int, default=6379, help="Redis port")
    parser.add_argument("--key", default="action_body", help="Redis key to visualize (or alias e.g. state_body)")
    parser.add_argument("--window", type=int, default=100, help="Window size for the plot (number of samples)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated list of indices to plot (e.g., '0,1,2'). If None, plots all dimensions (be careful with large vectors).")
    parser.add_argument("--ylim", type=str, default=None, help="Y-axis limits as 'min,max' (e.g., '-1,1')")
    args = parser.parse_args()

    # Common aliases mapping
    suffix = "_unitree_g1_with_hands"
    ALIASES = {
        "action_body": f"action_body{suffix}",
        "action_hand_left": f"action_hand_left{suffix}",
        "action_hand_right": f"action_hand_right{suffix}",
        "action_neck": f"action_neck{suffix}",
        "state_body": f"state_body{suffix}",
        "state_hand_left": f"state_hand_left{suffix}",
        "state_hand_right": f"state_hand_right{suffix}",
        "state_neck": f"state_neck{suffix}"
    }

    # Resolve key
    redis_key = ALIASES.get(args.key, args.key)

    # Connect to Redis
    try:
        r = redis.Redis(host=args.host, port=args.port, db=0, decode_responses=True)
        r.ping()
        print(f"Connected to Redis at {args.host}:{args.port}")
    except redis.ConnectionError as e:
        print(f"Error connecting to Redis: {e}")
        sys.exit(1)

    # Initial data fetch to determine dimensions
    print(f"Waiting for data on key: {redis_key}...")
    dim = 0
    try:
        while True:
            initial_data = r.get(redis_key)
            if initial_data:
                try:
                    val = json.loads(initial_data)
                    if isinstance(val, list):
                        dim = len(val)
                    elif isinstance(val, (int, float)):
                        dim = 1
                    else:
                        print(f"Warning: Unknown data format {type(val)}. Assuming 1D.")
                        dim = 1
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        sys.exit(0)

    print(f"Detected dimension: {dim}")

    # Parse plotting indices
    if args.indices:
        try:
            plot_indices = [int(x.strip()) for x in args.indices.split(',')]
            # Validate indices
            plot_indices = [i for i in plot_indices if 0 <= i < dim]
        except ValueError:
            print("Error parsing indices. Use format '0,1,2'")
            sys.exit(1)
    else:
        # If dimension is very large, warn user, but proceed
        if dim > 20:
            print(f"Warning: High dimensionality ({dim}). Plotting all might be slow/messy. Use --indices to select specific channels.")
        plot_indices = range(dim)

    if not plot_indices:
        print("No valid indices to plot.")
        sys.exit(1)

    print(f"Plotting indices: {list(plot_indices)}")

    # Setup Buffer
    data_buffer = deque(maxlen=args.window)
    # Initialize with zeros
    for _ in range(args.window):
        data_buffer.append([0.0] * len(plot_indices))

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    
    # Generate colors
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(plot_indices)))

    for i, idx in enumerate(plot_indices):
        line, = ax.plot([], [], label=f"Ch {idx}", color=colors[i], linewidth=1.5)
        lines.append(line)

    ax.set_xlim(0, args.window - 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f"Real-time Data: {redis_key}")
    ax.set_xlabel("Sample Window")
    
    if args.ylim:
        try:
            ymin, ymax = map(float, args.ylim.split(','))
            ax.set_ylim(ymin, ymax)
        except ValueError:
            print("Invalid ylim format. Using auto-scale.")

    # Shrink current axis by 20% to place legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=1)

    # Update function
    def update(frame):
        raw_data = r.get(redis_key)
        if raw_data:
            try:
                val = json.loads(raw_data)
                if isinstance(val, (int, float)):
                    val = [val]
                
                # Extract values for selected indices
                current_vals = []
                for idx in plot_indices:
                    if idx < len(val):
                        current_vals.append(float(val[idx]))
                    else:
                        current_vals.append(0.0)
                
                data_buffer.append(current_vals)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                pass # Skip bad frames
        
        # Convert buffer to numpy array for easier slicing: (window, n_lines)
        arr = np.array(data_buffer) # shape: (window, num_indices)
        
        if arr.size > 0:
            x_data = np.arange(len(data_buffer))
            
            for i, line in enumerate(lines):
                line.set_data(x_data, arr[:, i])

            # Auto-scale Y if not fixed
            if not args.ylim:
                min_y = np.min(arr)
                max_y = np.max(arr)
                if min_y == max_y:
                    margin = 1.0
                else:
                    margin = (max_y - min_y) * 0.1
                current_ylim = ax.get_ylim()
                
                # Smooth update of limits
                new_min = min(current_ylim[0], min_y - margin) if min_y < current_ylim[0] else min_y - margin
                new_max = max(current_ylim[1], max_y + margin) if max_y > current_ylim[1] else max_y + margin
                
                # Or just hard set for responsiveness
                ax.set_ylim(min_y - margin, max_y + margin)

        return lines

    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
