#!/bin/bash

source ~/miniforge3/bin/activate gmr

cd deploy_real
redis_ip=${REDIS_IP:-localhost}
echo "Using Redis IP: $redis_ip"

actual_human_height=1.7
python xrobot_teleop_to_robot_w_hand.py \
    --robot unitree_g1 \
    --actual_human_height $actual_human_height \
    --redis_ip $redis_ip \
    --target_fps 100 \
    --measure_fps 1
    # --smooth \
    # --pinch_mode
