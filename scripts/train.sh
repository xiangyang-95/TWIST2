#!/bin/bash
# Usage: bash train.sh <experiment_id> <device>
# bash train.sh 1103_twist2 cuda:0

source ~/miniforge3/bin/activate twist2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/miniforge3/envs/twist2/lib

# Check for experiment ID argument
if [ -z "$1" ]; then
    echo "Error: No experiment ID provided."
    echo "Usage: bash train.sh <experiment_id> <device>"
    exit 1
fi

exptid=$1
device=${2:-cuda:0}

robot_name="g1"
task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"

echo "Starting training with the following parameters:"
echo "Experiment ID: $exptid"
echo "Device: $device"

cd legged_gym/legged_gym/scripts
python train.py \
    --task "${task_name}" \
    --proj_name "${proj_name}" \
    --exptid "${exptid}" \
    --device "${device}" \
    --teacher_exptid "None" \
    # --resume \
    # --debug \
