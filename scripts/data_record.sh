#!/bin/bash

source ~/miniforge3/bin/activate twist2
cd deploy_real
robot_ip=${ROBOT_IP:-localhost}
data_frequency=30

echo "Starting data recording from robot at IP: ${robot_ip} with frequency: ${data_frequency} Hz"
python server_data_record.py --frequency ${data_frequency} --robot_ip ${robot_ip}
