#!/bin/bash

source ~/miniforge3/bin/activate twist2
cd deploy_real

robot_ip="192.168.123.164"
data_frequency=30

python server_data_record.py --frequency ${data_frequency} --robot_ip ${robot_ip}
