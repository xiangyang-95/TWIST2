#!/bin/bash

echo "Starting the data collector pipeline in TWIST2"
echo "Please start the XRoboToolkit first before you run the script."

# echo "Starting sim2sim environment ..."
# gnome-terminal -- bash -c "./scripts/sim2sim.sh; exec bash"

echo "Starting teleop node ..."
gnome-terminal -- bash -c "./scripts/teleop.sh; exec bash"

echo "Starting camera streaming server ..."
gnome-terminal -- bash -c "source ~/miniforge3/bin/activate teleimager && teleimager-server --rs; exec bash"

echo "Starting data recording services ..."
gnome-terminal -- bash -c "./scripts/data_record.sh; exec bash"
