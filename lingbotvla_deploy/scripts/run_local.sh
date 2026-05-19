#!/usr/bin/env bash
set -e

# Run this on local robot machine with ROS2 environment.
# source /opt/ros/<ros_distro>/setup.bash
# conda activate <your_ros_env>

cd "$(dirname "$0")/.."

python local_node/ros2_node.py
