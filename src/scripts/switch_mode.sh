#!/bin/bash

YAML_PATH="$HOME/algorithmic-robots-world/workspace/succulence_ws/src/succulence_rover_ros/config/params.yaml"

echo -e "========================================="
echo -e "   Succulence Rover Environment Switch   "
echo -e "========================================="
echo -e "| Select environment:"
echo -e "| - Simulation ('s', 'sim')"
echo -e "| - Physical Robot ('p', 'physical', 'robot')"
read -p "| Enter choice: " user_input

# Convert input to lowercase
user_input=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')

if [[ "$user_input" == "s" || "$user_input" == "sim" ]]; then
    MODE="Simulation"
elif [[ "$user_input" == "p" || "$user_input" == "physical" || "$user_input" == "robot" ]]; then
    MODE="Physical"
else
    echo -e "! Invalid input. Exiting without making changes."
    exit 1
fi

if [ ! -f "$YAML_PATH" ]; then
    echo -e "! Error: Could not find the file at:"
    echo -e "! $YAML_PATH"
    exit 1
fi

echo -e "| Processing $YAML_PATH..."

if [ "$MODE" == "Physical" ]; then
    # 1. Uncomment Physical
    sed -i 's|# scan_topic: "/scan"|scan_topic: "/scan"|g' "$YAML_PATH"
    sed -i 's|# odom_topic: "/odom"|odom_topic: "/odom"|g' "$YAML_PATH"
    sed -i 's|# odom_frame: "odom"|odom_frame: "odom"|g' "$YAML_PATH"
    sed -i 's|# base_link_frame: "base_link"|base_link_frame: "base_link"|g' "$YAML_PATH"

    # 2. Comment Simulation
    sed -i 's|scan_topic: "/succulence/scan"|# scan_topic: "/succulence/scan"|g' "$YAML_PATH"
    sed -i 's|odom_topic: "/succulence/odom"|# odom_topic: "/succulence/odom"|g' "$YAML_PATH"
    sed -i 's|odom_frame: "succulence/odom"|# odom_frame: "succulence/odom"|g' "$YAML_PATH"
    sed -i 's|base_link_frame: "succulence/base_link"|# base_link_frame: "succulence/base_link"|g' "$YAML_PATH"

elif [ "$MODE" == "Simulation" ]; then
    # 1. Comment Physical
    sed -i 's|scan_topic: "/scan"|# scan_topic: "/scan"|g' "$YAML_PATH"
    sed -i 's|odom_topic: "/odom"|# odom_topic: "/odom"|g' "$YAML_PATH"
    sed -i 's|odom_frame: "odom"|# odom_frame: "odom"|g' "$YAML_PATH"
    sed -i 's|base_link_frame: "base_link"|# base_link_frame: "base_link"|g' "$YAML_PATH"

    # 2. Uncomment Simulation
    sed -i 's|# scan_topic: "/succulence/scan"|scan_topic: "/succulence/scan"|g' "$YAML_PATH"
    sed -i 's|# odom_topic: "/succulence/odom"|odom_topic: "/succulence/odom"|g' "$YAML_PATH"
    sed -i 's|# odom_frame: "succulence/odom"|odom_frame: "succulence/odom"|g' "$YAML_PATH"
    sed -i 's|# base_link_frame: "succulence/base_link"|base_link_frame: "succulence/base_link"|g' "$YAML_PATH"
fi

echo -e "| Success! Configuration updated."
echo -e "| The rover is now configured for the $MODE environment."
echo -e "========================================="