#!/bin/bash

# --- Part 1: Locate the params.yaml file ---
PARAMS_SOURCE="../succulence_rover_ros/config/params.yaml"

# check if the params.yaml file exists
if [ ! -f "$PARAMS_SOURCE" ]; then
    echo -e "! Error: The source params file '$PARAMS_SOURCE' was not found."
    exit 1
fi


# --- Part 2: Define terms to replace ---