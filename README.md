# Algorithmic_Robotics_2026

**Extension of the handcrafted autonomous navigation stack (SLAM) for the Turtlebot 4 rover and simulation environment provided by [CollaborativeRoboticsLab](https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world) for the Algorithmic Robotics (12062), Semester 1, 2026 University of Canberra unit.**

**Group ID:** 

    Group_001

**Student ID's:**

    Hugh:     u3276400
    Patrick:  u3279178
    Caleb:    u3275593

<br>

## Group Member Contributions:

### Hugh:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover
3. Implemented a decoupled, asynchronous SLAM architecture:
    - Split SLAM into two independent nodes: `SlamEstimator` (Brain) and `GlobalMapper` (Artist).
    - Resolved system freezing during graph optimisation by offloading CPU-intensive calculations to a background multiprocessing worker.
    - Implemented "Atomic Swap" mapping to ensure the robot never loses sight of the map during rebuilds.
4. Implemented new additions to the SLAM stack:
    - Global costmapping, scan-to-map matching, and reactive emergency braking with soft recovery.
5. Upgraded discrete occupancy grid map from binary to a probabilistic log-odds grid (0-100).
    
### Patrick:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover
3. Completed README file

### Caleb:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover

***

<br>

## Table of Contents

- [⚙️ Pre Build/Run Requirements](#pre-buildrun-requirements)
    - [Install Docker](#install-docker)
    - [Clone Repo](#clone-repo)
    - [Configure Parameters](#configure-parameters)
- [🏗️ Build](#build)
    - [Build ROS packages in the workspace](#build-ros-packages-in-the-workspace-via-web-server)
- [🚀 Run](#run)
    - [Start Visualisation](#start-visualisation)
    - [Run in Simulation (Unity Mars Environment)](#run-in-simulation-unity-mars-environment)
    - [Run on Physical Turtlebot 4 Rover](#run-on-physical-turtlebot-4-rover)
    - [Advanced Testing Flags](#-advanced-testing-flags)
        - [Costmaps](#costmaps)
        - [Safety Shield Overrides](#safety-shield-overrides)
- [🖥️ NVIDIA Graphics Fix](#nvidia-graphics-fix)

***

<br>

## ⚙️ Pre Build/Run Requirements

### Install Docker

Follow the official instructions [here](https://docs.docker.com/engine/install/) 

### Clone Repo
```bash
git clone https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world.git
```

### Pull Latest Docker Containers
```bash
cd ~/algorithmic-robots-world

docker compose pull
```

### Configure Parameters
Required environmental variables need to be in a .env file. An example.env file is available. Rename that file to .env and update the values as required.

[More information on Parameters](https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world/blob/main/docs/parameters.md)

***

<br>

## 🏗️ Build
Fix workspace permissions, the Docker container’s persistent volume may have been created by root, so your user 
cannot write to it. Fix this by running:
```bash
cd ~/algorithmic-robots-world 
sudo chown -R $USER:$USER ./workspace 
```

In `algorithmic-robots-world/workspace` directory create a new workspace (e.g. succulence_ws)
```bash
cd ~/algorithmic-robots-world/workspace
mkdir succulence_ws
```

In the new workspace directory clone this github repo
```bash
cd succulence_ws
gh repo clone https://github.com/Hugh-UC/Algorithmic_Robotics_2026.git
```

<br>

### Build ROS packages in the workspace (via web-server)

1. Start Simulation environment

```bash
cd ~/algorithmic-robots-world
docker compose -f compose-simulation.yaml pull
xhost +local:root
docker compose -f compose-simulation.yaml up
```
2. With the stack up, open the web browser-based VS Code interface at http://127.0.0.1:8080.

3. Navigate to the workspace directory in the web-server terminal and build ROS packages
```bash
cd succulence_ws 
colcon build --packages-select succulence_rover_ros --symlink-install 
source install/setup.bash 
```

For the physical setup follow the above steps but pull and bring up `compose-physical.yaml` in step 1

***

<br>

## 🚀 Run

### Start Visualisation:
**Run RViz2 w/ Config:**
```bash
rviz2 -d succulence_ws/src/succulence_rover_ros/config/succulance_costmap.rviz    # in 'workspace/'
```

<br>

### Run in Simulation (Unity Mars Environment):
With sim stack up, press R to bring up HMI, select autonomous mode and add Kevin (default goal coordinates)  
In a new terminal in the web browser VS Code interface, open Rviz2 with the provided config file:

**Launch Mission:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=sim
```
_(Note: If no mode is provided, it defaults to `sim`.)_

Launching the `mission.launch` file, implements the full SLAM stack to autonomously navigate to the goal coordinates (Kevin).

<br>

### Run on Physical Turtlebot 4 Rover:
Follow steps in [Build](#Build) to bring up the physical container and build ROS packages (compose-physical.yaml instead of sim container)

**SSH to rover in the web browser VS Code interface, replace `001` with number on rover (for our labs):**
```bash
ssh ubuntu@turtlebot4-001
```
or replace with `ssh ubuntu@[ipaddress]` for your own Turtlebot 4

**Password:**
```bash
turtlebot4
```

**Launch Mission:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical
```
Launch the `mission.launch` file, implements the full SLAM stack to autonomously navigate to the goal coordinates (Kevin)

**_Note the goal coordiantes are configured for our lab environment, configure them for your own environment in the `params_physical.yaml` file found at:_**
```bash
workspace/succulence_ws/src/config
```

<br>

### 🧪 Advanced Testing Flags
You can rapidly override the navigation architecture without editing YAML files using launch flags. This is highly useful for debugging physical edge cases on the fly.

#### Costmaps
**Global Only | Disable the high-speed local reflex bubble:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical costmap_mode:=global
```

**Local Only | Disable the global SLAM map entirely:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical costmap_mode:=local
```

- `both`: Uses both local and global costmaps (default).
- `global`: Disables the high-speed local reflex bubble.
- `local`: Disables the global SLAM map entirely.
- `none`: Completely disables costmaps.

#### Safety Shield Overrides:

```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical safety_mode:=collision
```

- `both`: Enables Soft Recovery and Hard Motor Locks (default).
- `collision`: Enables only the Soft Recovery brake.
- `emergency`: Enables only the Hard Motor Lock.
- `none`: Disables the Lidar safety shield entirely (A* planner handles all dodging).


***

<br>

## 🖥️ NVIDIA Graphics Fix

### Force RVIZ 2 to Use NVIDIA Graphics
```sh
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia rviz2
```
- **`__NV_PRIME_RENDER_OFFLOAD=1`**: Tells the system to use NVIDIA's "PRIME Render Offload" technology, offloading rendering to the computer's dedicated GPU.
- **`__GLX_VENDOR_LIBRARY_NAME=nvidia`**: Ensures that GLX (_OpenGL Extension for X_) uses the NVIDIA vendor library, which is necessary for the application to properly interface with the Nvidia driver.
**Vulkan Applications:**
- **`__VK_LAYER_NV_optimus=NVIDIA_only`:** Added to force compatibility, but `__NV_PRIME_RENDER_OFFLOAD=1` is usually sufficient.

**Note:** These flags require a relatively modern NVIDIA driver (435.xx or newer) to function correctly. ([NVIDIA 2019](https://download.nvidia.com/XFree86/Linux-x86_64/435.17/README/primerenderoffload.html#:~:text=To%20configure%20a%20graphics%20application,yet%20support%20PRIME%20render%20offload.))
