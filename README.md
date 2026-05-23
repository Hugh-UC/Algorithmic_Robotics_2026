# Algorithmic_Robotics_2026 Group_001

Extension of the handcrafted autonomous navigation stack (SLAM) for the Tutrtlebot 4 rover and simulation environment provided by [CollaborativeRoboticsLab](https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world) for the Algorithmic Robotics (12062), Semester 1, 2026 University of Canberra unit.

## Group ID:

    Group_001

## Student ID's:

    Hugh:     u3276400
    Patrick:  u3279178
    Caleb:    u3275593

## Group Member Contributions:

### Hugh:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover
3. Implemented new additions to the SLAM stack:
    - Global costmapping
    - Scan to map matching
    - Two stage collision avoidance, first stage enforces rotational movement only when obstacles are within 0.25m within a 40deg forward cone, second stage enforces an all motor lock when obstacle is within 0.12m of forward cone (e.g. kevin)
4. Implemented additional features to the existing SLAM stack:
    - Added dynamic weighting to the look ahead parameter for pure pursuit, now weighted by the rover's current translational/rotational velocity
    - Upgraded discrete occupancy grid map from binary (0 or 1) to a probabilistic log-odds grid (0-100)
  
### Patrick:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover
3. Completed README file

### Caleb:
1. Collaborated throughout lab sessions to fill in provided packages and complete TODO's
2. Experimented with parameter tuning in simulation environment for implementation and testing on the physical rover

## Table of Contents

- [Pre Build/Run Requirements](#Pre-build/run-requirements)
- [Build](#Build)
- [Run](#Run)

## Pre Build/Run Requirements

### Install Docker

Follow the official instructions [here](https://docs.docker.com/engine/install/) 

### Clone the Repo

In the terminal run the following:
```bash
git clone https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world.git
```
Enter the folder
```bash
cd industrial-robots-and-systems-world
```

Pull the latest docker containers
```bash
docker compose pull
```


### Configure the Parameters

Required environmental variables need to be in a `.env` file. An `example.env` file is available. Rename that file to `.env` and update the values as required.



## Build

Fix workspace permissions, the Docker container’s persistent volume may have been created by root, so your user 
cannot write to it. Fix this by running:
```bash
cd ~/algorithmic-robots-world 
sudo chown -R $USER:$USER ./workspace 
```

In 'algorithmic-robots-world/workspace' directory create a new workspace (e.g. succulence_ws)
```bash
cd ~/algorithmic-robots-world/workspace
mkdir succulence_ws
```

In the new workspace directory clone this github repo
```bash
cd succulence_ws
gh repo clone https://github.com/Hugh-UC/Algorithmic_Robotics_2026.git
```

### Build ROS packages in the workspace via web-server

1. Start Simulation environment

```bash
cd algorithmic-robots-world
docker compose -f compose-simulation.yaml pull
xhost +local:root
docker compose -f compose-simulation.yaml up
```
2. With the stack up, open the web browser-based VS Code interface at http://127.0.0.1:8080.

3. Navigate to the workspace source directory in the web-server terminal and build ROS packages
```bash
cd succulence_ws 
colcon build --packages-select succulence_rover_ros --symlink-install 
source install/setup.bash 
```



<br>


## Lab 05 | Notes

### Quick Launch RVIZ 2 with Config File
```sh
rviz2 -d src/succulence_rover_ros/config/succulance.rviz
```

### Force RVIZ 2 to Use NVIDIA Graphics
```sh
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia rviz2
```
- **`__NV_PRIME_RENDER_OFFLOAD=1`**: Tells the system to use NVIDIA's "PRIME Render Offload" technology, offloading rendering to the computer's dedicated GPU.
- **`__GLX_VENDOR_LIBRARY_NAME=nvidia`**: Ensures that GLX (_OpenGL Extension for X_) uses the NVIDIA vendor library, which is necessary for the application to properly interface with the Nvidia driver.
**Vulkan Applications:**
- **`__VK_LAYER_NV_optimus=NVIDIA_only`:** Added to force compatibility, but `__NV_PRIME_RENDER_OFFLOAD=1` is usually sufficient.

**Note:** These flags require a relatively modern NVIDIA driver (435.xx or newer) to function correctly. ([NVIDIA 2019](https://download.nvidia.com/XFree86/Linux-x86_64/435.17/README/primerenderoffload.html#:~:text=To%20configure%20a%20graphics%20application,yet%20support%20PRIME%20render%20offload.))
