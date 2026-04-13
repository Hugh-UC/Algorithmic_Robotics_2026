# NVIDIA GPU FIX | Docker Simulation

This directory contains an alternative Docker Compose file (`compose-simulation-nvidia.yaml`) designed to enable hardware acceleration for the main lab's simulation. 

By default, the standard Docker containers may rely on CPU rendering or integrated graphics, which can cause **performance bottlenecks** or **rendering issues** in ROS2, RVIZ, and Unity environments. This configuration explicitly passes your host machine's NVIDIA GPU through to the containers, ensuring much smoother GUI performance.

***

### Release (v1.0)

<br>

## Table of Contents

- [Getting Started](#getting-started)
    - [1. Prerequisites](#1-prerequisites)
    - [2. How to Use the NVIDIA Compose File](#2-how-to-use-the-nvidia-compose-file)
- [Code Breakdown: What Was Added?](#code-breakdown-what-was-added)
    - [1. GPU Device Reservation](#1-gpu-device-reservation)
    - [2. Optimus / Prime Environment Variables](#2-optimus--prime-environment-variables)
- [Alternative Configurations](#alternative-configurations)

<br>

<br>

## Getting Started

### 1. Prerequisites
Before using this configuration, your host machine must have:
* A NVIDIA GPU.
* The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed and configured on your host system.
* Proprietary NVIDIA drivers installed. For more support on driver install/reinstall see:
    * [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/solved-install-any-proprietary-nvidia-driver-on-ubuntu-and-other-linux-distriutions-rminitadapter-via-esxi-solved/353557)
    * [ask Ubuntu](https://askubuntu.com/questions/1478024/how-do-i-install-an-arbitrary-proprietary-nvidia-gpu-driver-on-ubuntu-studio-22)

### 2. How to Use the NVIDIA Compose File

Because this file is named differently than the default `compose-simulation.yaml`, it will **not** run automatically. You must explicitly point Docker to it.

Navigate to the directory containing this file:
```sh
cd ~/algorithmic-robots-world/workspace/succulence_ws/nvidia_gpu_fix/
```
OR
```sh
cd ~/<YOUR_REPO_NAME>/workspace/succulence_ws/nvidia_gpu_fix
```

Start the Simulation (containers):
```sh
docker compose -f compose-simulation-nvidia.yaml pull
xhost +local:root
docker compose -f compose-simulation-nvidia.yaml up -d
```

Stop the Simulation (containers):
```sh
docker compose -f compose-simulation-nvidia.yaml down
```

<br>

## Code Breakdown: What Was Added?

To enable hardware acceleration, two main blocks of code were added to the `ar-simulation` and `ar-workspace` services.

### 1. GPU Device Reservation
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
This is the modern Docker Compose specification for requesting host hardware. It specifically instructs the Docker daemon to allocate one NVIDIA GPU to the container and exposes the basic `[gpu]` capabilities needed for rendering.

<br>

### 2. Optimus / Prime Environment Variables
**_(Added to the ar-simulation service only)_**
```yaml
    - __NV_PRIME_RENDER_OFFLOAD=1
    - __GLX_VENDOR_LIBRARY_NAME=nvidia
    - __VK_LAYER_NV_optimus=NVIDIA_only
```
Many modern laptops utilize "NVIDIA Optimus" (a hybrid graphics setup utilizing both an integrated Intel/AMD GPU for power saving and a discrete NVIDIA GPU for heavy lifting). These specific environment variables force the container's GUI framework (X11/OpenGL/Vulkan) to offload the rendering **strictly to the discrete NVIDIA GPU** rather than defaulting to the integrated graphics.

<br>

## Alternative Configurations
Depending on your hardware setup, you may want to tweak the `compose-simulation-nvidia.yaml` file. Here are some alternative options:

### 1. Multi-GPU Setups (`count: all`)
If you have a desktop or server with multiple NVIDIA GPUs and want the container to have access to all of them, change the count parameter.
- **Code:** Change `count: 1` to `count: all`
- **Effect:** Exposes every NVIDIA GPU on the host to the container instead of just the primary one

### 2. Requesting Specific GPU Capabilities (`capabilities: [gpu, compute, utility, graphics]`)
- **Code:** Change `capabilities: [gpu]` to `capabilities: [gpu, compute, utility, graphics]`
- **Effect:** By default, `[gpu]` is usually enough for basic rendering. However, if your ROS2 nodes utilize CUDA for heavy algorithmic processing (like point-cloud mapping or machine learning), explicitly adding compute ensures the CUDA toolkit APIs are fully accessible within the container.

### 3. Desktop Users (Removing Optimus Variables)
- **Code:** Delete the `__NV_PRIME_RENDER_OFFLOAD` and `__VK_LAYER_NV_optimus` environment variables
- **Effect:** If you are using a dedicated desktop workstation where the monitor is plugged directly into the NVIDIA GPU (no integrated graphics), these variables are redundant. Removing them won't break the simulation, but it cleans up the configuration for non-laptop users.

<br>

***