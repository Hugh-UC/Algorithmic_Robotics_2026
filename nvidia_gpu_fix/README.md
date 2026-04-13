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
- [GPU Monitoring](#gpu-monitoring)
    - [1. Find your Container Name](#1-find-your-container-name)
    - [2. Execute GPU Monitoring](#2-execute-gpu-monitoring)
        - [Expected Output](#expected-output)
- [Code Breakdown: What Was Added?](#code-breakdown-what-was-added)
    - [1. GPU Device Reservation](#1-gpu-device-reservation)
    - [2. Optimus / Prime Environment Variables](#2-optimus--prime-environment-variables)
- [Alternative Configurations](#alternative-configurations)

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

## GPU Monitoring

To verify that your host machine's GPU is successfully being utilized by the Docker containers, you can use the NVIDIA System Management Interface. While you can run `nvidia-smi` directly on your host machine, running it *inside* the container is the best way to confirm the hardware passthrough is functioning correctly.

**Basic Host Machine Monitoring:**
```sh
# for one time snapshot
nvidia-smi

# for continuous monitoring
watch -n 1 nvidia-smi
```

### 1. Find your Container Name
```sh
docker ps
```
_(Note: The simulation container is typically named `algorithmic-robots-world-ar-simulation-1`)_


### 2. Execute GPU Monitoring
Executes a one-time snapshot of your GPU's current status inside the container:
```sh
docker exec -it algorithmic-robots-world-ar-simulation-1 nvidia-smi
```
_Replace `algorithmic-robots-world-ar-simulation-1` with your container name, if it is different._


#### Continuous Monitoring
For real-time monitoring while running your simulation, use the `watch` command. The command below updates the interface every 1 second (you can adjust the time interval by changing the `-n 1` parameter):
```sh
docker exec -it algorithmic-robots-world-ar-simulation-1 watch -n 1 nvidia-smi
```


#### Expected Output:

You should see a table similar to the one below:
```sh
Mon Apr 00 18:00:00 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   57C    P8              4W /   35W |     645MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2945      G   /usr/bin/gnome-shell                      1MiB |
|    0   N/A  N/A           41228      G   /entry                                  633MiB |
+-----------------------------------------------------------------------------------------+
```
To confirm the hardware acceleration is working, check the Processes section at the bottom. You should see active processes (like `/entry` or your GUI elements) consuming GPU Memory.

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