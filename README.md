# Algorithmic_Robotics_2026

**Contains lab documetation and packages for Algorithmic Robotics course - [CollaborativeRoboticsLab](https://github.com/CollaborativeRoboticsLab/algorithmic-robots-world)**

**Group ID:** 

    Group_001

**Student ID's:**

    Hugh:     u3276400
    Patrick:  u3279178
    Caleb:    u3275593

***

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