from setuptools import setup, Extension
import os
from glob import glob

package_name = 'succulence_rover_ros'

# Postpone importing pybind11 until installed during build
class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()
    

# C++ Native Extension module declaration
ext_modules = [
    Extension(
        'succulence_cpp_optimizer',
        ['succulence_rover_ros/cpp_optimizer.cpp'],
        include_dirs=[
            str(get_pybind_include()),
            '/usr/include/eigen3'  # Standard Ubuntu/Docker path for Eigen
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3']
    ),
    Extension(
        'succulence_cpp_mapper',
        ['succulence_rover_ros/cpp_mapper.cpp'],
        include_dirs=[str(get_pybind_include())],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3']
    ),
]

# Setup function for ROS 2 package
setup(
    name=package_name,
    version='0.1.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        ('share/' + package_name + '/launch',
            glob(os.path.join('launch', '*.launch.py'))),
        # Install config files (YAML params + RViz layouts)
        ('share/' + package_name + '/config',
            glob(os.path.join('config', '*.yaml')) +
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools', 'pybind11'],
    ext_modules=ext_modules,    # Inject C++ compiler
    zip_safe=False,             # Must be False for C++ Extensions to load correctly
    maintainer='Hugh Brennan',
    maintainer_email='u3276400@uni.canberra.edu.au',
    description='SLAM and Navigation Package for the Succulence Rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Dead reckoning + occupancy grid mapping
            'motion_model_node = succulence_rover_ros.motion_model:main',
            'occupancy_grid_mapper_node = succulence_rover_ros.occupancy_grid_mapper:main',
            # Split Pose Graph SLAM Nodes (New Architecture)
            'slam_estimator_node = succulence_rover_ros.slam_estimator:main',
            'global_mapper_node = succulence_rover_ros.global_mapper:main',
            # A* planner + pure-pursuit navigator
            'planner_node = succulence_rover_ros.planner_node:main',
            'navigator_node = succulence_rover_ros.navigator_node:main',
        ],
    },
)
