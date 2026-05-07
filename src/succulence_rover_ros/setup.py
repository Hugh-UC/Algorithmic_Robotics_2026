from setuptools import setup
import os
from glob import glob

package_name = 'succulence_rover_ros'

setup(
    name=package_name,
    version='0.1.0',
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
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maleen Jayasuriya',
    maintainer_email='maleen.jayasuriya@canberra.edu.au',
    description='SLAM and navigation package for the Succulence Rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Week 5: Dead reckoning + occupancy grid mapping
            'motion_model_node = succulence_rover_ros.motion_model:main',
            'occupancy_grid_mapper_node = succulence_rover_ros.occupancy_grid_mapper:main',
            # Week 11: Pose graph SLAM
            'slam_node = succulence_rover_ros.slam_node:main',
            # Week 12: A* planner + pure-pursuit navigator
            'planner_node = succulence_rover_ros.planner_node:main',
            'navigator_node = succulence_rover_ros.navigator_node:main',
        ],
    },
)
