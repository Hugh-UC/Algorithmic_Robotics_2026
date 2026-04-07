# Succulence Rover: Systems Overview

> **Your reference map for Weeks 5–10.** Come back to this whenever you need to see where your current work fits into the bigger picture.

## The Mission

Kevin is lost somewhere on the Mars surface. A drone swarm will search the grid and broadcast his coordinates. Your job: **build the rover's brain** — the software that lets it map an unknown environment and autonomously navigate to Kevin.

The rover starts knowing *nothing* about the world. It gets Kevin's grid location, then must:
1. **Map** obstacles as it drives using lidar
2. **Localise** itself accurately despite wheel slip
3. **Plan** a path to Kevin through the map
4. **Navigate** along that path, replanning as new obstacles are discovered

---

## System Architecture

```
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║                  "OPERATION FIND KEVIN" — SYSTEM MAP                   ║
 ╠══════════════════════════════════════════════════════════════════════════╣
 ║                                                                        ║
 ║  DRONE SWARM (Weeks 11-13)                                             ║
 ║  ┌─────────────────────────┐                                           ║
 ║  │  Drones search the grid │                                           ║
 ║  │  and locate Kevin       │──── Kevin's position (x, y) ────┐        ║
 ║  └─────────────────────────┘                                  │        ║
 ║                                                               │        ║
 ║ ══════════════════════════════════════════════════════════════════════  ║
 ║  ROVER AUTONOMY STACK                                         │        ║
 ║ ══════════════════════════════════════════════════════════════════════  ║
 ║                                                               │        ║
 ║  SENSORS                                                      │        ║
 ║  ┌──────────────┐  ┌──────────────┐                           │        ║
 ║  │  Wheel       │  │  2D Lidar    │                           │        ║
 ║  │  Encoders    │  │  (LaserScan) │                           │        ║
 ║  │  (/odom)     │  │  (/scan)     │                           │        ║
 ║  └──────┬───────┘  └──────┬───────┘                           │        ║
 ║         │                 │                                   │        ║
 ║         ▼                 │                                   │        ║
 ║  ┌──────────────────────────────────────────────────────┐     │        ║
 ║  │             SLAM (Weeks 5-8)                         │     │        ║
 ║  │  "Where am I? What does the world look like?"        │     │        ║
 ║  │                                                      │     │        ║
 ║  │  ┌──────────────────────┐                            │     │        ║
 ║  │  │ MOTION MODEL (Wk 5-6)│                            │     │        ║
 ║  │  │                      │  Predict pose from          │     │        ║
 ║  │  │ Odometry ──► Pose    │  wheel motion. Track         │     │        ║
 ║  │  │ + Uncertainty (Σ)    │  growing uncertainty.        │     │        ║
 ║  │  └────────┬─────────────┘                            │     │        ║
 ║  │           │ estimated pose                           │     │        ║
 ║  │           ▼                                          │     │        ║
 ║  │  ┌──────────────────────┐  ┌──────────────────────┐  │     │        ║
 ║  │  │ OCCUPANCY GRID (Wk 6)│◄─│ SCAN MATCHER  (Wk 7) │◄─┘     │        ║
 ║  │  │                      │  │                      │        │        ║
 ║  │  │ Pose + Scan ──►      │  │ Align consecutive    │        │        ║
 ║  │  │ Bayesian grid map    │  │ scans to correct     │        │        ║
 ║  │  │ (free / occupied /   │  │ the pose estimate    │        │        ║
 ║  │  │  unknown cells)      │  │ (better than odom)   │        │        ║
 ║  │  └──────────────────────┘  └──────────┬───────────┘        │        ║
 ║  │                                       │                    │        ║
 ║  │                                       ▼                    │        ║
 ║  │                           ┌────────────────────────┐       │        ║
 ║  │                           │ POSE GRAPH (Wk 8)      │       │        ║
 ║  │                           │                        │       │        ║
 ║  │                           │ Fuse odometry + scan   │       │        ║
 ║  │                           │ match constraints.     │       │        ║
 ║  │                           │ Optimise all poses     │       │        ║
 ║  │                           │ simultaneously.        │       │        ║
 ║  │                           │ Rebuild corrected map. │       │        ║
 ║  │                           └────────────┬───────────┘       │        ║
 ║  └────────────────────────────────────────┼───────────────────┘        ║
 ║                                           │                            ║
 ║              ┌────────────────────────────┐│                           ║
 ║              │                            ││                           ║
 ║              ▼                            ▼│                           ║
 ║     Corrected Robot Pose         Corrected Map                         ║
 ║     "I am HERE"                  "The world looks like THIS"           ║
 ║              │                            │                            ║
 ║              │                            ▼                │           ║
 ║  ┌───────────────────────────────────────────────────────┐ │           ║
 ║  │           PLANNING (Week 10)                          │ │           ║
 ║  │           "How do I get to Kevin?"                    │ │           ║
 ║  │                                                       │ │           ║
 ║  │  ┌────────────────────────────────────────────────┐   │ │           ║
 ║  │  │ A* PATH PLANNER                               │◄──┼─┘           ║
 ║  │  │                                               │   │ Kevin's     ║
 ║  │  │ • Takes: map + my pose + goal                 │   │ location    ║
 ║  │  │ • Unknown cells = free (optimistic!)          │   │             ║
 ║  │  │ • Finds shortest safe path                    │   │             ║
 ║  │  │ • Replans every 2s as map grows               │   │             ║
 ║  │  └───────────────────┬────────────────────────────┘   │             ║
 ║  └──────────────────────┼────────────────────────────────┘             ║
 ║                         │                                              ║
 ║                         ▼                                              ║
 ║                Planned Path (waypoints)                                ║
 ║                         │                                              ║
 ║  ┌──────────────────────┼────────────────────────────────┐             ║
 ║  │  NAVIGATION (Week 10)                                 │             ║
 ║  │  "Follow the path!"                                  │             ║
 ║  │                                                       │             ║
 ║  │  ┌────────────────────────────────────────────────┐   │             ║
 ║  │  │ PATH FOLLOWER                                 │   │             ║
 ║  │  │                                               │   │             ║
 ║  │  │ • Steers toward next waypoint                 │   │             ║
 ║  │  │ • Slows down for turns                        │   │             ║
 ║  │  │ • Stops when Kevin is reached                 │   │             ║
 ║  │  └───────────────────┬────────────────────────────┘   │             ║
 ║  └──────────────────────┼────────────────────────────────┘             ║
 ║                         │                                              ║
 ║                         ▼                                              ║
 ║                 Wheel Commands (/cmd_vel)                              ║
 ║                         │                                              ║
 ║                         ▼                                              ║
 ║                 ┌───────────────┐                                      ║
 ║                 │   WHEELS      │──── Robot moves ──► back to          ║
 ║                 │   (motors)    │     sensors (continuous loop)        ║
 ║                 └───────────────┘                                      ║
 ╚════════════════════════════════════════════════════════════════════════╝
```

### The Sense-Think-Act Loop

The whole system runs as a **continuous feedback loop**:

1. **Sense** — wheels and lidar produce raw data
2. **Map & Localise (SLAM)** — fuse sensor data into a corrected pose + map
3. **Plan** — A* finds the best path to Kevin on the current map
4. **Act** — follow the path, sending wheel commands
5. **Repeat** — new sensor data flows in, the map updates, the path replans

---

## What You Build, Week by Week

| Week | Component | The Question It Answers |
|------|-----------|------------------------|
| 5 | Motion Model | *"If I trust only my wheels, where do I think I am?"* |
| 6 | Occupancy Grid | *"Given my (drifty) pose, what does the map look like?"* |
| 7 | Scan Matcher | *"Can I correct my pose by comparing consecutive lidar scans?"* |
| 8 | Pose Graph SLAM | *"Can I find the best set of poses that satisfies all my measurements?"* |
| 10 | A* Planner + Navigator | *"What's the shortest safe path to Kevin, and how do I follow it?"* |

### Why This Order?

Each week fixes a problem created by the previous week:

- **Week 5:** Dead reckoning drifts — *"my position estimate is wrong!"*
- **Week 6:** Drifty poses make ghosted maps — *"my map has double walls!"*
- **Week 7:** Scan matching gives better relative poses — *"I can measure displacement directly from lidar"*
- **Week 8:** Pose graph fuses everything and optimises — *"walls snap into place!"*
- **Week 10:** Now we have a good map — *"plan a path and go find Kevin!"*

---

## ROS2 Package Structure

This is the final structure of your `succulence_rover_ros` package — files are released to you week by week.

```
succulence_rover_ros/
├── config/
│   └── params.yaml                    # All tunable parameters (grows each week)
│
├── launch/
│   ├── dead_reckoning.launch.py # Weeks 5-6: motion model + occupancy grid
│   └── slam.launch.py          # Weeks 7-8: full SLAM pipeline
│
├── succulence_rover_ros/              # Python source (your code lives here)
│   ├── __init__.py
│   ├── utils.py                       # SE(2) math toolkit (provided)
│   ├── motion_model.py                # Week 5-6: odometry processing + noise model
│   ├── occupancy_grid_mapper.py       # Week 6:   Bayesian grid mapping
│   ├── scan_matcher.py                # Week 7:   correlation-based scan alignment
│   ├── pose_graph.py                  # Week 8:   graph data structure
│   ├── graph_optimizer.py             # Week 8:   Gauss-Newton least-squares solver
│   └── slam_node.py                   # Week 8:   ties everything together
│
├── package.xml                        # ROS2 package manifest
├── setup.py                           # Entry points for ROS2 nodes
└── setup.cfg                          # Install configuration
```

### What Each File Does

| File | Week | Role | You Implement |
|------|------|------|---------------|
| `utils.py` | — | SE(2) math: pose compose, inverse, Jacobians, covariance propagation | Provided (taught in Week 4) |
| `motion_model.py` | 5-6 | ROS2 node. Reads odometry, tracks pose + growing uncertainty | `compute_motion_covariance()` |
| `occupancy_grid_mapper.py` | 6 | ROS2 node. Builds Bayesian occupancy grid from pose + lidar | `_ray_trace()`, `update()` |
| `scan_matcher.py` | 7 | Library. Aligns two scans via grid correlation search | `_build_local_grid()`, `_score_alignment()`, `match()`, `_estimate_covariance_from_hessian()` |
| `pose_graph.py` | 8 | Data structure. Stores poses (nodes) and constraints (edges) | `add_node()`, `add_edge()` |
| `graph_optimizer.py` | 8 | Solver. Gauss-Newton optimisation over the pose graph | `compute_error()`, `compute_jacobians()` (templated), `optimize()` (scaffolded) |
| `slam_node.py` | 8 | ROS2 node. Full SLAM pipeline: keyframes, scan match, optimise, rebuild map | `_should_add_keyframe()`, `_process_keyframe()` |

---

## ROS2 Node & Topic Map

### Robot Configuration

All topic and frame names are configurable in `config/params.yaml`. Switch between virtual and physical robots by changing these parameters:

| | Virtual (Gazebo Sim) | Physical (TurtleBot) |
|---|---|---|
| Odometry topic | `/succulence/odom` | `/odom` |
| Lidar topic | `/succulence/scan` | `/scan` |
| Odom frame | `succulence/odom` | `odom` |
| Base link frame | `succulence/base_link` | `base_link` |
| Lidar frame | `succulence/lidar_link` | `base_scan` |

### Weeks 5-6: Dead Reckoning + Occupancy Grid

Launch: `ros2 launch succulence_rover_ros dead_reckoning.launch.py`

```
 SIMULATION / PHYSICAL ROBOT
 ───────────────────────────────────────────────────────────────────
   /succulence/odom                    /succulence/scan
   (nav_msgs/Odometry)                 (sensor_msgs/LaserScan)
         │                                    │
         │                                    │
         ▼                                    │
 ┌──────────────────────┐                     │
 │  motion_model_node   │                     │
 │  (OdometryProcessor) │                     │
 │                      │                     │
 │  Integrates odom,    │                     │
 │  propagates Σ        │                     │
 └──────┬───────┬───────┘                     │
        │       │                             │
        │       │                             │
        │       ▼                             │
        │  /succulence/dead_reckoning/path    │
        │  (nav_msgs/Path)                    │
        │  [trajectory for RViz]              │
        │                                     │
        ▼                                     │
   /succulence/dead_reckoning/odometry        │
   (nav_msgs/Odometry)                        │
   [pose + 6×6 covariance for RViz ellipse]   │
        │                                     │
        │              ┌──────────────────────┘
        │              │
        ▼              ▼
 ┌─────────────────────────────┐
 │  occupancy_grid_mapper_node │
 │  (OccupancyGridMapperNode)  │
 │                             │
 │  Bayesian log-odds grid,    │
 │  Bresenham ray tracing      │
 └──────────────┬──────────────┘
                │
                ▼
   /succulence/map/odom_only
   (nav_msgs/OccupancyGrid)
   [map with ghost walls from drift]
```

### Weeks 7-8: Full SLAM

Launch: `ros2 launch succulence_rover_ros slam.launch.py`

```
 SIMULATION / PHYSICAL ROBOT
 ───────────────────────────────────────────────────────────────────
   /succulence/odom                    /succulence/scan
   (nav_msgs/Odometry)                 (sensor_msgs/LaserScan)
         │                                    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │    slam_node     │
              │    (SlamNode)    │
              │                  │
              │  Internally:     │
              │  ┌────────────┐  │
              │  │ Motion     │  │
              │  │ Model      │  │
              │  └─────┬──────┘  │
              │        ▼         │
              │  ┌────────────┐  │
              │  │ Scan       │  │
              │  │ Matcher    │  │
              │  └─────┬──────┘  │
              │        ▼         │
              │  ┌────────────┐  │
              │  │ Pose Graph │  │
              │  │ + Optimizer│  │
              │  └─────┬──────┘  │
              │        ▼         │
              │  ┌────────────┐  │
              │  │ Occupancy  │  │
              │  │ Grid       │  │
              │  └────────────┘  │
              └───┬────┬────┬────┘
                  │    │    │
                  │    │    │
                  ▼    │    ▼
 /succulence/map  │   /succulence/slam/path
 (OccupancyGrid)  │   (nav_msgs/Path)
 [corrected map]  │   [optimised trajectory]
                  │
                  ▼
   /succulence/slam/odometry
   (nav_msgs/Odometry)
   [SLAM-corrected pose + covariance]
```

### Week 10: Full System (SLAM + Planning + Navigation)

*Coming in Week 10. The planner and navigator nodes will consume the SLAM outputs above.*

```
 /succulence/map ──────────────────────────────►┌──────────────┐
 /succulence/slam/odometry ────────────────────►│ planner_node │
 /succulence/goal_pose (Kevin's location) ─────►│ (A* search)  │
                                                └──────┬───────┘
                                                       │
                                          /succulence/plan
                                          (nav_msgs/Path)
                                                       │
 /succulence/slam/odometry ───────────►┌───────────────┘
                                       │
                                       ▼
                               ┌──────────────────┐
                               │ navigator_node   │
                               │ (path follower)  │
                               └────────┬─────────┘
                                        │
                                        ▼
                                    /cmd_vel
                                (geometry_msgs/Twist)
                                        │
                                        ▼
                                   ROBOT WHEELS
```

---

## Static TF Transforms

Both launch files publish two static transforms needed by the system:

```
map ──(identity)──► succulence/odom    (or odom for physical robot)

succulence/base_link ──(identity)──► succulence/lidar_link
```

The lidar frame is aligned with base_link (no rotation). Beam angles use the standard ROS formula: `robot_theta + (angle_min + i * angle_increment)`.

---

## Verification

### Build the package
```bash
cd ~/AR_Test_ROS
colcon build --packages-select succulence_rover_ros
source install/setup.bash
```

### Check topics are flowing
```bash
ros2 topic list | grep succulence
ros2 topic hz /succulence/odom
ros2 topic hz /succulence/scan
```

### RViz2 display setup (Fixed Frame: `map`)

| # | Display Type | Topic | Notes |
|---|-------------|-------|-------|
| 1 | Odometry | `/succulence/dead_reckoning/odometry` | Enable Covariance checkbox — watch ellipse grow |
| 2 | Path | `/succulence/dead_reckoning/path` | Color: Red — dead-reckoning trajectory |
| 3 | Map | `/succulence/map/odom_only` | Week 6 — map with ghost walls |
| 4 | Path | `/succulence/slam/path` | Color: Green — SLAM trajectory (Week 8) |
| 5 | Map | `/succulence/map` | Week 8 — corrected map, no ghosting |
| 6 | Odometry | `/succulence/slam/odometry` | Week 8 — smaller covariance ellipse |
| 7 | Path | `/succulence/plan` | Color: Yellow — A* planned path (Week 10) |
