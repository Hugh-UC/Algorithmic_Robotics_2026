# Succulence Rover Navigation Stack & Parameters

The autonomous stack relies on precisely tuned parameters located in `params_sim.yaml` (for simulation) or `params_physical.yaml` (for the real robot). The parameters control how each core ROS 2 node behaves.

---

### Release (v2.0)

---

## Table of Contents

- [🏗️ Architectural Upgrades (v2.0)](#️-architectural-upgrades-v20)
    - [1. Asynchronous Decoupled SLAM Architecture](#1-asynchronous-decoupled-slam-architecture)
    - [2. Multiprocessing Backend Optimisation](#2-multiprocessing-backend-optimisation)
    - [3. Unified Mission Launch Architecture](#3-unified-mission-launch-architecture)
    - [4. Layered Costmaps over Bayesian Occupancy Grid](#4-layered-costmaps-over-bayesian-occupancy-grid)
    - [5. Pose Graph SLAM: Scan-to-Map Edge Matching](#5-pose-graph-slam-scan-to-map-edge-matching)
    - [6. Adaptive Pure Pursuit Lookahead](#6-adaptive-pure-pursuit-lookahead)
    - [7. Dual-Stage Safety Shield (Reactive Braking)](#7-dual-stage-safety-shield-reactive-braking)
- [🚀 Mission Launch Instructions](#-mission-launch-instructions)
    - [⚡ General Startup](#-general-startup)
    - [Launch Mission](#launch-mission)
    - [🧪 Advanced Testing Flags](#-advanced-testing-flags)
- [⚙️ Parameter Tuning Guide (`params.yaml`)](#️-parameter-tuning-guide-paramsyaml)
    - [🏎️ Control & Speed (Navigator Node)](#️-control--speed-navigator-node)
    - [🛑 Safety Shield (Navigator Node)](#-safety-shield-navigator-node)
    - [🧠 Path Planning (Planner Node)](#-path-planning-planner-node)
    - [🛡️ Environmental Awareness (Costmaps)](#️-environmental-awareness-costmaps)
    - [🗺️ Localization & SLAM (Slam Node)](#️-localization--slam-slam-node)
- [📖 Node & Parameter Breakdown](#-node--parameter-breakdown)
    - [🚑 Common Issues & Fixes](#-common-issues--fixes)

---

## 🏗️ Architectural Upgrades (v2.0)

The Succulence stack has undergone a major overhaul to refine and upgrade **path planning** and **SLAM**.

### 1. Asynchronous Decoupled SLAM Architecture
- **How it works:** The monolithic SLAM node was split into two distinct nodes. `SlamEstimator` (The Brain) handles high-frequency scan matching and pose graph management, while `GlobalMapper` (The Artist) manages map construction in the background.
- **Benefit:** Eliminates pose "teleportation" and freezing. The robot never goes "blind" because the mapper performs atomic grid swaps only when the new map version is fully baked.

### 2. Multiprocessing Backend Optimisation
- **How it works:** Graph optimisation is CPU-bound and historically suffered from Python's Global Interpreter Lock (GIL), causing 8-second stutters. We offloaded the `graph_optimizer` to a `multiprocessing.Process` worker.
- **Benefit:** The main SLAM loop remains at high frequency (15Hz+) while the heavy matrix math runs on a separate CPU core, ensuring zero-latency odometry publishing even during graph snap-back.

### 3. Unified Mission Launch Architecture
- **How it works:** Deprecated redundant scripts (`mission_sim.launch.py`, `mission_physical.launch.py`) into a single `mission.launch.py` utilizing ROS 2 `LaunchConfiguration`.
- **Benefit:** Ensures perfect node synchronization. Developers can hot-swap parameter files and costmap architectures via CLI flags without modifying code (e.g., `ros2 launch succulence_rover_ros mission.launch.py mode:=physical costmap_mode:=local`).

### 4. Layered Costmaps over Bayesian Occupancy Grid
- **How it works:** Upgraded the foundational SLAM Occupancy Grid from a discrete binary map (0 or 1) to a probabilistic log-odds grid (0–100). On top of this, we introduced a dedicated Costmap overlay inside the A* Planner, which inflates raw obstacles into gradient penalty zones.
- **Global Costmap:** Inflates the static SLAM occupancy grid in the background. This acts as a permanent bounding aura that forces the macro-path into the center of safe corridors.
- **Local Costmap:** Injects the live Lidar `/scan` directly into the planner's grid on every tick. It applies a localized, temporary inflation halo to new laser hits, allowing the robot to route around dynamic obstacles (like humans) in real-time.
- **The Math:** Obstacle inflation applies a linear decay penalty based on the cell's distance (`d`) from the obstacle:
  `Cost(d) = max(existing_cost, inflation_weight * (1.0 - (d / inflation_radius_cells)))`
- **Benefit:** Gives the A* heuristic continuous mathematical gradients to evaluate. It prevents the robot from scraping walls and allows fluid dodging of moving objects rather than freezing in "No Path Found" states.

### 5. Pose Graph SLAM: Scan-to-Map Edge Matching
- **How it works:** Upgraded the SLAM optimizer from purely relative alignment (scan $N$ to scan $N-1$) to absolute environmental anchoring. The scan matcher now rasterizes a local window of the global map and scores the live Lidar scan against both the previous scan *and* the established room geometry.
- **The Math:** The trust ratio is controlled by `map_match_weight: 0.3` (30% trust in global architecture, 70% trust in frame-to-frame odometry).
- **Benefit:** Completely eliminates "Ghost Walls" and the "Orbit of Death" by forcing the pose graph to lock onto square room corners during violent rotations.

### 6. Adaptive Pure Pursuit Lookahead
- **How it works:** Deprecated the static `lookahead` distance. The pure-pursuit controller now dynamically shrinks and expands its tracking circle based on the robot's instantaneous forward velocity (`msg.twist.twist.linear.x`).
- **The Math:** `L = np.clip(|v| * lookahead_ratio, lookahead_min, lookahead_max)`
- **Benefit:** At high speeds ($0.3m/s$), the robot looks further ahead ($0.45m$) for smooth, sweeping arcs. When braking for tight corners ($0.1m/s$), the lookahead shrinks ($0.2m$) forcing strict path adherence and preventing dangerous corner-cutting.

### 7. Dual-Stage Safety Shield (Reactive Braking)
- **How it works:** Implemented a two-tier, hardware-level safety override directly in `navigator_node.py` that processes the raw `/scan` topic, bypassing the latency of the A* planner (~1.0s). Behaviors can be hot-swapped using the `safety_mode` launch flag.
- **Tier 1: Soft Collision Recovery:** If an obstacle enters the 40° forward Lidar cone within `0.25m`, forward speed ($v$) is locked to $0.0$, but angular speed ($w$) is permitted (capped at $0.3$ rad/s). This allows the robot to safely spin in place to face a newly calculated A* detour path without moving closer to the hazard.
- **Tier 2: Hard Emergency Lock:** If an obstacle breaches the critical `0.12m` threshold, all motors are forcibly locked ($v=0.0, w=0.0$) to act as a final perimeter defense for the physical chassis.
- **Benefit:** Provides professional-grade autonomous safety. It allows the robot to seamlessly and fluidly recover from nearby dynamic obstacles (e.g., humans walking past) while guaranteeing absolute hardware protection if an object gets dangerously close.

---

<br>

## 🚀 Mission Launch Instructions

The mission launch file brings up the entire autonomous stack (SLAM, A* Planner, and Pure Pursuit Navigator) and correctly manages TF trees, coordinate frames, and odometry resets based on the target environment.

### ⚡ General Startup
**Navigate to Workspace Sub-Directory:**
```bash
cd succulence_ws/
```

**Colcon Build Program:**
```bash
colcon build --packages-select succulence_rover_ros --symlink-install
source install/setup.bash
```

### Run Visualiser
**Run RViz2 w/ Config:**
```bash
rviz2 -d succulence_ws/src/succulence_rover_ros/config/succulance_costmap.rviz    # in 'workspace/'
```

### Launch Mission
**Run on the Physical TurtleBot:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical
```

**Run in Simulation (Unity Mars):**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=sim
```
_(Note: If no mode is provided, it defaults to `sim`.)_


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

---

<br>

## ⚙️ Parameter Tuning Guide (`params.yaml`)

### 🏎️ Control & Speed (Navigator Node)

| Parameter | Type | Default | Range | Unit | Description |
| --- | --- | --- | --- | --- | --- |
| `control.max_linear_v` | `float` | `0.46` | `0.2 - 0.7` | m/s | Maximum forward speed. Crank this up for faster competition times. |
| `control.max_angular_v` | `float` | `0.34` | `0.15 - 2.0` | rad/s | Maximum turning speed. Higher values allow taking corners without braking. |
| `control.lookahead_min` | `float` | `0.6` | `0.1 - 0.4` | m | Adaptive clamp. Strict path tracking distance for tight corridors/slow speeds. |
| `control.lookahead_max` | `float` | `8.0` | `0.4 - 1.5` | m | Adaptive clamp. Sweeping corner distance for high speeds. |
| `control.lookahead_ratio` | `float` | `2.5` | `1.0 - 3.0` | - | Multiplier determining how fast lookahead scales with linear velocity. |
| `control.rate_hz` | `float` | `15.0` | `10.0 - 30.0` | Hz | Motor command publish rate. Higher = smoother arc driving. |
| `control.goal_tolerance` | `float` | `0.10` | `0.05 - 0.5` | m | Stop radius around the goal. Physical inertia requires >= 0.15m to prevent start/stop orbiting and SLAM-jitter stuttering. |

### 🛑 Safety Shield (Navigator Node)

| Parameter | Type | Default | Range | Unit | Description |
| --- | --- | --- | --- | --- | --- |
| `safety.mode` | `string` | `"both"` | `"both", "collision", "emergency", "none"` | - | Determines which hardware-level brake overrides are active. |
| `safety.collision_brake_dist` | `float` | `0.25` | `0.15 - 0.5` | m | Soft brake threshold. Halts forward motion but allows pure pursuit to spin and recover. |
| `safety.recovery_spin_rate` | `float` | `0.3` | `0.1 - 0.5` | rad/s | Maximum turn speed allowed during a soft collision recovery spin. |
| `safety.emergency_brake_dist` | `float` | `0.12` | `0.05 - 0.2` | m | Hard brake threshold. Completely locks motors to protect hardware chassis. |
| `safety.forward_cone_angle` | `float` | `0.7` | `0.3 - 1.5` | rad | Width of the protective Lidar field-of-view (~40 degrees). |

### 🧠 Path Planning (Planner Node)

| Parameter | Type | Default | Range | Unit | Description |
| --- | --- | --- | --- | --- | --- |
| `planning.replan_period` | `float` | `1.0` | `0.1 - 3.0` | s | Time between recalculations. Lower to `0.2` for instant dynamic obstacle dodging. |
| `planning.heuristic_weight` | `float` | `1.0` | `1.0 - 2.5` | - | Epsilon multiplier for A*. `1.0` = optimal. `1.5+` = greedy/fast. |
| `planning.smooth_weight` | `float` | `0.85` | `0.1 - 0.8` | - | "Rubber band" strength. Higher pulls jagged A* corners into sweeping curves. |
| `planning.data_weight` | `float` | `0.08` | `0.0 - 0.5` | - | Path fidelity weight. Prevents the smoother from pulling the route into walls. |
| `planning.goal_smooth_dist` | `float` | `4.0` | `0.0 - 20.0` | cells | Distance from the goal below which smoothing is disabled to allow tight, accurate final approaches. |
| `planning.smooth_tolerance` | `float` | `0.01` | `0.001 - 0.1` | - | Numerical stopping condition for the gradient descent smoother. |
| `goal.tolerance` | `float` | `0.10` | `0.05 - 0.5` | m | Distance at which the planner suspends path generation (Must match Navigator). |

### 🛡️ Environmental Awareness (Costmaps)

| Parameter | Type | Default | Range | Unit | Description |
| --- | --- | --- | --- | --- | --- |
| `costmaps.mode` | `string` | `"both"` | `"both", "global", "local", "none"` | - | Determines which costmaps the planner generates and routes through. Useful for rapid testing. |
| `costmaps.global.inflation_radius_cells` | `float` | `20.0` | `10.0 - 20.0` | cells | Bounding penalty aura for permanent walls. Forces macro-path into the center of hallways. |
| `costmaps.global.inflation_weight` | `float` | `50.0` | `20.0 - 100.0` | - | Fear of touching a permanent wall. High values prevent dangerous shortcuts. |
| `costmaps.local.inflation_radius_cells` | `float` | `16.0` | `4.0 - 10.0` | cells | The penalty aura applied strictly to live dynamic laser hits. |
| `costmaps.local.inflation_weight` | `float` | `25.0` | `5.0 - 30.0` | - | Fear of dynamic objects. Kept lower than global so the robot doesn't panic. |
| `costmaps.local.max_obstacle_range` | `float` | `4.0` | `1.5 - 12.0` | m | Ignores laser hits further than this distance to save CPU during replanning. |
| `costmaps.local.min_obstacle_range` | `float` | `0.25` | `0.05 - 0.2` | m | Ignores laser hits closer than this distance to prevent self-collision mapping. |

### 🗺️ Localization & SLAM (Slam Node)

| Parameter | Type | Default | Range | Unit | Description |
| --- | --- | --- | --- | --- | --- |
| `slam.keyframe_distance` | `float` | `0.1` | `0.05 - 0.5` | m | Drops a graph node every 10cm (at 0.3 m/s = updates every ~0.3s). |
| `slam.keyframe_angle` | `float` | `0.1` | `0.05 - 0.5` | rad | Drops a node every ~6 degrees to perfectly track physical Create 3 base skidding. |
| `slam.map_match_weight` | `float` | `0.3` | `0.0 - 1.0` | - | Blends relative (scan-to-scan) tracking with absolute (scan-to-map) loop closure. |
| `slam.map_publish_interval` | `float` | `0.5` | `0.3 - 2.0` | s | Global SLAM map publish rate. Lowering to `0.3` repairs dynamic "ghost walls" faster. |
| `slam.scan_rate_limit` | `float` | `15.0` | `5.0 - 20.0` | Hz | Throttles incoming laser scans to prevent CPU locking. Matches the physical RPLIDAR-A1 spin rate. |
| `scan_matcher.search_theta` | `float` | `0.3` | `0.1 - 0.4` | rad | Search radius. Increase to `0.4` if the robot slips during high-speed turns. |
| `occupancy_grid.max_range` | `float` | `10.0` | `3.0 - 15.0` | m | View distance. Set equal to map size to allow SLAM to lock onto distant corners. |

---

<br>

## 📖 Node & Parameter Breakdown

### 🚑 Common Issues & Fixes

* **Symptom: The robot hugs walls or scrapes corners.**
  * **Tweak:** Increase `costmaps.global.inflation_weight` (e.g., to `60.0`) or `costmaps.global.inflation_radius_cells`.
  * **Why:** Increases the mathematical penalty of routing near walls, forcing the A* path to stay dead-center in the corridors.

* **Symptom: The robot freezes or says "No Path Found" in narrow doorways.**
  * **Tweak:** Decrease `costmaps.global.inflation_radius_cells` or `costmaps.local.inflation_weight`.
  * **Why:** Shrinks the perceived bounding box of the walls/obstacles, giving the planner a wider safe zone to calculate a route through tight spaces without the penalty exceeding the obstacle threshold.

* **Symptom: The robot wobbles, overshoots, or cuts corners too aggressively (Drunk Driving).**
  * **Tweak:** Decrease `control.lookahead_ratio` (e.g., to `1.2`) or clamp `control.lookahead_max` down to `0.4`.
  * **Why:** The new Adaptive Lookahead might be scaling too aggressively at high speeds. Shrinking it forces the Pure Pursuit controller to track the immediate A* path more strictly instead of sweeping wide, lazy arcs.

* **Symptom: The robot gets stuck in an infinite start/stop loop near the goal (Orbit of Death).**
  * **Tweak:** Ensure `goal.tolerance` in the Planner Node matches `control.goal_tolerance` in the Navigator Node, and increase them both (e.g., to `0.15`+).
  * **Why:** Physical inertia makes it mathematically impossible for a heavy TurtleBot to brake perfectly inside a 5cm target window at 0.3 m/s. A larger standoff distance absorbs this physical overshoot.

* **Symptom: The robot suddenly stops and refuses to move forward, even if A* shows a clear path.**
  * **Tweak:** Decrease `safety.emergency_brake_dist` (e.g., to `0.15`) or `safety.forward_cone_angle`.
  * **Why:** The hardware-level Reactive Emergency Braking shield is falsely triggering on Lidar noise, a stray cable on the floor, or the edge of a wall during a tight turn.

* **Symptom: The SLAM map "teleports" or hallucinates Ghost Walls while turning.**
  * **Tweak:** Lower `slam.keyframe_angle` to `0.05` and ensure `occupancy_grid.max_range` is at least `7.5`.
  * **Why:** Violent physical rotations cause the TurtleBot wheels to slip. Tightening the keyframes forces SLAM to evaluate the Lidar faster, while a high `max_range` allows the algorithm to anchor onto the square walls of the room instead of identical nearby rocks.

* **Symptom: The planner is lagging, or the CPU is maxing out.**
  * **Tweak:** Increase `planning.heuristic_weight` (e.g., to `1.5`+), lower `costmaps.local.max_obstacle_range` (e.g., to `2.0`), and ensure `slam.scan_rate_limit` is not exceeding `10.0`.
  * **Why:** Makes the A* search much faster (greedier) and stops the robot from burning CPU cycles processing distant, irrelevant dynamic laser hits or polling the Lidar hardware faster than it physically spins.

***