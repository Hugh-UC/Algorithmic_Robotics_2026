# Succulence Rover Navigation Stack & Parameters

The autonomous stack relies on precisely tuned parameters located in `params_sim.yaml` (for simulation) or `params_physical.yaml` (for the real robot). The parameters control how each core ROS 2 node behaves.

---

<br>

## ⚙️ Parameter Tuning Guide (`params.yaml`)

### 🏎️ Control & Speed (Navigator Node)
| Parameter | Type | Default | Range | Unit | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `control.max_linear_v` | `float` | `0.3` | `0.2 - 0.7` | m/s | Maximum forward speed. Crank this up for faster competition times. |
| `control.max_angular_v` | `float` | `0.75` | `0.5 - 2.0` | rad/s | Maximum turning speed. Higher values allow taking corners without braking. |
| `control.lookahead` | `float` | `1.0` | `0.5 - 1.5` | m | The pure-pursuit target distance. Lower = strict path tracking. Higher = sweeping corners. |
| `control.rate_hz` | `float` | `20.0` | `10.0 - 30.0` | Hz | Motor command publish rate. Higher = smoother arc driving. |
| `control.goal_tolerance`| `float` | `0.05` | `0.05 - 0.5` | m | Stop radius around the goal. For dynamic mapping environments, keep >= 0.35 to prevent SLAM-jitter stuttering. |

### 🧠 Path Planning (Planner Node)
| Parameter | Type | Default | Range | Unit | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `planning.replan_period` | `float` | `0.5` | `0.1 - 1.0` | s | Time between recalculations. Lower to `0.2` for instant dynamic obstacle dodging. |
| `planning.heuristic_weight` | `float` | `1.2` | `1.0 - 2.5` | - | Epsilon multiplier for A*. `1.0` = optimal. `1.5+` = greedy/fast. |
| `planning.smooth_weight` | `float` | `0.5` | `0.1 - 0.8` | - | "Rubber band" strength. Higher pulls jagged A* corners into sweeping curves. |
| `planning.data_weight` | `float` | `0.1` | `0.0 - 0.5` | - | Path fidelity weight. Prevents the smoother from pulling the route into walls. |
| `planning.goal_smooth_dist` | `float` | `2.0` | `0.0 - 10.0` | cells | Distance from the goal below which smoothing is disabled to allow tight, accurate final approaches. |
| `planning.smooth_tolerance` | `float` | `0.01`| `0.001 - 0.1` | - | Numerical stopping condition for the gradient descent smoother. |
| `goal.tolerance` | `float` | `0.05` | `0.05 - 0.5` | m | Distance at which the planner suspends path generation. |

### 🛡️ Environmental Awareness (Costmaps)
| Parameter | Type | Default | Range | Unit | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `costmaps.mode` | `string` | `"both"` | `"both", "global", "local", "none"` | - | Determines which costmaps the planner generates and routes through. Useful for rapid testing. |
| `costmaps.global.inflation_radius_cells` | `float`| `14.0` | `10.0 - 20.0` | cells | Bounding penalty aura for permanent walls. Forces macro-path into the center of hallways. |
| `costmaps.global.inflation_weight` | `float` | `50.0` | `20.0 - 100.0`| - | Fear of touching a permanent wall. High values prevent dangerous shortcuts. |
| `costmaps.local.inflation_radius_cells` | `float`| `8.0` | `4.0 - 10.0` | cells | The penalty aura applied strictly to live dynamic laser hits. |
| `costmaps.local.inflation_weight` | `float` | `15.0` | `5.0 - 30.0` | - | Fear of dynamic objects. Kept lower than global so the robot doesn't panic. |
| `costmaps.local.max_obstacle_range` | `float` | `3.0` | `1.5 - 12.0` | m | Ignores laser hits further than this distance to save CPU during replanning. |
| `costmaps.local.min_obstacle_range` | `float` | `0.1` | `0.05 - 0.2` | m | Ignores laser hits closer than this distance to prevent self-collision mapping. |

### 🗺️ Localization & SLAM (Slam Node)
| Parameter | Type | Default | Range | Unit | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `slam.map_match_weight` | `float` | `0.3` | `0.0 - 1.0` | - | Blends relative (scan-to-scan) tracking with absolute (scan-to-map) loop closure. |
| `slam.map_publish_interval`| `float` | `2.0` | `0.5 - 2.0` | s | Global SLAM map publish rate. Lowering to `0.5` repairs dynamic "ghost walls" faster. |
| `slam.scan_rate_limit` | `float` | `10.0`| `5.0 - 20.0`| Hz | Throttles incoming laser scans to prevent CPU locking. |
| `scan_matcher.search_theta`| `float` | `0.1` | `0.1 - 0.4` | rad | Search radius. Increase to `0.4` if the robot slips during high-speed turns. |

---

<br>

## 📖 Node & Parameter Breakdown

### 🚑 Common Issues & Fixes

* **Symptom: The robot hugs walls or scrapes corners.**
  * **Tweak:** Increase `costmaps.global.inflation_weight` (e.g., to `60.0`).
  * **Why:** Increases the mathematical penalty of routing near walls, forcing the A* path to stay dead-center in the corridors.

* **Symptom: The robot freezes or says "No Path Found" in narrow doorways.**
  * **Tweak:** Decrease `costmaps.global.inflation_radius_cells` or `costmaps.local.inflation_weight`.
  * **Why:** Shrinks the perceived bounding box of the walls/obstacles, giving the planner a wider safe zone to calculate a route through tight spaces.

* **Symptom: The robot wobbles, overshoots, or cuts corners too aggressively (Drunk Driving).**
  * **Tweak:** Decrease `control.lookahead` (e.g., to `0.3`) and ensure `control.max_angular_v` is high enough (`0.75`).
  * **Why:** A smaller lookahead forces the Pure Pursuit controller to track the immediate A* path more strictly instead of sweeping wide arcs. Higher angular velocity allows it to physically make the turn.

* **Symptom: The robot gets stuck in an infinite start/stop loop near the goal.**
  * **Tweak:** Ensure `goal.tolerance` in the Planner Node matches `control.goal_tolerance` in the Navigator Node, and increase them both (e.g., to `0.35`).
  * **Why:** SLAM noise from dynamic targets can cause the map coordinate frame to jump. A larger standoff distance absorbs this noise.

* **Symptom: The planner is lagging or CPU is maxing out.**
  * **Tweak:** Increase `planning.heuristic_weight` (e.g., to `1.5`+) and lower `costmaps.local.max_obstacle_range` (e.g., to `3.0`).
  * **Why:** Makes the A* search much faster (greedier) and stops the robot from burning CPU cycles processing distant, irrelevant dynamic laser hits.

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
You can rapidly override the costmap architecture without editing YAML files using the `costmap` flag. This is useful for debugging physical edge cases on the fly.

**Global Only | Disable the high-speed local reflex bubble:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical costmap_mode:=global
```

**Local Only | Disable the global SLAM map entirely:**
```bash
ros2 launch succulence_rover_ros mission.launch.py mode:=physical costmap_mode:=local
```

**Other Options:**
- `both`: Uses both local and global costmaps (default).
- `none`: Completely disables costmaps from SLAM.