# ROADMAP

## ✅ Completed Architectural Upgrades

### 1. Gradient Descent Path Smoothing (The "Drunk Driving" Fix)
* **The Problem:** A* searches an 8-connected grid, resulting in extremely jagged paths consisting exclusively of straight lines and sharp 45° or 90° corners. When the path follower encountered these, the curvature-limited speed cap forced the robot to slam on the brakes.
* **The Solution:** Implemented a gradient descent smoothing algorithm in `planner_node.py`. The A* waypoints are iteratively pulled into a smooth B-Spline-like curve. The Pure Pursuit controller can now maintain a high, constant velocity ($v$) all the way to the goal without stuttering.

### 2. Weighted A* (High-Speed Planning)
* **The Problem:** Exhaustive A* search was too slow to react to dynamic obstacles (like Kevin) in real-time.
* **The Solution:** Artificially inflated the heuristic by multiplying it by a weight (`f = tentative_g + (epsilon * hscore)`). This transforms the search into Weighted A*, making it vastly greedier. The planner now runs in a fraction of the time, allowing us to hit a 5Hz replanning rate for instant reflex dodging.

### 3. Dual Gradient Costmaps (Wall Hugging Fix)
* **The Problem:** The original binary `inflate_obstacles` function created hard True/False walls, causing the robot to clip corners or refuse to plan through narrow doorways.
* **The Solution:** Replaced boolean grids with float gradients. Walls are `np.inf`, but the surrounding inflation cells carry a decaying penalty cost. By injecting this into the A* `tentative_g` calculation, the robot naturally prefers to drive straight down the middle of hallways. Includes a toggleable Local Costmap for dynamic obstacle avoidance.

### 4. Coarse-to-Fine Scan Matching
* **The Problem:** The original `match()` function used an exhaustive, brute-force 3D grid search ($O(N^3)$) that consumed massive amounts of CPU.
* **The Solution:** Rewrote the matcher to perform a "coarse" search with large steps to find the general peak, followed by a "fine" search strictly around that local peak. SLAM now runs flawlessly with significantly reduced computational overhead.

---

## 🚀 Future Enhancements

### 1. Dynamic Obstacle Tracking (Velocity Prediction)
Currently, our Local Costmap treats Kevin as a static wall every time it refreshes. If we implement a basic Kalman Filter to track moving clusters in the laser scan, we could project Kevin's velocity vector and add a "forward penalty" to the costmap, allowing the robot to route *behind* him instead of just stopping in front of him.

### 2. Adaptive Pure Pursuit Lookahead
Currently, `control.lookahead` is a fixed distance. We could map the lookahead distance dynamically to the robot's current linear velocity ($v$). At high speeds, the robot looks further ahead for smooth, sweeping turns. As it slows down for tight corridors, the lookahead dynamically shrinks to ensure strict path adherence.

### 3. Frontier Exploration (Auto-Mapping)
Instead of relying on a hardcoded coordinate to drive to, we could write an exploration node that analyzes the SLAM occupancy grid, identifies "Frontiers" (the boundary between known free space and unknown space), and continuously publishes the nearest frontier as the new A* goal until the entire lab is mapped.