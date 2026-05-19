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
* **The Problem:** The original `match()` function used an exhaustive, brute-force 3D grid search $(O(N^3))$ that consumed massive amounts of CPU.
* **The Solution:** Rewrote the matcher to perform a "coarse" search with large steps to find the general peak, followed by a "fine" search strictly around that local peak. SLAM now runs flawlessly with significantly reduced computational overhead.

### 5. Adaptive Pure Pursuit Lookahead
* **The Problem:** The pure pursuit controller previously used a fixed lookahead distance. This caused the robot to either cut corners too sharply at high speeds or oscillate wildly in tight corridors.
* **The Solution:** Mapped the lookahead distance dynamically to the robot's current linear velocity ($v$). At high speeds, the lookahead extends for smooth, sweeping arcs. As the robot brakes or navigates tight spaces, the lookahead dynamically shrinks to ensure strict path adherence and prevent wall collisions.

---

## 🚀 Future Enhancements

### 1. Dynamic Obstacle Tracking (Velocity Prediction)
Currently, our Local Costmap treats Kevin as a static wall every time it refreshes. If we implement a basic Kalman Filter to track moving clusters in the laser scan, we could project Kevin's velocity vector and add a "forward penalty" to the costmap, allowing the robot to route *behind* him instead of just stopping in front of him.
<br>

### 2. Frontier Exploration (Auto-Mapping)
Instead of relying on a hardcoded coordinate to drive to, we could write an exploration node that analyzes the SLAM occupancy grid, identifies "Frontiers" (the boundary between known free space and unknown space), and continuously publishes the nearest frontier as the new A* goal until the entire lab is mapped.
<br>

### 3. Move Global Parameters into Global Definition
**For Example:**
```yaml
/**:
  ros__parameters:
    goal:             # Acts like a global variable for all nodes
      x: 0.45
      y: 2.35
      tolerance: 0.05
```
**OR**
```yaml
/**:
  ros__parameters:
    occupancy_grid:
      resolution: 0.025
      width: 300
      height: 300
```

Essentially, any param that will always be the same between files (nodes), should be declared like this.
I.e. if the both nodes depend on a same definition (e.g. 'occupancy_grid.resolution') and the code requires
these deifintions to be the same (or atleast highly preferable) they should be made into global parameters,
to reduce declaration redundancies and making parameter tweeking simpler.
<br>

### 4. Robust Loss Functions (M-Estimators)
**The Problem:** The current pose graph optimizer uses standard Non-Linear Least Squares (Gauss-Newton). Because the error is squared, a single massive outlier (e.g., a false-positive scan match caused by a human walking past the Lidar) has a disproportionately large pull on the graph, which can warp the map.
**The Solution:** Implement a Robust M-Estimator (such as the Huber or Cauchy loss function) to mathematically inoculate the pose graph against catastrophic failure.

**Implementation Steps:**
1. Update `graph_optimizer.py` to intercept the raw error residuals before they are squared.
2. Define a threshold parameter (`tuning_k`) in `params.yaml` to distinguish between normal noise and statistical outliers.
3. Implement the Huber piecewise function: If the error is less than `tuning_k`, apply standard quadratic scaling. If it exceeds `tuning_k`, scale the penalty linearly.
4. Down-weight the Information Matrix ($\Omega$) for outlier edges during the Hessian assembly step.
<br>

### 5. Levenberg-Marquardt (LM) Optimization
**The Problem:** In featureless environments (like long, blank corridors), the Lidar cannot determine forward motion. Mathematically, this causes the Hessian matrix ($H$) to become singular or ill-conditioned. If Gauss-Newton attempts to invert it, the math "explodes," causing the robot to jump off the map.
**The Solution:** Upgrade the graph optimizer from Gauss-Newton to the Levenberg-Marquardt algorithm to guarantee mathematical stability in degenerate environments.

**Implementation Steps:**
1. Modify the linear solver equation from standard Gauss-Newton to the LM formulation: `delta_x = (H + lambda * I)^(-1) * b`.
2. Introduce the dynamic damping parameter (`lambda`) to the diagonal of the Hessian matrix.
3. Implement the control loop: Evaluate the new system error after a step. If the error drops, decrease `lambda` (acting like fast Gauss-Newton). If the error rises, increase `lambda` (smoothly transitioning into safe Gradient Descent).
<br>

### 6. Sub-Grid Quadratic Interpolation for Scan Matching
**The Problem:** The correlative scan matcher evaluates scores on a discrete grid (currently `0.025m`). The SLAM accuracy is mathematically bottlenecked by this cell size, preventing sub-centimeter localization without exponentially increasing CPU load.
**The Solution:** Achieve millimeter-level accuracy on a coarse grid using Taylor Expansion and 2D surface fitting.

**Implementation Steps:**
1. Execute the standard discrete correlative search to find the highest-scoring pixel.
2. Extract the 3x3 matrix of correlation scores immediately surrounding this peak pixel.
3. Fit a 2D paraboloid (quadratic surface) to these 9 discrete points.
4. Calculate the first derivatives of the surface equation and set them to zero to find the exact continuous mathematical peak, yielding sub-pixel `x` and `y` offsets.
<br>

### 7. Eigenvalue-Driven Dynamic Covariance
**The Problem:** Currently, the `map_match_weight` applies a static, circular covariance to scan-to-map matches. In a straight hallway, the robot is highly certain of its lateral distance to the walls, but highly uncertain of its forward progress, meaning a circular covariance is mathematically incorrect.
**The Solution:** Dynamically stretch the covariance ellipse based on the geometry of the surrounding environment.

**Implementation Steps:**
1. After a scan match completes, extract the 3x3 Hessian matrix of the score surface.
2. Calculate the eigenvalues and eigenvectors of the Hessian.
3. Identify degenerate dimensions (e.g., an eigenvalue near zero indicates a featureless direction like a hallway vector).
4. Dynamically scale the Information Matrix along these specific eigenvectors, forcing the optimizer to trust wheel odometry for forward translation while strictly trusting the Lidar for lateral/rotational alignment.
<br>

### 8. Multi-Resolution Branch and Bound (B&B) Search
**The Problem:** The current "Coarse-to-Fine" search is a heuristic. If the true optimal alignment lies inside a very sharp, narrow mathematical peak, the coarse step might step completely over it, forcing the fine step to optimize the wrong local minimum.
**The Solution:** Implement a Branch and Bound search algorithm to mathematically guarantee finding the absolute global optimum while remaining computationally faster than brute-force methods.

**Implementation Steps:**
1. Pre-compute a "pyramid" of down-sampled occupancy grids (e.g., generate 5cm, 10cm, and 20cm resolution maps from the base 2.5cm map).
2. Evaluate candidate poses at the lowest resolution (top of the pyramid) to establish mathematical upper-bound scores.
3. Compare branch upper-bounds against the highest known valid score.
4. Prune (discard) entire branches of the search tree if their maximum possible upper-bound is lower than the current best score, safely shrinking the $O(N^3)$ search space.
