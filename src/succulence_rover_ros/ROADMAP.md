# ROADMAP

## Edges:

- Scan to Scan edges,
- Scan to Map edges,


## Fixing the "Aggressive Slow Down" (Smooth Navigation)

### The Root Cause:
Look at astar.py. It searches an 8-connected grid. This means your generated path is extremely jagged, consisting exclusively of straight lines and sharp 45° or 90° corners.

### The Reaction:
Now look at path_follower.py. Whenever the robot reaches one of these artificial A* corners, heading_error spikes. The controller includes the line v *= max(0.0, np.cos(heading_error)) and a curvature-limited speed cap. The robot mathematically has to slam on the brakes to make the jagged turn, then floors it once it's straight again.

### How to fix it:
#### Easy Fix:
Increase control.lookahead in your params. A longer lookahead makes the robot "cut corners" and naturally smooths out the jagged A* path, keeping speeds high.

#### Advanced Fix (Competition Winner):
Build a Path Smoothing Node (or just add a function in planner_node.py). Before publishing the nav_msgs/Path, run the A* waypoints through a smoothing algorithm (like Gradient Descent path smoothing or generating a B-Spline). If the path is a smooth curve, the Pure Pursuit controller will maintain a high, constant v all the way to Kevin.


## Other Tricks
### Weighted A (Speed up planning by 10x):*
In astar.py, you calculate f = tentative_g + hscore. If you artificially inflate the heuristic by multiplying it by a weight (e.g., f = tentative_g + (1.5 * hscore)), it becomes Weighted A*. It will search drastically fewer cells, meaning your planner can run much faster (e.g., replan every 0.2 seconds instead of 1.0 seconds). Faster replanning means the robot reacts to pop-up obstacles instantly.

### Gradient Costmaps (Don't hug the walls):
Currently, your inflate_obstacles function creates a hard binary wall (True/False). If you change this to a gradient (e.g., cost is 100 at the wall, 50 a cell away, 10 two cells away) and add that cost to tentative_g in A*, the robot will naturally prefer to drive right down the middle of hallways. This allows you to safely crank up max_linear_v without worrying about clipping corners.

### Coarse-to-Fine Scan Matching:
Your match() function in scan_matcher.py uses an exhaustive, brute-force 3D grid search ($O(N^3)$). It eats massive amounts of CPU. If you rewrite it to do a "coarse" search (big steps), find the peak, and then do a "fine" search (small steps) just around that peak, SLAM will run flawlessly. You can then increase your map_publish_rate and control.rate_hz, making the whole robot vastly more responsive.