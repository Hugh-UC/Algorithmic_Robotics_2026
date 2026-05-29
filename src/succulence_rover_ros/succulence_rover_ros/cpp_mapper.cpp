#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

void rebuild_map_cpp(
    py::array_t<float> grid_array,
    py::array_t<double> poses_array,
    py::array_t<double> scans_array,
    double origin_x, double origin_y, double resolution,
    double angle_min, double angle_increment,
    double range_min, double range_max,
    double l_occ, double l_free,
    double l_max, double l_min,
    double lidar_x_offset, double lidar_y_offset, double lidar_yaw_offset,
    double fov_half_rad, double edge_trim_rad
) {
    // Safely map memory as a 2D array, allowing native grid(y, x) indexing
    // This automatically handles memory strides regardless of Numpy's backend layout.
    auto grid = grid_array.mutable_unchecked<2>();
    auto poses = poses_array.unchecked<2>();
    auto scans = scans_array.unchecked<2>();

    int h = grid.shape(0); 
    int w = grid.shape(1);

    int num_poses = poses.shape(0);
    int num_rays = scans.shape(1);

    // Release GIL (Global Interpreter Lock), removes thread block on ROS 2 executor
    py::gil_scoped_release release;

    // Loop through every historical keyframe
    for (int i = 0; i < num_poses; ++i) {
        double px = poses(i, 0);
        double py = poses(i, 1);
        double pth = poses(i, 2);

        // Rigidity transformation to find the physical lidar sensor center
        double lx = px + lidar_x_offset * std::cos(pth) - lidar_y_offset * std::sin(pth);
        double ly = py + lidar_x_offset * std::sin(pth) + lidar_y_offset * std::cos(pth);
        double lth = pth + lidar_yaw_offset;

        int x0 = static_cast<int>(std::floor((lx - origin_x) / resolution));
        int y0 = static_cast<int>(std::floor((ly - origin_y) / resolution));

        // Ray Trace every laser beam
        for (int j = 0; j < num_rays; ++j) {
            double r = scans(i, j);
            
            // 1. Skip invalid, too-close, or too-far readings
            if (std::isnan(r) || std::isinf(r) || r < range_min || r > range_max) continue;

            // 2. Hardware FOV Cutoff: Ignore rays behind the robot (> 135 degrees)
            double local_angle = angle_min + j * angle_increment;
            double norm_angle = std::fmod(local_angle + M_PI, 2.0 * M_PI);

            if (norm_angle < 0) norm_angle += 2.0 * M_PI;
            norm_angle -= M_PI;
            
            // Ignored rays behind the robot defined by yaml parameters
            if (std::abs(norm_angle) > (fov_half_rad - edge_trim_rad)) continue;

            // 3. Solid hit raytracing
            double ray_angle = lth + local_angle;
            double end_x = lx + r * std::cos(ray_angle);
            double end_y = ly + r * std::sin(ray_angle);

            int x1 = static_cast<int>(std::floor((end_x - origin_x) / resolution));
            int y1 = static_cast<int>(std::floor((end_y - origin_y) / resolution));

            // Bresenham's Line Algorithm
            int dx = std::abs(x1 - x0);
            int dy = std::abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx - dy;

            int cx = x0;
            int cy = y0;

            while (true) {
                // Bounds checking against grid dimensions
                if (cx >= 0 && cx < w && cy >= 0 && cy < h) {
                    if (cx == x1 && cy == y1) {
                        grid(cy, cx) += l_occ;
                        if (grid(cy, cx) > l_max) grid(cy, cx) = l_max;
                        break;  // Stop ray at the wall
                    } else {
                        // SUBTRACT free space to hollow out paths
                        grid(cy, cx) -= l_free;
                        if (grid(cy, cx) < l_min) grid(cy, cx) = l_min;
                    }
                } else {
                    break;  // Ray left the map boundary
                }

                if (cx == x1 && cy == y1) break;

                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    cx += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    cy += sy;
                }
            }
        }
    }
}

// Pybind11 Registration
PYBIND11_MODULE(succulence_cpp_mapper, m) {
    m.doc() = "C++ High-Performance Ray Tracing for Occupancy Grid Mapping";
    m.def("rebuild_map_cpp", &rebuild_map_cpp, "Rebuilds grid using bare-metal Bresenham ray tracing");
}