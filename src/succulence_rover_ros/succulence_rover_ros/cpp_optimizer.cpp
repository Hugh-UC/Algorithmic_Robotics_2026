#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <cmath>
#include <iostream>

namespace py = pybind11;
using namespace Eigen;

// We use RowMajor to natively match Python/NumPy memory layout (Zero-Copy)
typedef Matrix<double, Dynamic, 3, RowMajor> NodesMatrix;
typedef Matrix<double, Dynamic, 14, RowMajor> EdgesMatrix;

inline double normalize_angle(double theta) {
    while (theta > M_PI) theta -= 2.0 * M_PI;
    while (theta < -M_PI) theta += 2.0 * M_PI;
    return theta;
}

NodesMatrix optimize_graph_cpp(
    Eigen::Ref<NodesMatrix> nodes, 
    Eigen::Ref<EdgesMatrix> edges, 
    std::vector<int> anchors, 
    int num_iterations, 
    double huber_k,
    double anchor_weight)
{
    int n = nodes.rows();
    int num_edges = edges.rows();
    int dim = 3 * n;

    for (int iter = 0; iter < num_iterations; ++iter) {
        SparseMatrix<double> H(dim, dim);
        VectorXd b = VectorXd::Zero(dim);
        
        std::vector<Triplet<double>> triplets;
        triplets.reserve(num_edges * 36 + anchors.size() * 3);

        // 1. Compute Constraints
        for (int i = 0; i < num_edges; ++i) {
            int f_id = static_cast<int>(edges(i, 0));
            int t_id = static_cast<int>(edges(i, 1));
            
            double mx = edges(i, 2);
            double my = edges(i, 3);
            double mtheta = edges(i, 4);

            Matrix3d omega;
            omega << edges(i, 5), edges(i, 6), edges(i, 7),
                     edges(i, 8), edges(i, 9), edges(i, 10),
                     edges(i, 11), edges(i, 12), edges(i, 13);

            double xi = nodes(f_id, 0), yi = nodes(f_id, 1), thi = nodes(f_id, 2);
            double xj = nodes(t_id, 0), yj = nodes(t_id, 1), thj = nodes(t_id, 2);

            double c = std::cos(thi);
            double s = std::sin(thi);
            double dx = xj - xi;
            double dy = yj - yi;

            // Residual
            Vector3d e;
            e(0) = c * dx + s * dy - mx;
            e(1) = -s * dx + c * dy - my;
            e(2) = normalize_angle(thj - thi - mtheta);

            // Huber Kernel
            double mahalanobis_sq = e.transpose() * omega * e;
            double mahalanobis_d = std::sqrt(mahalanobis_sq);
            double weight = 1.0;
            if (mahalanobis_d > huber_k) {
                weight = huber_k / mahalanobis_d;
            }
            Matrix3d omega_w = omega * weight;

            // Jacobians
            Matrix3d Ji, Jj;
            Ji << -c, -s, -s * dx + c * dy,
                   s, -c, -c * dx - s * dy,
                   0,  0, -1.0;
            Jj <<  c,  s, 0.0,
                  -s,  c, 0.0,
                   0,  0, 1.0;

            Matrix3d JiT_omega = Ji.transpose() * omega_w;
            Matrix3d JjT_omega = Jj.transpose() * omega_w;

            Matrix3d H_ii = JiT_omega * Ji;
            Matrix3d H_ij = JiT_omega * Jj;
            Matrix3d H_ji = JjT_omega * Ji;
            Matrix3d H_jj = JjT_omega * Jj;

            Vector3d b_i = JiT_omega * e;
            Vector3d b_j = JjT_omega * e;

            int idx_i = 3 * f_id;
            int idx_j = 3 * t_id;

            // Add to b vector and H triplets
            for (int r = 0; r < 3; ++r) {
                b(idx_i + r) += b_i(r);
                b(idx_j + r) += b_j(r);
                for (int col = 0; col < 3; ++col) {
                    triplets.emplace_back(idx_i + r, idx_i + col, H_ii(r, col));
                    triplets.emplace_back(idx_i + r, idx_j + col, H_ij(r, col));
                    triplets.emplace_back(idx_j + r, idx_i + col, H_ji(r, col));
                    triplets.emplace_back(idx_j + r, idx_j + col, H_jj(r, col));
                }
            }
        }

        // 2. Add Anchors
        for (int a_id : anchors) {
            int idx = a_id * 3;
            triplets.emplace_back(idx, idx, anchor_weight);
            triplets.emplace_back(idx + 1, idx + 1, anchor_weight);
            triplets.emplace_back(idx + 2, idx + 2, anchor_weight);
        }

        H.setFromTriplets(triplets.begin(), triplets.end());

        // 3. Levenberg-Marquardt Damping
        for (int i = 0; i < dim; ++i) {
            H.coeffRef(i, i) += 1e-5;
        }

        // 4. Solve H * dx = -b using Sparse Cholesky Factorization
        SimplicialLDLT<SparseMatrix<double>> solver;
        solver.compute(H);
        if (solver.info() != Success) {
            break; // Factorization failure, abort iteration
        }

        VectorXd dx_vec = solver.solve(-b);
        if (solver.info() != Success) {
            break; // Solve failure, abort iteration
        }

        // 5. Apply Updates
        for (int i = 0; i < n; ++i) {
            nodes(i, 0) += dx_vec(i * 3);
            nodes(i, 1) += dx_vec(i * 3 + 1);
            nodes(i, 2) = normalize_angle(nodes(i, 2) + dx_vec(i * 3 + 2));
        }

        // 6. Convergence check
        if (dx_vec.norm() < 1e-5) {
            break;
        }
    }
    return nodes; // Returns the updated Numpy Array back to Python instantly
}

// Pybind11 Python Binding
PYBIND11_MODULE(succulence_cpp_optimizer, m) {
    m.doc() = "C++ Eigen Backend for Pose Graph Optimization";
    m.def("optimize_graph_cpp", &optimize_graph_cpp, 
          "Runs LM Optimization on Pose Graph using Eigen Sparse",
          py::arg("nodes"), py::arg("edges"), py::arg("anchors"), 
          py::arg("num_iterations"), py::arg("huber_k"), py::arg("anchor_weight"),
          py::call_guard<py::gil_scoped_release>());    // Release GIL (Global Interpreter Lock), removes thread block on ROS 2 executor
}