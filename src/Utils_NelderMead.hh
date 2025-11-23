/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022-2025                                                 |
 |                                                                          |
 |         , __                 , __                                        |
 |        /|/  \               /|/  \                                       |
 |         | __/ _   ,_         | __/ _   ,_                                |
 |         |   \|/  /  |  |   | |   \|/  /  |  |   |                        |
 |         |(__/|__/   |_/ \_/|/|(__/|__/   |_/ \_/|/                       |
 |                           /|                   /|                        |
 |                           \|                   \|                        |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: Utils_NelderMead.hh
//

#pragma once

#ifndef UTILS_NELDER_MEAD_dot_HH
#define UTILS_NELDER_MEAD_dot_HH

#include "Utils.hh"
#include "Utils_fmt.hh"
#include "Utils_eigen.hh"

#include <functional>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <string>
#include <vector>
#include <iostream>

namespace Utils {

  using std::abs;
  using std::vector;
  using std::string;

  // ===========================================================================
  // ENUMS & RESULT STRUCT
  // ===========================================================================

  enum class NelderMeadStatus {
    RUNNING,
    CONVERGED,              // Function tolerance satisfied
    CONVERGED_TOL_X,        // Simplex size tolerance satisfied
    MAX_ITERATIONS,         // Max iterations reached
    MAX_FUN_EVALUATIONS,    // Max evaluations reached
    FAIL_NAN,               // NaN detected
    FAILED
  };

  inline std::string status_to_string(NelderMeadStatus s) {
    switch (s) {
      case NelderMeadStatus::RUNNING: return "RUNNING";
      case NelderMeadStatus::CONVERGED: return "CONVERGED";
      case NelderMeadStatus::CONVERGED_TOL_X: return "CONVERGED_X";
      case NelderMeadStatus::MAX_ITERATIONS: return "MAX_ITER";
      case NelderMeadStatus::MAX_FUN_EVALUATIONS: return "MAX_EVAL";
      case NelderMeadStatus::FAIL_NAN: return "FAIL_NAN";
      case NelderMeadStatus::FAILED: return "FAILED";
      default: return "UNKNOWN";
    }
  }

  template <typename VectorType, typename ScalarType>
  struct NelderMeadResult {
    VectorType solution;
    ScalarType final_function_value{0};
    ScalarType initial_function_value{0};
    NelderMeadStatus status{NelderMeadStatus::FAILED};

    // Statistics
    size_t outer_iterations{0};      // Main loop cycles
    size_t inner_iterations{0};      // Total simplex steps (sum of all sub-runs)
    size_t total_iterations{0};      // Combined count

    size_t outer_evaluations{0};
    size_t inner_evaluations{0};
    size_t total_evaluations{0};
  };

  // ===========================================================================
  // CLASS: NelderMead_classic (Inner Solver)
  // ===========================================================================

  template <typename Scalar = double>
  class NelderMead_classic {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Callback = std::function<Scalar(Vector const &)>;
    using Result   = NelderMeadResult<Vector, Scalar>;

    struct Options {
      size_t max_iterations{5000};
      size_t max_function_evaluations{10000};
      Scalar tolerance{1e-8};
      Scalar initial_step{0.05}; 
      bool   verbose{false};
    };

  private:
    Options m_options;
    Vector  m_lower, m_upper;
    bool    m_use_bounds{false};
    size_t  m_evals_count{0};

    Scalar safe_evaluate(Vector const & x, Callback const & cb) {
      if (m_use_bounds) {
        if ((x.array() < m_lower.array()).any() || (x.array() > m_upper.array()).any()) {
          return std::numeric_limits<Scalar>::infinity();
        }
      }
      Scalar val = cb(x);
      m_evals_count++;
      return val;
    }

  public:
    explicit NelderMead_classic(Options const & opts = Options()) : m_options(opts) {}

    void set_options(Options const & opts) { m_options = opts; }
    void set_bounds(Vector const & l, Vector const & u) { m_lower = l; m_upper = u; m_use_bounds = true; }

    Result minimize(Vector const & x0, Callback const & callback) {
      Result res;
      size_t dim = x0.size();
      m_evals_count = 0;

      // Standard Nelder-Mead parameters
      const Scalar rho = 1.0;  // Reflection
      const Scalar chi = 2.0;  // Expansion
      const Scalar gam = 0.5;  // Contraction
      const Scalar sig = 0.5;  // Shrink

      // Data structures
      std::vector<Vector> simplex(dim + 1);
      std::vector<Scalar> f_values(dim + 1);
      std::vector<size_t> idx(dim + 1); 

      // 1. Build Initial Simplex
      simplex[0] = x0;
      f_values[0] = safe_evaluate(simplex[0], callback);

      for (size_t i = 0; i < dim; ++i) {
        Vector next = x0;
        // Robust initial step
        Scalar step = (std::abs(x0(i)) < 1e-6) ? 0.00025 : m_options.initial_step * x0(i);
        next(i) += step;
        simplex[i+1] = next;
        f_values[i+1] = safe_evaluate(next, callback);
      }

      // Sort helper
      std::iota(idx.begin(), idx.end(), 0);
      auto sort_simplex = [&]() {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { 
            if (std::isnan(f_values[a])) return false; // Push NaNs to end
            if (std::isnan(f_values[b])) return true;
            return f_values[a] < f_values[b]; 
        });
      };
      sort_simplex();

      res.initial_function_value = f_values[idx[0]];
      size_t iter = 0;

      // Optimization Loop
      for (; iter < m_options.max_iterations; ++iter) {
        if (m_evals_count >= m_options.max_function_evaluations) {
          res.status = NelderMeadStatus::MAX_FUN_EVALUATIONS;
          break;
        }

        // Convergence Check
        Scalar f_best  = f_values[idx[0]];
        Scalar f_worst = f_values[idx.back()];
        if (std::abs(f_worst - f_best) < m_options.tolerance) {
          res.status = NelderMeadStatus::CONVERGED;
          break;
        }

        // Centroid
        Vector centroid = Vector::Zero(dim);
        for (size_t i = 0; i < dim; ++i) centroid += simplex[idx[i]];
        centroid /= static_cast<Scalar>(dim);

        // Reflection
        Vector xr = centroid + rho * (centroid - simplex[idx.back()]);
        Scalar fr = safe_evaluate(xr, callback);

        if (fr < f_values[idx[0]]) {
          // Expansion
          Vector xe = centroid + chi * (xr - centroid);
          Scalar fe = safe_evaluate(xe, callback);
          if (fe < fr) { 
            simplex[idx.back()] = xe; f_values[idx.back()] = fe; 
          } else { 
            simplex[idx.back()] = xr; f_values[idx.back()] = fr; 
          }
        } else if (fr < f_values[idx[dim - 1]]) {
          // Accept Reflection
          simplex[idx.back()] = xr; f_values[idx.back()] = fr;
        } else {
          // Contraction
          bool contraction_ok = false;
          if (fr < f_values[idx.back()]) {
            // Outside
            Vector xc = centroid + gam * (xr - centroid);
            Scalar fc = safe_evaluate(xc, callback);
            if (fc <= fr) {
              simplex[idx.back()] = xc; f_values[idx.back()] = fc;
              contraction_ok = true;
            }
          } else {
            // Inside
            Vector xc = centroid + gam * (simplex[idx.back()] - centroid);
            Scalar fc = safe_evaluate(xc, callback);
            if (fc < f_values[idx.back()]) {
              simplex[idx.back()] = xc; f_values[idx.back()] = fc;
              contraction_ok = true;
            }
          }

          if (!contraction_ok) {
            // Shrink
            Vector x_best = simplex[idx[0]];
            for (size_t i = 1; i <= dim; ++i) {
               size_t k = idx[i];
               simplex[k] = x_best + sig * (simplex[k] - x_best);
               f_values[k] = safe_evaluate(simplex[k], callback);
            }
          }
        }
        sort_simplex();
      }

      if (res.status == NelderMeadStatus::FAILED && iter >= m_options.max_iterations) 
         res.status = NelderMeadStatus::MAX_ITERATIONS;

      res.solution = simplex[idx[0]];
      res.final_function_value = f_values[idx[0]];
      res.inner_iterations = iter;
      res.inner_evaluations = m_evals_count;
      res.total_iterations = iter;
      res.total_evaluations = m_evals_count;

      return res;
    }
  };

  // ===========================================================================
  // CLASS: NelderMead_BlockCoordinate (Outer Solver)
  // ===========================================================================

  template <typename Scalar = double>
  class NelderMead_BlockCoordinate {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Callback = std::function<Scalar(Vector const &)>;
    using Result   = NelderMeadResult<Vector, Scalar>;

    struct Options {
      size_t block_size{10};
      size_t max_outer_iterations{100};
      size_t max_function_evaluations{100000};
      Scalar tolerance{1e-6};
      bool   verbose{true};
      typename NelderMead_classic<Scalar>::Options sub_options;
    };

  private:
    Options m_options;
    NelderMead_classic<Scalar> m_solver;
    Vector m_lower, m_upper;
    bool m_use_bounds{false};

    // Cyclic block selection
    vector<size_t> select_block(size_t n_dims, size_t iter) {
      vector<size_t> indices;
      indices.reserve(m_options.block_size);
      size_t start = (iter * m_options.block_size) % n_dims;
      for (size_t i = 0; i < m_options.block_size; ++i) indices.push_back((start + i) % n_dims);
      std::sort(indices.begin(), indices.end());
      indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
      return indices;
    }

  public:
    explicit NelderMead_BlockCoordinate(Options const & opts = Options()) : m_options(opts) {
      m_solver.set_options(m_options.sub_options);
    }

    void set_bounds(Vector const & l, Vector const & u) { m_lower = l; m_upper = u; m_use_bounds = true; }

    Result minimize(Vector x, Callback const & global_callback) {
      size_t n = x.size();

      // 1. Check for small problem fallback
      if ( 2*n <= 1+m_options.block_size) {
          if (m_options.verbose) {
              fmt::print("   [Info] Dim <= BlockSize. Switching to DIRECT CLASSIC solver.\n");
          }
          auto full_opts = m_options.sub_options;
          full_opts.max_function_evaluations = m_options.max_function_evaluations;
          full_opts.max_iterations = m_options.max_function_evaluations; 
          full_opts.tolerance = m_options.tolerance;
          full_opts.verbose = false;
          
          m_solver.set_options(full_opts);
          if (m_use_bounds) m_solver.set_bounds(m_lower, m_upper);
          return m_solver.minimize(x, global_callback);
      }
      
      // 2. Block Coordinate Logic
      Result res;
      size_t count_outer_evals = 0;
      size_t count_inner_evals = 0;
      size_t count_inner_iters = 0;

      Scalar current_f = global_callback(x);
      count_outer_evals++;
      res.initial_function_value = current_f;

      if (std::isnan(current_f) || std::isinf(current_f)) {
          res.status = NelderMeadStatus::FAIL_NAN;
          res.solution = x;
          return res;
      }

      size_t outer_iter = 0;
      bool converged = false;
      
      // Stagnation control
      size_t no_improv_count = 0;
      size_t blocks_per_cycle = (n + m_options.block_size - 1) / m_options.block_size;

      if (m_options.verbose) {
         fmt::print("   {:<5} | {:<8} | {:<14} | {:<10} | {:<10}\n", 
                    "Iter", "BlockDim", "F Value", "In.Iter", "In.Eval");
         fmt::print("   {:-<60}\n", "");
      }

      while (outer_iter < m_options.max_outer_iterations && !converged) {
        outer_iter++;
        if (count_outer_evals + count_inner_evals >= m_options.max_function_evaluations) {
          res.status = NelderMeadStatus::MAX_FUN_EVALUATIONS;
          break;
        }

        vector<size_t> idxs = select_block(n, outer_iter - 1);
        size_t k = idxs.size();

        // Prepare Subspace
        Vector x_sub(k), l_sub(k), u_sub(k);
        for (size_t i = 0; i < k; ++i) {
           x_sub(i) = x(idxs[i]);
           if (m_use_bounds) { l_sub(i) = m_lower(idxs[i]); u_sub(i) = m_upper(idxs[i]); }
        }
        if (m_use_bounds) m_solver.set_bounds(l_sub, u_sub);

        // Proxy Callback
        auto sub_cb = [&](Vector const & sub_v) -> Scalar {
          count_inner_evals++;
          Vector x_temp = x; 
          for (size_t i = 0; i < k; ++i) x_temp(idxs[i]) = sub_v(i);
          return global_callback(x_temp);
        };

        // Run Inner Solver
        auto sub_res = m_solver.minimize(x_sub, sub_cb);
        count_inner_iters += sub_res.inner_iterations;

        // Update Global State
        Scalar improvement = current_f - sub_res.final_function_value;
        if (sub_res.final_function_value < current_f) {
           current_f = sub_res.final_function_value;
           for (size_t i = 0; i < k; ++i) x(idxs[i]) = sub_res.solution(i);
        }

        if (m_options.verbose) {
           fmt::print("   {:<5} | {:<8} | {:<14.6e} | {:<10} | {:<10}\n", 
                      outer_iter, k, current_f, sub_res.inner_iterations, sub_res.inner_evaluations);
        }

        // Stagnation Check Logic
        if (improvement > m_options.tolerance) {
            no_improv_count = 0; // Reset counter on success
        } else {
            no_improv_count++;
        }

        // Stop only if we cycled through ALL blocks twice without significant improvement
        if (no_improv_count >= blocks_per_cycle * 2) {
            converged = true;
        }
      }

      res.solution = x;
      res.final_function_value = current_f;
      res.outer_iterations = outer_iter;
      res.inner_iterations = count_inner_iters;
      res.total_iterations = outer_iter + count_inner_iters;
      res.outer_evaluations = count_outer_evals;
      res.inner_evaluations = count_inner_evals;
      res.total_evaluations = count_outer_evals + count_inner_evals;
      
      if (res.status == NelderMeadStatus::FAILED) {
          if (converged) res.status = NelderMeadStatus::CONVERGED;
          else if (outer_iter >= m_options.max_outer_iterations) res.status = NelderMeadStatus::MAX_ITERATIONS;
      }
      return res;
    }
  };
}

#endif
