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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++98-compat"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wswitch-enum"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wundefined-func-template"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsigned-enum-bitfield"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wunused-template"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif

namespace Utils {

  /**
   * @class NelderMead_minimizer
   * @brief Header-only implementation of Nelder-Mead simplex optimization with box constraints
   *
   * This class implements the Nelder-Mead simplex algorithm for derivative-free
   * optimization, with added support for box constraints using projection methods.
   *
   * ## Algorithm Overview
   *
   * The Nelder-Mead method is a simplex-based algorithm that uses geometric
   * operations (reflection, expansion, contraction, shrinkage) to navigate
   * the search space without requiring gradient information.
   *
   * ### Key Operations
   *
   * 1. **Reflection**: Reflect worst point through centroid of better points
   * 2. **Expansion**: If reflection is good, expand further in that direction
   * 3. **Contraction**: If reflection is poor, contract toward centroid
   * 4. **Shrinkage**: If all else fails, shrink simplex toward best point
   *
   * ### Box Constraints
   *
   * Box constraints are handled using projection:
   * - All simplex vertices are projected to feasible bounds
   * - Trial points are projected before function evaluation
   * - The algorithm maintains feasibility throughout optimization
   *
   * ## References
   *
   * -# J.A. Nelder and R. Mead (1965). "A Simplex Method for Function Minimization".
   *    Computer Journal, 7(4), 308-313. DOI: 10.1093/comjnl/7.4.308
   *
   * -# F. Gao and L. Han (2010). "Implementing the Nelder-Mead simplex algorithm
   *    with adaptive parameters". Computational Optimization and Applications,
   *    51(1), 259-277. DOI: 10.1007/s10589-010-9329-3
   *
   * -# M.A. Price, C.J. Pegg, and A.J. Keane (2018). "A hybridized Nelder-Mead
   *    simplex algorithm for constrained optimization". Engineering Optimization,
   *    50(6), 927-944. DOI: 10.1080/0305215X.2017.1361420
   *
   * ## Usage Example
   *
   * @code{.cpp}
   * using namespace Utils;
   *
   * // Define objective function (Rosenbrock)
   * auto rosenbrock = [](Vector const& x) -> double {
   *   return 100*(x(1)-x(0)*x(0))*(x(1)-x(0)*x(0)) + (1-x(0))*(1-x(0));
   * };
   *
   * // Setup minimizer
   * NelderMead_minimizer<double>::Options opts;
   * opts.max_iterations = 1000;
   * opts.tolerance = 1e-6;
   * opts.verbose = true;
   *
   * NelderMead_minimizer<double> minimizer(opts);
   *
   * // Set bounds (optional)
   * Vector lower(2), upper(2);
   * lower << -2.0, -1.0;
   * upper << 2.0, 3.0;
   * minimizer.set_bounds(lower, upper);
   *
   * // Initial point
   * Vector x0(2);
   * x0 << -1.2, 1.0;
   *
   * // Minimize
   * auto result = minimizer.minimize(x0, rosenbrock);
   *
   * if (result.status == NelderMead_minimizer<double>::Status::CONVERGED) {
   *   std::cout << "Solution: " << result.solution.transpose() << std::endl;
   *   std::cout << "Minimum: " << result.final_function_value << std::endl;
   * }
   * @endcode
   *
   * @tparam Scalar Floating-point type (double or float)
   *
   * @author Enrico Bertolazzi
   * @date 2025
   */

  template <typename Scalar = double>
  class NelderMead_minimizer {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Callback = std::function<Scalar(Vector const &)>;

    /**
     * @brief Optimization status codes
     */
    enum class Status {
      CONVERGED,              ///< Converged by function value tolerance
      MAX_ITERATIONS,         ///< Maximum iterations reached
      MAX_FUN_EVALUATIONS,    ///< Maximum function evaluations reached
      SIMPLEX_TOO_SMALL,      ///< Simplex diameter below tolerance
      FAILED                  ///< General failure
    };

    /**
     * @brief Nelder-Mead operation types
     */
    enum class Operation {
      INIT,                   ///< Initialization
      REFLECT,                ///< Reflection operation
      EXPAND,                 ///< Expansion operation
      OUTSIDE_CONTRACT,       ///< Outside contraction
      INSIDE_CONTRACT,        ///< Inside contraction
      SHRINK,                 ///< Shrink operation
      RESTART                 ///< Simplex restart
    };

    /**
     * @brief Optimization results structure
     */
    struct Result {
      Status status{Status::FAILED};              ///< Optimization status
      size_t iterations{0};                       ///< Number of iterations
      size_t function_evaluations{0};             ///< Number of function evaluations
      Scalar final_function_value{0};             ///< Best function value found
      Scalar initial_function_value{0};           ///< Initial function value
      Vector solution;                            ///< Best solution found
      Scalar simplex_volume{0};                   ///< Final simplex volume
      Scalar simplex_diameter{0};                 ///< Final simplex diameter
    };

    /**
     * @brief Optimization options
     */
    struct Options {
      size_t max_iterations{1000};                ///< Maximum iterations
      size_t max_function_evaluations{5000};      ///< Maximum function evaluations
      Scalar tolerance{1e-8};                     ///< Function value tolerance
      Scalar simplex_tolerance{1e-12};            ///< Simplex size tolerance
      Scalar volume_tolerance{0.01};              ///< Volume ratio for restart
      Scalar rho{1.0};                            ///< Reflection coefficient
      Scalar chi{2.0};                            ///< Expansion coefficient
      Scalar gamma{0.5};                          ///< Contraction coefficient
      Scalar sigma{0.5};                          ///< Shrink coefficient
      Scalar initial_step{0.1};                   ///< Initial simplex step size
      bool verbose{false};                        ///< Print progress information
      bool adaptive_parameters{true};             ///< Use adaptive parameters
    };

  private:
    Options m_options;                            ///< Optimization options
    Vector  m_lower;                              ///< Lower bounds
    Vector  m_upper;                              ///< Upper bounds
    bool    m_use_bounds{false};                  ///< Bounds activation flag

    // Simplex storage
    Matrix m_simplex;                             ///< Simplex vertices (n x n+1)
    Vector m_values;                              ///< Function values at vertices
    Vector m_centroid;                            ///< Centroid workspace
    Vector m_trial_point;                         ///< Trial point workspace

    // Internal state
    size_t m_dim{0};                              ///< Problem dimension
    Scalar m_eps{std::numeric_limits<Scalar>::epsilon()};

    /**
     * @brief Project point to feasible bounds
     */
    void project_point(Vector & x) const {
      if (m_use_bounds) {
        x = x.cwiseMax(m_lower).cwiseMin(m_upper);
      }
    }

    /**
     * @brief Initialize simplex using coordinate perturbation
     */
    void initialize_simplex(Vector const & x0, Callback const & callback) {
      m_dim = x0.size();
      m_simplex.resize(m_dim, m_dim + 1);
      m_values.resize(m_dim + 1);
      m_centroid.resize(m_dim);
      m_trial_point.resize(m_dim);

      // Set first vertex to initial point
      if (m_use_bounds) m_simplex.col(0) = x0.col(0).cwiseMax(m_lower).cwiseMin(m_upper);
      else              m_simplex.col(0) = x0;
  
      m_values(0) = callback(m_simplex.col(0));

      // Create other vertices by perturbing each coordinate
      for (size_t i = 0; i < m_dim; ++i) {
        m_simplex.col(i + 1) = x0;
        
        // Try positive perturbation first
        m_simplex(i, i + 1) += m_options.initial_step;
        if (m_use_bounds) m_simplex.col(i + 1) = m_simplex.col(i + 1).cwiseMax(m_lower).cwiseMin(m_upper);
        Scalar f_plus = callback(m_simplex.col(i + 1));

        // Try negative perturbation
        m_simplex(i, i + 1) = x0(i) - m_options.initial_step;
        if (m_use_bounds) m_simplex.col(i + 1) = m_simplex.col(i + 1).cwiseMax(m_lower).cwiseMin(m_upper);
        Scalar f_minus = callback(m_simplex.col(i + 1));

        // Keep the better perturbation
        if (f_plus < f_minus) {
          m_simplex(i, i + 1) = x0(i) + m_options.initial_step;
          if (m_use_bounds) m_simplex.col(i + 1) = m_simplex.col(i + 1).cwiseMax(m_lower).cwiseMin(m_upper);
          m_values(i + 1) = f_plus;
        } else {
          m_values(i + 1) = f_minus;
        }
      }
    }

    /**
     * @brief Update centroid (average of all but worst point)
     */
    void update_centroid(size_t worst_index) {
      m_centroid.setZero();
      for (size_t i = 0; i < m_dim + 1; ++i) {
        if (i != worst_index) {
          m_centroid += m_simplex.col(i);
        }
      }
      m_centroid /= static_cast<Scalar>(m_dim);
    }

    /**
     * @brief Compute simplex diameter
     */
    Scalar compute_diameter() const {
      Scalar max_dist = 0;
      for (size_t i = 0; i < m_dim + 1; ++i) {
        for (size_t j = i + 1; j < m_dim + 1; ++j) {
          Scalar dist = (m_simplex.col(i) - m_simplex.col(j)).norm();
          max_dist = std::max(max_dist, dist);
        }
      }
      return max_dist;
    }

    /**
     * @brief Compute approximate simplex volume
     */
    Scalar compute_volume() const {
      Matrix basis(m_dim, m_dim);
      for (size_t i = 0; i < m_dim; ++i) {
        basis.col(i) = m_simplex.col(i + 1) - m_simplex.col(0);
      }
      return std::abs(basis.determinant()) / std::tgamma(m_dim + 1);
    }

    /**
     * @brief Sort vertices by function value and return indices
     */
    std::vector<size_t> sort_vertices() const {
      std::vector<size_t> indices(m_dim + 1);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [this](size_t i, size_t j) { return m_values(i) < m_values(j); });
      return indices;
    }

    /**
     * @brief Perform reflection operation
     */
    Scalar reflect_point(Callback const & callback) {
      m_trial_point = m_centroid + m_options.rho * (m_centroid - m_simplex.col(m_worst));
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform expansion operation
     */
    Scalar expand_point(Callback const & callback) {
      m_trial_point = m_centroid + m_options.chi * (m_trial_point - m_centroid);
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform contraction operation
     */
    Scalar contract_point(Callback const & callback, bool outside = true) {
      if (outside) {
        m_trial_point = m_centroid + m_options.gamma * (m_trial_point - m_centroid);
      } else {
        m_trial_point = m_centroid - m_options.gamma * (m_centroid - m_simplex.col(m_worst));
      }
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform shrink operation
     */
    void shrink_simplex(Callback const & callback) {
      Vector best = m_simplex.col(m_best);
      for (size_t i = 0; i < m_dim + 1; ++i) {
        if (i != m_best) {
          m_simplex.col(i) = best + m_options.sigma * (m_simplex.col(i) - best);
          if (m_use_bounds) m_simplex.col(i) = m_simplex.col(i).cwiseMax(m_lower).cwiseMin(m_upper);
          m_values(i) = callback(m_simplex.col(i));
        }
      }
    }

    // Indices for best, worst, second worst points
    size_t m_best, m_worst, m_second_worst;

  public:
    /**
     * @brief Construct Nelder-Mead minimizer with given options
     */
    explicit NelderMead_minimizer(Options const & opts = Options())
    : m_options(opts) {}

    /**
     * @brief Set box constraints for optimization
     *
     * @param lower Lower bounds vector
     * @param upper Upper bounds vector
     *
     * @pre lower.size() == upper.size()
     * @pre lower[i] <= upper[i] for all i
     */
    void set_bounds(Vector const & lower, Vector const & upper) {
      UTILS_ASSERT(
        lower.size() == upper.size(),
        "NelderMead_minimizer::set_bounds: lower and upper must have same size\n"
      );
      UTILS_ASSERT(
        (lower.array() <= upper.array()).all(),
        "NelderMead_minimizer::set_bounds: lower must be <= upper for all components\n"
      );

      m_lower = lower;
      m_upper = upper;
      m_use_bounds = true;
    }

    void
    set_bounds(size_t n, Scalar const lower[], Scalar const upper[]) {
      m_lower.resize(n);
      m_upper.resize(n);
      std::copy_n(lower, n, m_lower.data());
      std::copy_n(upper, n, m_upper.data());
      m_use_bounds = true;
    }

    /**
     * @brief Perform Nelder-Mead optimization
     *
     * @param x0 Initial starting point
     * @param callback Objective function f(x) to minimize
     *
     * @return Result structure containing optimization results
     */
    Result minimize(Vector const & x0, Callback const & callback) {
      Result result;
      result.initial_function_value = callback(x0);

      // Initialize simplex
      initialize_simplex(x0, callback);
      result.function_evaluations = m_dim + 1;

      // Adaptive parameter adjustment
      if (m_options.adaptive_parameters) {
        // Gao and Han (2010) adaptive parameters
        m_options.rho = 1.0;
        m_options.chi = 1.0 + 2.0 / m_dim;
        m_options.gamma = 0.75 - 1.0 / (2.0 * m_dim);
        m_options.sigma = 1.0 - 1.0 / m_dim;
      }

      // Main optimization loop
      for (result.iterations = 0; 
           result.iterations < m_options.max_iterations; 
           ++result.iterations) {

        // Check function evaluation limit
        if (result.function_evaluations >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        // Sort vertices and get indices
        auto indices = sort_vertices();
        m_best = indices[0];
        m_second_worst = indices[m_dim - 1];
        m_worst = indices[m_dim];

        // Store best solution
        result.solution = m_simplex.col(m_best);
        result.final_function_value = m_values(m_best);

        // Check convergence by function value spread
        Scalar value_range = m_values(m_worst) - m_values(m_best);
        if (value_range < m_options.tolerance) {
          result.status = Status::CONVERGED;
          break;
        }

        // Check simplex size
        result.simplex_diameter = compute_diameter();
        if (result.simplex_diameter < m_options.simplex_tolerance) {
          result.status = Status::SIMPLEX_TOO_SMALL;
          break;
        }

        // Update centroid (excluding worst point)
        update_centroid(m_worst);

        // Reflection
        Scalar f_reflect = reflect_point(callback);
        ++result.function_evaluations;

        if (m_options.verbose) {
          fmt::print("[NelderMead] iter={:4d} f_best={:12.6g} f_worst={:12.6g} diameter={:12.6g}\n",
                     result.iterations, m_values(m_best), m_values(m_worst),
                     result.simplex_diameter);
        }

        if (f_reflect < m_values(m_best)) {
          // Expansion
          Scalar f_expand = expand_point(callback);
          ++result.function_evaluations;

          if (f_expand < f_reflect) {
            m_simplex.col(m_worst) = m_trial_point;
            m_values(m_worst) = f_expand;
          } else {
            m_simplex.col(m_worst) = m_trial_point;
            m_values(m_worst) = f_reflect;
          }
        } else if (f_reflect < m_values(m_second_worst)) {
          // Accept reflection
          m_simplex.col(m_worst) = m_trial_point;
          m_values(m_worst) = f_reflect;
        } else {
          if (f_reflect < m_values(m_worst)) {
            // Outside contraction
            Scalar f_contract = contract_point(callback, true);
            ++result.function_evaluations;

            if (f_contract <= f_reflect) {
              m_simplex.col(m_worst) = m_trial_point;
              m_values(m_worst) = f_contract;
            } else {
              shrink_simplex(callback);
              result.function_evaluations += m_dim; // n new evaluations
            }
          } else {
            // Inside contraction
            Scalar f_contract = contract_point(callback, false);
            ++result.function_evaluations;

            if (f_contract < m_values(m_worst)) {
              m_simplex.col(m_worst) = m_trial_point;
              m_values(m_worst) = f_contract;
            } else {
              shrink_simplex(callback);
              result.function_evaluations += m_dim; // n new evaluations
            }
          }
        }

        // Check for volume-based restart
        result.simplex_volume = compute_volume();
        Scalar regular_volume = std::pow(m_options.initial_step, m_dim) / std::tgamma(m_dim + 1);
        
        if (result.simplex_volume / regular_volume < m_options.volume_tolerance) {
          if (m_options.verbose) {
            fmt::print("[NelderMead] Restarting simplex due to small volume\n");
          }
          // Restart around best point
          initialize_simplex(result.solution, callback);
          result.function_evaluations += m_dim + 1;
        }
      }

      // Final updates
      if (result.iterations >= m_options.max_iterations) {
        result.status = Status::MAX_ITERATIONS;
      }

      // Ensure best solution is returned
      auto indices = sort_vertices();
      m_best = indices[0];
      result.solution = m_simplex.col(m_best);
      result.final_function_value = m_values(m_best);
      result.simplex_volume = compute_volume();
      result.simplex_diameter = compute_diameter();

      if (m_options.verbose) {
        fmt::print("[NelderMead] Finished: {}\n", status_to_string(result.status));
        fmt::print("[NelderMead] Final: f={:12.6g}, iterations={}, evaluations={}\n",
                   result.final_function_value, result.iterations, result.function_evaluations);
      }

      return result;
    }

    /**
     * @brief Get the current best solution found
     *
     * @return const reference to the current best solution vector
     *
     * @note This method can be called after minimize() to get the solution,
     *       or during optimization if needed.
     */
    Vector solution() const { return m_simplex.col(m_best); }

    /**
     * @brief Convert status to string
     */
    static std::string status_to_string(Status status) {
      switch (status) {
        case Status::CONVERGED:           return "CONVERGED";
        case Status::MAX_ITERATIONS:      return "MAX_ITERATIONS";
        case Status::MAX_FUN_EVALUATIONS: return "MAX_FUN_EVALUATIONS";
        case Status::SIMPLEX_TOO_SMALL:   return "SIMPLEX_TOO_SMALL";
        case Status::FAILED:              return "FAILED";
        default:                          return "UNKNOWN";
      }
    }

    /**
     * @brief Convert operation to string
     */
    static std::string operation_to_string(Operation op) {
      switch (op) {
        case Operation::INIT:             return "INIT";
        case Operation::REFLECT:          return "REFLECT";
        case Operation::EXPAND:           return "EXPAND";
        case Operation::OUTSIDE_CONTRACT: return "OUTSIDE_CONTRACT";
        case Operation::INSIDE_CONTRACT:  return "INSIDE_CONTRACT";
        case Operation::SHRINK:           return "SHRINK";
        case Operation::RESTART:          return "RESTART";
        default:                          return "UNKNOWN";
      }
    }
  };

} // namespace Utils

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

//
// eof: Utils_NelderMead.hh
