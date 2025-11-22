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
#include <random>
#include <memory>

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

  using std::abs;
  using std::min;
  using std::max;
  using std::vector;
  using std::string;
  using std::pow;
  using std::exp;
  using std::sqrt;
  using std::lgamma;
  using std::log1p;

  /**
   * @class NelderMead_classic
   * @brief Implementation of the classic Nelder-Mead simplex optimization algorithm
   *
   * @tparam Scalar Numeric type for computations (default: double)
   *
   * This class implements the Nelder-Mead simplex algorithm for unconstrained
   * and bound-constrained optimization. It includes advanced features like
   * adaptive parameters, restart mechanisms, and robust convergence checking.
   */
  template <typename Scalar = double>
  class NelderMead_classic {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;    ///< Vector type
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;  ///< Matrix type
    using Callback = std::function<Scalar(Vector const &)>;     ///< Objective function type

    /**
     * @brief Optimization status codes
     */
    enum class Status {
      CONVERGED,              ///< Successfully converged to optimum
      MAX_ITERATIONS,         ///< Reached maximum iteration limit
      MAX_FUN_EVALUATIONS,    ///< Reached maximum function evaluation limit
      SIMPLEX_TOO_SMALL,      ///< Simplex became too small to continue
      STAGNATED,              ///< Algorithm stagnated (no improvement)
      FAILED                  ///< Optimization failed
    };

    /**
     * @brief Configuration options for Nelder-Mead algorithm
     */
    struct Options {
      // Global budget (including restarts)
      size_t max_iterations{10000};           ///< Maximum total iterations
      size_t max_function_evaluations{50000}; ///< Maximum function evaluations
      Scalar tolerance{1e-8};                 ///< Convergence tolerance
      Scalar stagnation_tolerance{1e-8};      ///< Stagnation detection tolerance
      Scalar simplex_tolerance{1e-10};        ///< Minimum simplex size tolerance
      
      // Standard Nelder-Mead parameters
      Scalar rho{1.0};        ///< Reflection coefficient
      Scalar chi{2.0};        ///< Expansion coefficient
      Scalar gamma{0.5};      ///< Contraction coefficient
      Scalar sigma{0.5};      ///< Shrink coefficient
      
      Scalar initial_step{0.1};   ///< Initial step size for simplex construction
      
      bool   adaptive_parameters{true}; ///< Enable adaptive parameter adjustment
      bool   verbose{false};            ///< Enable verbose output
      size_t progress_frequency{100};   ///< Progress reporting frequency
      
      // Restart mechanism
      bool   enable_restart{true};         ///< Enable restart strategy
      size_t max_restarts{5};              ///< Maximum number of restarts
      size_t stagnation_threshold{30};     ///< Iterations before stagnation detection
      bool   use_relative_tolerance{true}; ///< Use relative convergence criteria
      
      Scalar restart_perturbation_ratio{0.25}; ///< Restart perturbation scale
      bool   track_best_point{true};           ///< Track best point across restarts
        
      // Enhanced convergence parameters
      Scalar min_step_size{1e-10};         ///< Minimum step size to avoid degeneracy
      bool   use_robust_convergence{true}; ///< Use robust convergence criteria
      Scalar convergence_relaxation{10.0}; ///< Convergence criterion relaxation factor
      
      // Restart condition thresholds
      Scalar restart_relative_improvement_threshold{0.05};   ///< 5% min improvement for restart
      Scalar restart_progress_per_eval_threshold{1e-6};      ///< Progress per evaluation threshold
      Scalar restart_simplex_geometry_threshold{0.3};        ///< Improvement threshold for small simplex
      Scalar restart_shrink_count_threshold{8};              ///< Shrink operations before restart
      Scalar restart_after_shrink_improvement{0.02};         ///< Improvement after shrink threshold
      Scalar restart_expected_progress_ratio{0.05};          ///< Expected progress ratio for high dimensions
      Scalar restart_quality_metric_threshold{1e-4};         ///< Quality metric threshold
      Scalar restart_degenerate_improvement_threshold{0.1};  ///< Improvement threshold for degenerate simplex
      Scalar restart_improvement_ratio{0.95};                ///< Improvement ratio to accept restart
      Scalar restart_absolute_improvement_threshold{0.1};    ///< Absolute improvement threshold
      
      // Geometry factors for restart conditions
      Scalar restart_simplex_diameter_factor1{50.0};         ///< Diameter factor for condition 3
      Scalar restart_simplex_diameter_factor2{20.0};         ///< Diameter factor for condition 7
      Scalar restart_std_dev_factor{10.0};                   ///< Standard deviation factor
    };

    /**
     * @brief Optimization result structure
     */
    struct Result {
      Vector solution;                    ///< Best solution found
      Status status{Status::FAILED};      ///< Optimization status
      size_t iterations{0};               ///< Total iterations performed
      size_t function_evaluations{0};     ///< Total function evaluations
      Scalar final_function_value{0};     ///< Function value at solution
      Scalar initial_function_value{0};   ///< Initial function value
      Scalar simplex_volume{0};           ///< Final simplex volume
      Scalar simplex_diameter{0};         ///< Final simplex diameter
      size_t restarts_performed{0};       ///< Number of restarts performed
      size_t shrink_operations{0};        ///< Number of shrink operations
    };

  private:
    /**
     * @brief Statistics for simplex analysis
     */
    struct SimplexStats {
      Scalar diameter;          ///< Maximum distance between vertices
      Scalar std_dev;           ///< Standard deviation of function values
      Scalar value_range;       ///< Range of function values (max-min)
      Scalar centroid_distance; ///< Average distance to centroid
    };

    Options m_options;            ///< Algorithm configuration
    Vector  m_lower;              ///< Lower bounds (if used)
    Vector  m_upper;              ///< Upper bounds (if used)
    bool    m_use_bounds{false};  ///< Whether bounds are active
    
    Callback const * m_callback{nullptr};  ///< Objective function callback
    size_t m_global_iterations{0};         ///< Global iteration counter
    size_t m_global_evals{0};              ///< Global function evaluation counter
    
    vector<Vector> m_simplex;              ///< Simplex vertices
    vector<Scalar> m_values;               ///< Function values at vertices
    Vector m_centroid;                     ///< Current centroid (excluding worst)
    Vector m_trial_point;                  ///< Trial point for operations
    size_t m_dim{0};                       ///< Problem dimension

    mutable bool           m_simplex_ordered{false}; ///< Whether simplex is sorted
    mutable vector<size_t> m_sorted_indices;         ///< Indices sorted by function value

    size_t m_stagnation_count{0};          ///< Consecutive stagnation iterations
    Scalar m_previous_best{std::numeric_limits<Scalar>::max()}; ///< Previous best value
    size_t m_shrink_count{0};              ///< Shrink operation counter
    
    Vector m_best_point;                   ///< Best point found (across restarts)
    Scalar m_best_value{std::numeric_limits<Scalar>::max()}; ///< Best value found
    
    Scalar m_current_rho, m_current_chi, m_current_gamma, m_current_sigma; ///< Current adaptive parameters

    /**
     * @brief Safely evaluate objective function with bounds checking
     * @param x Point to evaluate
     * @return Function value or large value if out of bounds/invalid
     */
    Scalar
    safe_evaluate( Vector const & x ) {
      UTILS_ASSERT(m_callback != nullptr, "NelderMead_classic::safe_evaluate(x) Callback not set!");
        
      // Check bounds before evaluation
      if ( m_use_bounds ) {
        bool out_of_bound = (x.array() < m_lower.array()).any() ||
                            (x.array() > m_upper.array()).any();
        if (out_of_bound) {
          if (m_options.verbose) {
            fmt::print("Warning: Point outside bounds, returning large value\n");
          }
          return std::numeric_limits<Scalar>::max();
        }
      }
        
      Scalar value{(*m_callback)(x)};
      ++m_global_evals;
        
      if (!std::isfinite(value)) {
        if (m_options.verbose) {
          fmt::print("Warning: Non-finite function value at x={}\n", x.transpose());
        }
        return std::numeric_limits<Scalar>::max();
      }
      return value;
    }

    /**
     * @brief Project point to feasible region (respect bounds)
     * @param x Point to project (modified in-place)
     */
    void
    project_point(Vector & x) const {
      if (m_use_bounds) x = x.cwiseMax(m_lower).cwiseMin(m_upper);
    }

    /**
     * @brief Initialize adaptive parameters based on problem dimension
     */
    void
    initialize_adaptive_parameters() {
      if (m_options.adaptive_parameters && m_dim > 0) {
        Scalar n = static_cast<Scalar>(m_dim);
        // More conservative parameters for high-dimensional problems
        if (m_dim > 10) {
          m_current_rho   = 1.0;
          m_current_chi   = 1.0 + 1.0 / n;  // Less aggressive expansion
          m_current_gamma = 0.75 - 0.5 / n;
          m_current_sigma = 0.8 - 0.5 / n;  // Less aggressive shrink
        } else {
          m_current_rho   = 1.0;
          m_current_chi   = 1.0 + 2.0 / n;
          m_current_gamma = 0.75 - 0.5 / n;
          m_current_sigma = 1.0 - 1.0 / n;
        }
      } else {
        m_current_rho   = m_options.rho;
        m_current_chi   = m_options.chi;
        m_current_gamma = m_options.gamma;
        m_current_sigma = m_options.sigma;
      }
    }

    /**
     * @brief Get sorted indices of simplex vertices by function value
     * @return Const reference to sorted indices vector
     */
    vector<size_t> const &
    get_sorted_indices() const {
      if (!m_simplex_ordered) {
        m_sorted_indices.resize(m_dim + 1);
        std::iota(m_sorted_indices.begin(), m_sorted_indices.end(), 0);
        std::sort(m_sorted_indices.begin(), m_sorted_indices.end(),
                  [this](size_t i, size_t j) { return m_values[i] < m_values[j]; });
        m_simplex_ordered = true;
      }
      return m_sorted_indices;
    }

    /**
     * @brief Mark simplex as unordered (needs re-sorting)
     */
    void
    mark_simplex_unordered() {
      m_simplex_ordered = false;
    }

    /**
     * @brief Compute smart step size for simplex initialization
     * @param dimension_index Dimension index
     * @param current_val Current coordinate value
     * @return Appropriate step size
     */
    Scalar
    get_smart_step(size_t dimension_index, Scalar current_val) const {
      Scalar step = m_options.initial_step;
        
      if (m_use_bounds) {
        Scalar lower = m_lower(dimension_index);
        Scalar upper = m_upper(dimension_index);
            
        if (std::isfinite(lower) && std::isfinite(upper)) {
          Scalar range = upper - lower;
          if (range > std::numeric_limits<Scalar>::epsilon()) {
            step = std::clamp(range * 0.05, m_options.min_step_size, Scalar(10.0));
          }
        } else if (std::isfinite(lower) && !std::isfinite(upper)) {
          Scalar dist_from_lower = current_val - lower;
          if (dist_from_lower > 0) {
            step = max(m_options.initial_step, dist_from_lower * 0.1);
          }
        } else if (!std::isfinite(lower) && std::isfinite(upper)) {
          Scalar dist_from_upper = upper - current_val;
          if (dist_from_upper > 0) {
            step = max(m_options.initial_step, dist_from_upper * 0.1);
          }
        }
      }
        
      Scalar abs_val = abs(current_val);
      if (abs_val > 1.0) step = max(m_options.initial_step, abs_val * 0.05);
        
      return max(step, m_options.min_step_size);
    }

    /**
     * @brief Initialize simplex around starting point
     * @param x0 Starting point
     */
    void
    initialize_simplex(Vector const & x0) {
      m_dim = x0.size();
        
      m_simplex.resize(m_dim + 1);
      m_values.resize(m_dim + 1);
      m_centroid.resize(m_dim);
      m_trial_point.resize(m_dim);
      m_sorted_indices.resize(m_dim + 1);

      for (auto & vec : m_simplex) vec.resize(m_dim);

      m_simplex[0] = x0;
      project_point(m_simplex[0]);
      m_values[0] = safe_evaluate(m_simplex[0]);
        
      Vector x_base   = m_simplex[0];
      Scalar min_step = m_options.min_step_size;
        
      for ( size_t i{0}; i < m_dim; ++i ) {
        Scalar step = get_smart_step(i, x_base(i));

        // Ensure minimum step to avoid degenerate simplex
        if (abs(step) < min_step) step = (step >= 0) ? min_step : -min_step;

        Vector x_next = x_base;
        x_next(i) += step;

        project_point(x_next);

        // Verify point is different from base point
        if ((x_next - x_base).norm() < min_step) {
          // If still too close, try opposite direction
          x_next = x_base;
          x_next(i) -= step;
          project_point(x_next);
        }

        m_simplex[i + 1] = x_next;
        m_values[i + 1] = safe_evaluate(x_next);
      }

      mark_simplex_unordered();
        
      // Verify simplex is not degenerate
      if (compute_volume() < std::numeric_limits<Scalar>::epsilon()) {
        if (m_options.verbose) {
          fmt::print("  [Warning] Initial simplex has zero volume, adding perturbation\n");
        }
        // Add small random perturbation to avoid degeneracy
        for (size_t i{1}; i <= m_dim; ++i) {
          Vector perturbation = Vector::Random(m_dim) * min_step * 10;
          m_simplex[i] += perturbation;
          project_point(m_simplex[i]);
          m_values[i] = safe_evaluate(m_simplex[i]);
        }
      }

      if (m_options.track_best_point) {
        auto indices = get_sorted_indices();
        if (m_values[indices[0]] < m_best_value) {
          m_best_value = m_values[indices[0]];
          m_best_point = m_simplex[indices[0]];
        }
        m_previous_best = m_best_value;
      }
    }

    /**
     * @brief Update centroid excluding worst point
     * @param worst_index Index of worst point to exclude
     */
    void
    update_centroid(size_t worst_index) {
      m_centroid.setZero();
      for (size_t i{0}; i <= m_dim; ++i) {
        if (i != worst_index) m_centroid += m_simplex[i];
      }
      m_centroid /= static_cast<Scalar>(m_dim);
    }

    /**
     * @brief Compute simplex diameter (maximum vertex distance)
     * @return Simplex diameter
     */
    Scalar
    compute_diameter() const {
      Scalar max_dist{0};
      for (size_t i{0}; i <= m_dim; ++i) {
        for (size_t j{i + 1}; j <= m_dim; ++j) {
          Scalar dist = (m_simplex[i] - m_simplex[j]).norm();
          max_dist = max(max_dist, dist);
        }
      }
      return max_dist;
    }

    /**
     * @brief Compute simplex volume
     * @return Simplex volume (approximated for high dimensions)
     */
    Scalar
    compute_volume() const {
      if (m_dim == 0) return 0;
      if (m_dim > 100) return std::pow(compute_diameter(), m_dim) * std::exp(-std::lgamma(m_dim+1));
    
      Matrix basis(m_dim, m_dim);
      for (size_t i{0}; i < m_dim; ++i) {
        basis.col(i) = m_simplex[i + 1] - m_simplex[0];
      }
    
      Eigen::ColPivHouseholderQR<Matrix> qr(basis);
      if (qr.rank() < m_dim) return 0;
    
      Scalar log_det   = qr.logAbsDeterminant();
      Scalar log_gamma = std::lgamma(m_dim + 1);
    
      return std::exp(log_det - log_gamma);
    }

    /**
     * @brief Compute comprehensive simplex statistics
     * @return Simplex statistics structure
     */
    SimplexStats
    compute_simplex_stats() const {
      SimplexStats stats;
        
      stats.diameter = compute_diameter();
        
      // Use Eigen for efficient mean computation
      Scalar mean = Eigen::Map<const Eigen::VectorXd>(m_values.data(), m_values.size()).mean();
        
      Scalar variance = 0;
      for (auto const & v : m_values) {
        variance += (v - mean) * (v - mean);
      }
      stats.std_dev = std::sqrt(variance / m_values.size());
        
      auto indices = get_sorted_indices();
      stats.value_range = m_values[indices.back()] - m_values[indices[0]];
        
      Vector centroid = Vector::Zero(m_dim);
      for (auto const & v : m_simplex) centroid += v;
      centroid /= m_simplex.size();
        
      Scalar total_dist = 0;
      for (auto const & v : m_simplex) {
        total_dist += (v - centroid).norm();
      }
      stats.centroid_distance = total_dist / m_simplex.size();
        
      return stats;
    }

    /**
     * @brief Check convergence using robust criteria
     * @param best_value Best function value in simplex
     * @param worst_value Worst function value in simplex
     * @return True if converged
     */
    bool
    check_convergence_robust(Scalar best_value, Scalar worst_value) const {
      auto const & tolerance              = m_options.tolerance;
      auto const & simplex_tolerance      = m_options.simplex_tolerance;
      auto const & convergence_relaxation = m_options.convergence_relaxation;
    
      if (!m_options.use_robust_convergence) {
        // Usa worst_value qui
        auto stats              = compute_simplex_stats();
        bool value_converged    = (worst_value - best_value) < tolerance;
        bool geometry_converged = stats.diameter < simplex_tolerance;
        return value_converged && geometry_converged;
      }
        
      auto stats = compute_simplex_stats();
        
      // 1. Primary convergence: function values
      bool value_converged;
      if (m_options.use_relative_tolerance) {
        Scalar relative_range = stats.value_range / (1.0 + abs(best_value));
        value_converged = relative_range < tolerance;
      } else {
        value_converged = stats.value_range < tolerance;
      }
        
      // 2. Secondary convergence: simplex geometry
      bool geometry_converged = stats.diameter < simplex_tolerance * std::sqrt(m_dim);
        
      // 3. Variance-based convergence
      bool variance_converged = stats.std_dev < tolerance * convergence_relaxation;
        
      // Converge if primary condition OR all secondary conditions are satisfied
      bool converged = value_converged ||
                        (geometry_converged && stats.value_range < tolerance * convergence_relaxation) ||
                        (variance_converged && geometry_converged);
        
      if (m_options.verbose && (converged || m_global_iterations % m_options.progress_frequency == 0)) {
        fmt::print(
          " [Conv Check] V:{} G:{} S:{} | Range={:<12.4e} Diam={:<12.4e} StdDev={:<12.4e}\n",
          (value_converged ? "✓" : "✗"),
          (geometry_converged ? "✓" : "✗"),
          (variance_converged ? "✓" : "✗"),
          stats.value_range,
          stats.diameter,
          stats.std_dev
        );
      }
      return converged;
    }

    /**
     * @brief Check for optimization stagnation
     * @param current_best Current best function value
     * @return True if stagnated
     */
    bool
    check_stagnation(Scalar current_best) {
      if (!m_options.enable_restart) return false;
        
      Scalar improvement = abs(current_best - m_previous_best);
      Scalar relative_improvement = improvement / (1.0 + abs(m_previous_best));
        
      if (relative_improvement < m_options.stagnation_tolerance) {
        ++m_stagnation_count;
        return m_stagnation_count >= m_options.stagnation_threshold;
      } else {
        m_stagnation_count = 0;
        m_previous_best    = current_best;
        return false;
      }
    }

    /**
     * @brief Determine if restart is worthwhile based on current results
     * @param current_result Current optimization result
     * @return True if restart should be performed
     */
    bool
    is_restart_worthwhile(Result const & current_result) const {
      // Don't restart if we've reached satisfactory tolerance
      if (current_result.status == Status::CONVERGED) return false;
    
      // Calculate relative improvement from start
      Scalar absolute_improvement = abs(current_result.initial_function_value - current_result.final_function_value);
      Scalar relative_improvement = absolute_improvement / (1.0 + abs(current_result.initial_function_value));
    
      // 1. Restart for stagnation with insufficient improvement
      if (current_result.status == Status::STAGNATED) {
        if (relative_improvement < m_options.restart_relative_improvement_threshold) {
          return true;
        }
      }
    
      // 2. Restart for insufficient progress relative to resources used
      Scalar progress_per_eval = relative_improvement / (1.0 + current_result.function_evaluations);
      if (progress_per_eval < m_options.restart_progress_per_eval_threshold &&
          current_result.iterations > 100) {
        return true;
      }
    
      // 3. Restart for problematic simplex geometry
      Scalar scale               = 1.0 + m_best_point.norm();
      Scalar normalized_diameter = current_result.simplex_diameter / scale;
    
      if (normalized_diameter < m_options.simplex_tolerance * m_options.restart_simplex_diameter_factor1 &&
          relative_improvement < m_options.restart_simplex_geometry_threshold) {
        return true;
      }
    
      // 4. Restart for too many shrink operations without progress
      if (m_shrink_count > m_options.restart_shrink_count_threshold &&
          relative_improvement < m_options.restart_after_shrink_improvement) {
        return true;
      }
    
      // 5. Restart for high-dimensional problems with slow progress
      if (m_dim > 10) {
        Scalar expected_progress = 1.0 / std::sqrt(1.0 + current_result.iterations);
        if (relative_improvement < expected_progress * m_options.restart_expected_progress_ratio &&
          current_result.iterations > 200) {
          return true;
        }
      }

      // 6. Restart based on solution quality metric
      Scalar quality_metric = relative_improvement / (1.0 + std::log1p(current_result.function_evaluations));
      if (quality_metric < m_options.restart_quality_metric_threshold &&
        current_result.function_evaluations > 500) {
        return true;
      }
    
      // 7. Restart if simplex is degenerate but not converged
      auto stats = compute_simplex_stats();
      Scalar normalized_std_dev = stats.std_dev / (1.0 + abs(m_best_value));
      if (normalized_diameter < m_options.simplex_tolerance * m_options.restart_simplex_diameter_factor2 &&
        normalized_std_dev < m_options.tolerance * m_options.restart_std_dev_factor &&
        relative_improvement < m_options.restart_degenerate_improvement_threshold) {
        return true;
      }
        
      return false;
    }

    /**
     * @brief Adjust adaptive parameters based on algorithm behavior
     */
    void
    adaptive_parameter_adjustment() {
      // More conservative adaptive parameters
      if (m_shrink_count > 8) {
        m_current_sigma = max(0.3, m_current_sigma * 0.9);
        m_current_gamma = max(0.3, m_current_gamma * 0.95);
        m_shrink_count  = 0;
      }
    }

    /**
     * @brief Perform reflection operation
     * @param worst_index Index of worst point
     * @return Function value at reflected point
     */
    Scalar
    reflect_point(size_t worst_index) {
      m_trial_point = m_centroid + m_current_rho * (m_centroid - m_simplex[worst_index]);
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }

    /**
     * @brief Perform expansion operation
     * @return Function value at expanded point
     */
    Scalar
    expand_point() {
      m_trial_point = m_centroid + m_current_chi * (m_trial_point - m_centroid);
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }

    /**
     * @brief Perform contraction operation
     * @param worst_index Index of worst point
     * @param outside Whether to contract outside or inside
     * @return Function value at contracted point
     */
    Scalar
    contract_point(size_t worst_index, bool outside) {
      if (outside) {
        m_trial_point = m_centroid + m_current_gamma * (m_trial_point - m_centroid);
      } else {
        m_trial_point = m_centroid - m_current_gamma * (m_centroid - m_simplex[worst_index]);
      }
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }

    /**
     * @brief Perform shrink operation towards best point
     * @param best_index Index of best point
     */
    void
    shrink_simplex(size_t best_index) {
      Vector best = m_simplex[best_index];
      for (size_t i{0}; i <= m_dim; ++i) {
        if (i != best_index) {
          m_simplex[i] = best + m_current_sigma * (m_simplex[i] - best);
          project_point(m_simplex[i]);
          m_values[i] = safe_evaluate(m_simplex[i]);
        }
      }
      ++m_shrink_count;
      //mark_simplex_unordered();
    }

    /**
     * @brief Perform one iteration of Nelder-Mead algorithm
     * @return True if converged
     */
    bool
    nelder_mead_iteration() {
      auto indices = get_sorted_indices();
      size_t best_idx         = indices[0];
      size_t second_worst_idx = indices[m_dim - 1];
      size_t worst_idx        = indices[m_dim];

      Scalar best_value  = m_values[best_idx];
      Scalar worst_value = m_values[worst_idx];

      if (m_options.track_best_point && best_value < m_best_value) {
        m_best_value = best_value;
        m_best_point = m_simplex[best_idx];
      }

      if (check_convergence_robust(best_value, worst_value)) {
        return true;
      }

      update_centroid(worst_idx);
      Scalar f_reflect       = reflect_point(worst_idx);
      Vector reflected_point = m_trial_point;

      if (f_reflect < best_value) {
        Scalar f_expand = expand_point();
        if (f_expand < f_reflect) {
          m_simplex[worst_idx] = m_trial_point;
          m_values[worst_idx]  = f_expand;
        } else {
          m_simplex[worst_idx] = reflected_point;
          m_values[worst_idx]  = f_reflect;
        }
      } else if (f_reflect < m_values[second_worst_idx]) {
        m_simplex[worst_idx] = reflected_point;
        m_values[worst_idx]  = f_reflect;
      } else {
        if (f_reflect < worst_value) {
          Scalar f_contract = contract_point(worst_idx, true);
          if (f_contract <= f_reflect) {
            m_simplex[worst_idx] = m_trial_point;
            m_values[worst_idx]  = f_contract;
          } else {
            shrink_simplex(best_idx);
            adaptive_parameter_adjustment();
          }
        } else {
          Scalar f_contract = contract_point(worst_idx, false);
          if (f_contract < worst_value) {
            m_simplex[worst_idx] = m_trial_point;
            m_values[worst_idx]  = f_contract;
          } else {
            shrink_simplex(best_idx);
            adaptive_parameter_adjustment();
          }
        }
      }
      mark_simplex_unordered();
      return false;
    }

    /**
     * @brief Run single Nelder-Mead optimization (without restarts)
     * @param x0 Starting point
     * @return Optimization result
     */
    Result
    minimize_single_run(Vector const & x0) {
      Result result;
      size_t initial_evals = m_global_evals;
      initialize_adaptive_parameters();
    
      if (m_global_iterations >= m_options.max_iterations) {
        result.status               = Status::MAX_ITERATIONS;
        result.solution             = x0;
        result.final_function_value = safe_evaluate(x0);
        result.function_evaluations = m_global_evals - initial_evals;
        return result;
      }
    
      initialize_simplex(x0);
      auto indices = get_sorted_indices();
      result.initial_function_value = m_values[indices[0]];
    
      if (m_options.verbose) {
        fmt::print("  [NM-Run]  Start | Dim={:<10} | F_0={:<12.6e}  |", m_dim, result.initial_function_value);
      }
    
      size_t local_iter = 0;
    
      while (true) {
        if (m_global_iterations >= m_options.max_iterations) {
          result.status = Status::MAX_ITERATIONS;
          break;
        }
    
        if (m_global_evals >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }
    
        ++m_global_iterations;
        ++local_iter;
    
        if (m_options.enable_restart && local_iter % 50 == 0) {
          indices = get_sorted_indices();
          if (check_stagnation(m_values[indices[0]])) {
            result.status = Status::STAGNATED;
            break;
          }
        }
    
        if (nelder_mead_iteration()) {
          result.status = Status::CONVERGED;
          break;
        }
    
        result.simplex_diameter = compute_diameter();
        if (result.simplex_diameter < m_options.simplex_tolerance) {
          result.status = Status::SIMPLEX_TOO_SMALL;
          break;
        }
    
        if (m_options.verbose && (m_global_iterations % m_options.progress_frequency) == 0) {
          indices = get_sorted_indices();
          fmt::print(
            "  [NM-Iter] {:>5} | F={:<12.6e} | Diam={:<12.6e} |",
            m_global_iterations, m_values[indices[0]], result.simplex_diameter
          );
        }
      }
    
      if (result.status == Status::FAILED) {
        result.status = Status::MAX_ITERATIONS;
      }
    
      indices                     = get_sorted_indices();
      result.solution             = m_simplex[indices[0]];
      result.final_function_value = m_values[indices[0]];
      result.simplex_volume       = compute_volume();
      result.simplex_diameter     = compute_diameter();
      result.iterations           = local_iter;
      result.function_evaluations = m_global_evals - initial_evals;
      result.shrink_operations    = m_shrink_count;
    
      return result;
    }

    /**
     * @brief Print optimization header information
     * @param x0 Starting point
     */
    void
    print_header(Vector const & x0) const {
      if (!m_options.verbose) return;
      fmt::print(
        "╔════════════════════════════════════════════════════════════════╗\n"
        "║                    Nelder-Mead Optimization                    ║\n"
        "╠════════════════════════════════════════════════════════════════╣\n"
        "║  Dimension          : {:<39}  ║\n"
        "║  Max Iterations     : {:<39}  ║\n"
        "║  Max Evals          : {:<39}  ║\n"
        "║  Tolerance          : {:<39.6e}  ║\n"
        "║  Restarts           : {:<39}  ║\n"
        "║  Bounds             : {:<39}  ║\n"
        "║  Robust Convergence : {:<39}  ║\n"
        "╚════════════════════════════════════════════════════════════════╝\n",
        x0.size(),
        m_options.max_iterations,
        m_options.max_function_evaluations,
        m_options.tolerance,
        (m_options.enable_restart ? fmt::format("Enabled (Max: {})", m_options.max_restarts) : "Disabled"),
        (m_use_bounds ? "Active" : "None"),
        (m_options.use_robust_convergence ? "Yes" : "No")
      );
    }

    /**
     * @brief Print optimization statistics
     * @param res Optimization result to print
     */
    void
    print_statistics(Result const & res) const {
      if (!m_options.verbose) return;
      fmt::print(
        "╔════════════════════════════════════════════════════════════════╗\n"
        "║                    Optimization Finished                       ║\n"
        "╠════════════════════════════════════════════════════════════════╣\n"
        "║  Final Status       : {:<39}  ║\n"
        "║  Final Value        : {:<39.6e}  ║\n"
        "║  Total Iterations   : {:<39}  ║\n"
        "║  Total Evals        : {:<39}  ║\n"
        "║  Restarts           : {:<39}  ║\n"
        "║  Shrink Operations  : {:<39}  ║\n"
        "║  Simplex Diameter   : {:<39.6e}  ║\n"
        "╚════════════════════════════════════════════════════════════════╝\n\n",
        status_to_string(res.status),
        res.final_function_value,
        res.iterations,
        res.function_evaluations,
        res.restarts_performed,
        res.shrink_operations,
        res.simplex_diameter
      );
    }

  public:
    /**
     * @brief Construct Nelder-Mead optimizer with given options
     * @param opts Optimization options
     */
    explicit
    NelderMead_classic(Options const & opts = Options()) : m_options(opts) {}
    
    /**
     * @brief Set optimization bounds
     * @param lower Lower bounds
     * @param upper Upper bounds
     */
    void
    set_bounds(Vector const & lower, Vector const & upper) {
      UTILS_ASSERT(lower.size() == upper.size(), "Bounds size mismatch");
      UTILS_ASSERT((lower.array() <= upper.array()).all(), "Lower <= Upper");
      m_lower      = lower;
      m_upper      = upper;
      m_use_bounds = true;
    }

    /**
     * @brief Clear optimization bounds
     */
    void
    clear_bounds() {
      m_use_bounds = false;
    }

    /**
     * @brief Minimize objective function starting from x0
     * @param x0 Starting point
     * @param callback Objective function
     * @return Optimization result
     */
    Result
    minimize(Vector const & x0, Callback const & callback) {
      m_callback = &callback;

      m_stagnation_count  = 0;
      m_shrink_count      = 0;
      // Initialize with NaN to handle negative values properly
      m_best_value        = std::numeric_limits<Scalar>::max();
      m_previous_best     = std::numeric_limits<Scalar>::max();
      m_global_iterations = 0;
      m_global_evals      = 0;
      m_simplex_ordered   = false;
    
      size_t restarts = 0;

      print_header(x0);

      Result best_result = minimize_single_run(x0);
    
      // Initialize m_best_value if not yet set
      if (std::isnan(m_best_value)) {
        m_best_value = best_result.final_function_value;
        m_best_point = best_result.solution;
      }

      while ( m_options.enable_restart &&
              restarts < m_options.max_restarts &&
              is_restart_worthwhile(best_result)) {
        if (m_global_evals >= m_options.max_function_evaluations) break;
        if (m_global_iterations >= m_options.max_iterations) break;

        if (m_options.verbose) {
          fmt::print(
            "  [NM-Restart] #{}/{} | Reason: {:<16} | F={:12.6e}\n",
            (restarts+1), m_options.max_restarts,
            status_to_string(best_result.status),
            best_result.final_function_value
          );
        }

        Scalar perturbation_scale = m_options.restart_perturbation_ratio * (1.0 + restarts * 0.1);
            
        Vector restart_x0;
        Vector scale_vec = Vector::Ones(x0.size());
            
        if (m_use_bounds) {
          for (size_t i{0}; i < x0.size(); ++i) {
            Scalar r = m_upper(i) - m_lower(i);
            scale_vec(i) = std::isfinite(r) ? r : max(Scalar(1.0), abs(best_result.solution(i)));
          }
        } else {
          for (size_t i{0}; i < x0.size(); ++i) {
            scale_vec(i) = max(Scalar(1.0), abs(best_result.solution(i)));
          }
        }

        Vector perturbation = Vector::Random(x0.size()).cwiseProduct(scale_vec) * perturbation_scale;
            
        if (m_options.track_best_point) {
          restart_x0 = m_best_point + perturbation;
        } else {
          restart_x0 = best_result.solution + perturbation;
        }

        project_point(restart_x0);

        m_stagnation_count = 0;
        m_shrink_count = 0;
        m_simplex_ordered = false;

        Result current_result = minimize_single_run(restart_x0);
        ++restarts;

        // ROBUST IMPROVEMENT CONDITION FOR POSITIVE AND NEGATIVE VALUES
        bool improvement = false;
            
        if (std::isnan(current_result.final_function_value)) {
          improvement = false;
        } else if (std::isnan(best_result.final_function_value)) {
          improvement = true;
        } else {
          // Calculate normalized relative improvement
          Scalar abs_best = abs(best_result.final_function_value);
          Scalar abs_current = abs(current_result.final_function_value);
            
          // Case 1: Both positive or zero - standard improvement
          if (best_result.final_function_value >= 0 &&
            current_result.final_function_value >= 0) {
            improvement = current_result.final_function_value < best_result.final_function_value * m_options.restart_improvement_ratio;
          }
          // Case 2: Both negative - improvement means more negative
          else if (best_result.final_function_value < 0 &&
                   current_result.final_function_value < 0) {
            improvement = current_result.final_function_value < best_result.final_function_value;
          }
          // Case 3: Transition from positive to negative - always improvement
          else if (best_result.final_function_value >= 0 &&
                   current_result.final_function_value < 0) {
            improvement = true;
          }
          // Case 4: Transition from negative to positive - usually worsening
          else {
            improvement = false;
          }
                
          // Additional check: significant absolute improvement
          Scalar absolute_improvement = best_result.final_function_value - current_result.final_function_value;
          Scalar relative_improvement = absolute_improvement / (1.0 + abs_best);
                
          if (!improvement && relative_improvement > m_options.restart_absolute_improvement_threshold) {
            improvement = true;
          }
        }

        if (improvement) {
          best_result = current_result;
          best_result.restarts_performed = restarts;
        } else if (m_options.verbose) {
          fmt::print("    [Restart rejected: no improvement]\n");
        }
      }

      best_result.iterations           = m_global_iterations;
      best_result.function_evaluations = m_global_evals;
      best_result.restarts_performed   = restarts;
      best_result.shrink_operations    = m_shrink_count;

      // Final best point update - correct comparison for minimization
      if (m_options.track_best_point) {
        if ( std::isnan(best_result.final_function_value) ||
             (!std::isnan(m_best_value) && m_best_value < best_result.final_function_value)) {
          best_result.final_function_value = m_best_value;
          best_result.solution = m_best_point;
        }
      }

      print_statistics(best_result);

      return best_result;
    }

    /**
     * @brief Convert status enum to string
     * @param status Optimization status
     * @return String representation of status
     */
    static
    string status_to_string(Status status) {
      switch (status) {
      case Status::CONVERGED:           return "CONVERGED";
      case Status::MAX_ITERATIONS:      return "MAX_ITERATIONS";
      case Status::MAX_FUN_EVALUATIONS: return "MAX_FUN_EVALUATIONS";
      case Status::SIMPLEX_TOO_SMALL:   return "SIMPLEX_TOO_SMALL";
      case Status::STAGNATED:           return "STAGNATED";
      case Status::FAILED:              return "FAILED";
      default:                          return "UNKNOWN";
      }
    }

    // Access methods for debugging and monitoring
    size_t get_total_evaluations() const { return m_global_evals; }  ///< Get total function evaluations
    size_t get_total_iterations()  const { return m_global_iterations; }  ///< Get total iterations
    Scalar get_best_value()        const { return m_best_value; }  ///< Get best function value found
    Vector get_best_point()        const { return m_best_point; }  ///< Get best point found
  };

  /**
   * @class NelderMead_Subspace
   * @brief Nelder-Mead optimization on a subspace of coordinates
   *
   * This variant works on a subset of dimensions while keeping others fixed.
   */
  template <typename Scalar = double>
  class NelderMead_Subspace {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix   = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Callback = std::function<Scalar(Vector const &)>;
  
    enum class Status {
      CONVERGED,
      MAX_ITERATIONS,
      MAX_FUN_EVALUATIONS,
      SIMPLEX_TOO_SMALL,
      STAGNATED,
      FAILED
    };
  
    struct Options {
      size_t max_iterations{5000};
      size_t max_function_evaluations{20000};
      Scalar tolerance{1e-6};
      Scalar stagnation_tolerance{1e-8};
      Scalar simplex_tolerance{1e-8};
    
      Scalar rho{1.0};
      Scalar chi{2.0};
      Scalar gamma{0.5};
      Scalar sigma{0.5};
      Scalar initial_step{0.1};
    
      bool   adaptive_parameters{true};
      bool   verbose{false};
      size_t progress_frequency{100};
    
      Scalar min_step_size{1e-10};
      bool   use_robust_convergence{true};
      bool   use_relative_tolerance{true};
    };
  
    struct Result {
      Vector solution;
      Status status{Status::FAILED};
      size_t iterations{0};
      size_t function_evaluations{0};
      Scalar final_function_value{0};
      Scalar initial_function_value{0};
      Scalar simplex_diameter{0};
    };

  private:
    Options        m_options;
    Vector         m_full_point;  // Full dimensional point
    vector<size_t> m_active_dims; // Active dimensions for optimization
    Vector         m_lower;
    Vector         m_upper;
    bool           m_use_bounds{false};
  
    Callback const * m_callback{nullptr};
    size_t         m_evals{0};
    size_t         m_iterations{0};
  
    vector<Vector> m_simplex;
    vector<Scalar> m_values;
    Vector         m_centroid;
    Vector         m_trial_point;
    size_t         m_subspace_dim{0};
  
    mutable bool           m_simplex_ordered{false};
    mutable vector<size_t> m_sorted_indices;
  
    Scalar m_current_rho;
    Scalar m_current_chi;
    Scalar m_current_gamma;
    Scalar m_current_sigma;
  
    // Map from subspace coordinates to full space
    Vector
    subspace_to_full( Vector const & x_sub ) const {
      Vector x_full = m_full_point;
      for (size_t i = 0; i < m_active_dims.size(); ++i) {
        x_full(m_active_dims[i]) = x_sub(i);
      }
      return x_full;
    }
  
    // Extract subspace coordinates from full space
    Vector
    full_to_subspace(Vector const & x_full) const {
      Vector x_sub(m_active_dims.size());
      for (size_t i = 0; i < m_active_dims.size(); ++i) {
        x_sub(i) = x_full(m_active_dims[i]);
      }
      return x_sub;
    }
  
    Scalar
    safe_evaluate(Vector const & x_sub) {
      Vector x_full = subspace_to_full(x_sub);
    
      if (m_use_bounds) {
        for (size_t i = 0; i < m_active_dims.size(); ++i) {
          size_t dim = m_active_dims[i];
          if (x_full(dim) < m_lower(dim) || x_full(dim) > m_upper(dim)) {
            return std::numeric_limits<Scalar>::max();
          }
        }
      }
    
      Scalar value = (*m_callback)(x_full);
      ++m_evals;

      if (!std::isfinite(value)) {
        return std::numeric_limits<Scalar>::max();
      }
      return value;
    }
  
    void
    project_point(Vector & x_sub) const {
      if (!m_use_bounds) return;
    
      for (size_t i{0}; i < m_active_dims.size(); ++i) {
        size_t dim = m_active_dims[i];
        x_sub(i) = std::clamp(x_sub(i), m_lower(dim), m_upper(dim));
      }
    }
  
    void
    initialize_adaptive_parameters() {
      if (m_options.adaptive_parameters && m_subspace_dim > 0) {
        Scalar n = static_cast<Scalar>(m_subspace_dim);
        if (m_subspace_dim > 10) {
          m_current_rho   = 1.0;
          m_current_chi   = 1.0 + 1.0 / n;
          m_current_gamma = 0.75 - 0.5 / n;
          m_current_sigma = 0.8 - 0.5 / n;
        } else {
          m_current_rho   = 1.0;
          m_current_chi   = 1.0 + 2.0 / n;
          m_current_gamma = 0.75 - 0.5 / n;
          m_current_sigma = 1.0 - 1.0 / n;
        }
      } else {
        m_current_rho   = m_options.rho;
        m_current_chi   = m_options.chi;
        m_current_gamma = m_options.gamma;
        m_current_sigma = m_options.sigma;
      }
    }
  
    vector<size_t> const &
    get_sorted_indices() const {
      if (!m_simplex_ordered) {
        m_sorted_indices.resize(m_subspace_dim + 1);
        std::iota(m_sorted_indices.begin(), m_sorted_indices.end(), 0);
        std::sort(m_sorted_indices.begin(), m_sorted_indices.end(),
                  [this](size_t i, size_t j) { return m_values[i] < m_values[j]; });
        m_simplex_ordered = true;
      }
      return m_sorted_indices;
    }
  
    void
    mark_simplex_unordered() {
      m_simplex_ordered = false;
    }
  
    Scalar
    get_smart_step(size_t idx, Scalar current_val) const {
      Scalar step = m_options.initial_step;
      size_t dim  = m_active_dims[idx];
    
      if (m_use_bounds) {
        Scalar lower = m_lower(dim);
        Scalar upper = m_upper(dim);
      
        if (std::isfinite(lower) && std::isfinite(upper)) {
          Scalar range = upper - lower;
          if (range > std::numeric_limits<Scalar>::epsilon()) {
            step = std::clamp(range * 0.05, m_options.min_step_size, Scalar(10.0));
          }
        }
      }
    
      Scalar abs_val = std::abs(current_val);
      if (abs_val > 1.0) {
        step = std::max(m_options.initial_step, abs_val * 0.05);
      }
   
      return std::max(step, m_options.min_step_size);
    }
  
    void
    initialize_simplex(Vector const & x0_sub) {
      m_subspace_dim = x0_sub.size();
      
      m_simplex.resize(m_subspace_dim + 1);
      m_values.resize(m_subspace_dim + 1);
      m_centroid.resize(m_subspace_dim);
      m_trial_point.resize(m_subspace_dim);
      m_sorted_indices.resize(m_subspace_dim + 1);
      
      for (auto & vec : m_simplex) {
        vec.resize(m_subspace_dim);
      }
      
      m_simplex[0] = x0_sub;
      project_point(m_simplex[0]);
      m_values[0] = safe_evaluate(m_simplex[0]);
      
      Vector x_base = m_simplex[0];
      Scalar min_step = m_options.min_step_size;
      
      for (size_t i = 0; i < m_subspace_dim; ++i) {
        Scalar step = get_smart_step(i, x_base(i));
        if (std::abs(step) < min_step) {
          step = (step >= 0) ? min_step : -min_step;
        }
      
        Vector x_next = x_base;
        x_next(i) += step;
        project_point(x_next);
     
        if ((x_next - x_base).norm() < min_step) {
          x_next = x_base;
          x_next(i) -= step;
          project_point(x_next);
        }
     
        m_simplex[i + 1] = x_next;
        m_values[i + 1] = safe_evaluate(x_next);
      }
     
      mark_simplex_unordered();
    }
  
    void
    update_centroid(size_t worst_index) {
      m_centroid.setZero();
      for (size_t i{0}; i <= m_subspace_dim; ++i) {
        if (i != worst_index) {
          m_centroid += m_simplex[i];
        }
      }
      m_centroid /= static_cast<Scalar>(m_subspace_dim);
    }
  
    Scalar
    compute_diameter() const {
      Scalar max_dist = 0;
      for (size_t i{0}; i <= m_subspace_dim; ++i) {
        for (size_t j{i + 1}; j <= m_subspace_dim; ++j) {
          Scalar dist = (m_simplex[i] - m_simplex[j]).norm();
          max_dist = std::max(max_dist, dist);
        }
      }
      return max_dist;
    }
  
    bool
    check_convergence(Scalar best_value, Scalar worst_value) const {
      Scalar value_range = worst_value - best_value;
      Scalar diameter    = compute_diameter();

      if (m_options.use_relative_tolerance) value_range /= 1.0 + std::abs(best_value);
    
      bool value_converged    = value_range < m_options.tolerance;
      bool geometry_converged = diameter    < m_options.simplex_tolerance * std::sqrt(m_subspace_dim);
    
      return value_converged || geometry_converged;
    }
  
    Scalar
    reflect_point(size_t worst_index) {
      m_trial_point = m_centroid + m_current_rho * (m_centroid - m_simplex[worst_index]);
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }
  
    Scalar
    expand_point() {
      m_trial_point = m_centroid + m_current_chi * (m_trial_point - m_centroid);
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }
  
    Scalar
    contract_point(size_t worst_index, bool outside) {
      if (outside) {
        m_trial_point = m_centroid + m_current_gamma * (m_trial_point - m_centroid);
      } else {
        m_trial_point = m_centroid - m_current_gamma * (m_centroid - m_simplex[worst_index]);
      }
      project_point(m_trial_point);
      return safe_evaluate(m_trial_point);
    }
  
    void
    shrink_simplex(size_t best_index) {
      Vector best = m_simplex[best_index];
      for (size_t i{0}; i <= m_subspace_dim; ++i) {
        if (i != best_index) {
          m_simplex[i] = best + m_current_sigma * (m_simplex[i] - best);
          project_point(m_simplex[i]);
          m_values[i] = safe_evaluate(m_simplex[i]);
        }
      }
      mark_simplex_unordered();
    }
  
    bool
    nelder_mead_iteration() {
      auto   indices          = get_sorted_indices();
      size_t best_idx         = indices[0];
      size_t second_worst_idx = indices[m_subspace_dim - 1];
      size_t worst_idx        = indices[m_subspace_dim];
      
      Scalar best_value  = m_values[best_idx];
      Scalar worst_value = m_values[worst_idx];
      
      if (check_convergence(best_value, worst_value)) {
        return true;
      }
      
      update_centroid(worst_idx);
      Scalar f_reflect = reflect_point(worst_idx);
      Vector reflected_point = m_trial_point;
      
      if (f_reflect < best_value) {
        Scalar f_expand = expand_point();
        if (f_expand < f_reflect) {
          m_simplex[worst_idx] = m_trial_point;
          m_values[worst_idx]  = f_expand;
        } else {
          m_simplex[worst_idx] = reflected_point;
          m_values[worst_idx]  = f_reflect;
        }
      } else if (f_reflect < m_values[second_worst_idx]) {
        m_simplex[worst_idx] = reflected_point;
        m_values[worst_idx]  = f_reflect;
      } else {
        if (f_reflect < worst_value) {
          Scalar f_contract = contract_point(worst_idx, true);
          if (f_contract <= f_reflect) {
            m_simplex[worst_idx] = m_trial_point;
            m_values[worst_idx]  = f_contract;
          } else {
            shrink_simplex(best_idx);
          }
        } else {
          Scalar f_contract = contract_point(worst_idx, false);
          if (f_contract < worst_value) {
            m_simplex[worst_idx] = m_trial_point;
            m_values[worst_idx]  = f_contract;
          } else {
            shrink_simplex(best_idx);
          }
        }
      }
    
      mark_simplex_unordered();
      return false;
    }

  public:

    explicit
    NelderMead_Subspace(Options const & opts = Options()) : m_options(opts) {}
  
    void
    set_bounds(Vector const & lower, Vector const & upper) {
      m_lower      = lower;
      m_upper      = upper;
      m_use_bounds = true;
    }
  
    /**
     * @brief Optimize over a subspace of dimensions
     * @param full_point Current full-dimensional point (fixed dimensions remain unchanged)
     * @param active_dims Indices of dimensions to optimize
     * @param callback Objective function (takes full-dimensional vector)
     * @return Optimization result (solution in subspace coordinates)
     */
    Result
    minimize( Vector         const & full_point,
              vector<size_t> const & active_dims,
              Callback       const & callback ) {
      m_callback        = &callback;
      m_full_point      = full_point;
      m_active_dims     = active_dims;
      m_evals           = 0;
      m_iterations      = 0;
      m_simplex_ordered = false;
    
      // Extract subspace starting point
      Vector x0_sub = full_to_subspace(full_point);
    
      initialize_adaptive_parameters();
      initialize_simplex(x0_sub);
    
      Result result;
      auto indices = get_sorted_indices();
      result.initial_function_value = m_values[indices[0]];
    
      while (true) {
        if (m_iterations >= m_options.max_iterations) {
          result.status = Status::MAX_ITERATIONS;
          break;
        }
    
        if (m_evals >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }
    
        ++m_iterations;
    
        if (nelder_mead_iteration()) {
          result.status = Status::CONVERGED;
          break;
        }
    
        result.simplex_diameter = compute_diameter();
        if (result.simplex_diameter < m_options.simplex_tolerance) {
          result.status = Status::SIMPLEX_TOO_SMALL;
          break;
        }
      }
    
      indices                     = get_sorted_indices();
      result.solution             = m_simplex[indices[0]];
      result.final_function_value = m_values[indices[0]];
      result.iterations           = m_iterations;
      result.function_evaluations = m_evals;
      result.simplex_diameter     = compute_diameter();
   
      // Update the full point with optimized subspace coordinates
      m_full_point = subspace_to_full(result.solution);
   
      return result;
    }
  
    Vector get_full_solution() const { return m_full_point; }
    size_t get_evaluations()   const { return m_evals; }
  
    static string status_to_string(Status status) {
      switch (status) {
      case Status::CONVERGED:           return "CONVERGED";
      case Status::MAX_ITERATIONS:      return "MAX_ITERATIONS";
      case Status::MAX_FUN_EVALUATIONS: return "MAX_FUN_EVALUATIONS";
      case Status::SIMPLEX_TOO_SMALL:   return "SIMPLEX_TOO_SMALL";
      case Status::STAGNATED:           return "STAGNATED";
      case Status::FAILED:              return "FAILED";
      default:                          return "UNKNOWN";
      }
    }
  };

  /**
   * @class NelderMead_BlockCoordinate
   * @brief Block coordinate descent using Nelder-Mead on subspaces
   *
   * Solves high-dimensional problems by alternating optimization over blocks
   * of coordinates. Supports multiple block selection strategies.
   */
  template <typename Scalar = double>
  class NelderMead_BlockCoordinate {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Callback = std::function<Scalar(Vector const &)>;
  
    enum class BlockStrategy {
      SEQUENTIAL,      // Cycle through fixed blocks sequentially
      RANDOM,          // Random blocks each iteration
      GRADIENT_BASED,  // Select dimensions with largest estimated gradients
      ADAPTIVE         // Adaptive selection based on improvement history
    };
  
    enum class Status {
      CONVERGED,
      MAX_OUTER_ITERATIONS,
      MAX_FUNCTION_EVALUATIONS,
      STAGNATED,
      FAILED
    };
  
    struct Options {
      size_t block_size{5};                    // Size of each coordinate block
      size_t max_outer_iterations{100};        // Maximum outer (block) iterations
      size_t max_function_evaluations{100000}; // Total function evaluation budget
      Scalar tolerance{1e-6};                  // Convergence tolerance
      Scalar stagnation_tolerance{1e-8};       // Stagnation detection
      size_t stagnation_threshold{5};          // Cycles before declaring stagnation
    
      BlockStrategy strategy{BlockStrategy::SEQUENTIAL};
    
      bool verbose{false};
      size_t progress_frequency{10};
    
      // Options for subspace optimizer
      typename NelderMead_Subspace<Scalar>::Options subspace_options;
    
      // Gradient estimation parameters (for GRADIENT_BASED strategy)
      Scalar gradient_epsilon{1e-6};
      size_t gradient_sample_size{10};  // Number of dimensions to sample for gradient
    
      // Adaptive strategy parameters
      Scalar improvement_threshold{0.01};  // Minimum improvement to consider block useful
      size_t history_window{10};           // Window for tracking block performance
    };
  
    struct Result {
      Vector solution;
      Status status{Status::FAILED};
      size_t outer_iterations{0};
      size_t total_subspace_iterations{0};
      size_t function_evaluations{0};
      Scalar final_function_value{0};
      Scalar initial_function_value{0};
      vector<size_t> block_usage_count;  // How many times each block was optimized
    };
  
  private:
    Options m_options;
    Vector  m_lower;
    Vector  m_upper;
    bool    m_use_bounds{false};
    size_t  m_global_evals{0};
  
    Callback const * m_callback{nullptr};
    size_t m_dim{0};
    size_t m_num_blocks{0};
  
    NelderMead_Subspace<Scalar> m_subspace_optimizer;
  
    // For adaptive strategy
    vector<Scalar> m_block_improvement_history;
    vector<size_t> m_block_usage;
  
    std::mt19937 m_rng;
  
    /**
     * @brief Safely evaluate objective function with bounds checking
     * @param x Point to evaluate
     * @return Function value or large value if out of bounds/invalid
     */
    Scalar
    safe_evaluate( Vector const & x ) {
      UTILS_ASSERT(m_callback != nullptr, "NelderMead_classic::safe_evaluate(x) Callback not set!");
        
      // Check bounds before evaluation
      if ( m_use_bounds ) {
        bool out_of_bound = (x.array() < m_lower.array()).any() ||
                            (x.array() > m_upper.array()).any();
        if (out_of_bound) {
          if (m_options.verbose) {
            fmt::print("Warning: Point outside bounds, returning large value\n");
          }
          return std::numeric_limits<Scalar>::max();
        }
      }
        
      Scalar value{(*m_callback)(x)};
      ++m_global_evals;
        
      if (!std::isfinite(value)) {
        if (m_options.verbose) {
          fmt::print("Warning: Non-finite function value at x={}\n", x.transpose());
        }
        return std::numeric_limits<Scalar>::max();
      }
      return value;
    }

    /**
     * @brief Generate sequential blocks of coordinates
     */
    vector<vector<size_t>>
    generate_sequential_blocks() const {
      vector<vector<size_t>> blocks;
    
      for (size_t start{0}; start < m_dim; start += m_options.block_size) {
        vector<size_t> block;
        for (size_t i{start}; i < std::min(start + m_options.block_size, m_dim); ++i) {
          block.push_back(i);
        }
        blocks.push_back(block);
      }
    
      return blocks;
    }
  
    /**
     * @brief Generate random block of coordinates
     */
    vector<size_t>
    generate_random_block() {
      vector<size_t> all_dims(m_dim);
      std::iota(all_dims.begin(), all_dims.end(), 0);
      std::shuffle(all_dims.begin(), all_dims.end(), m_rng);
    
      size_t block_size = std::min(m_options.block_size, m_dim);
      return vector<size_t>(all_dims.begin(), all_dims.begin() + block_size);
    }
  
    /**
     * @brief Estimate gradient and select dimensions with largest components
     */
    vector<size_t>
    generate_gradient_based_block(Vector const & x, Scalar f_x) {
      size_t sample_size = std::min(m_options.gradient_sample_size, m_dim);
    
      // Sample random dimensions to estimate gradient
      vector<size_t> sampled_dims(m_dim);
      std::iota(sampled_dims.begin(), sampled_dims.end(), 0);
      std::shuffle(sampled_dims.begin(), sampled_dims.end(), m_rng);
      sampled_dims.resize(sample_size);
    
      vector<std::pair<Scalar, size_t>> gradient_magnitudes;
    
      for (size_t dim : sampled_dims) {
        Vector x_plus = x;
        x_plus(dim) += m_options.gradient_epsilon;
    
        if (m_use_bounds) {
          x_plus(dim) = std::clamp(x_plus(dim), m_lower(dim), m_upper(dim));
        }
    
        Scalar f_plus = safe_evaluate(x_plus);
        Scalar grad_estimate = abs(f_plus - f_x) / m_options.gradient_epsilon;
    
        gradient_magnitudes.emplace_back(grad_estimate, dim);
      }
    
      // Sort by gradient magnitude
      std::sort(gradient_magnitudes.begin(), gradient_magnitudes.end(),
                [](auto const & a, auto const & b) { return a.first > b.first; });
    
      // Select top dimensions
      vector<size_t> block;
      size_t block_size = std::min(m_options.block_size, sample_size);
      for (size_t i{0}; i < block_size; ++i) {
        block.push_back(gradient_magnitudes[i].second);
      }
    
      return block;
    }
  
    /**
     * @brief Adaptively select block based on past performance
     */
    vector<size_t>
    generate_adaptive_block() {
      // Calculate scores for each block based on recent improvement history
      vector<Scalar> block_scores(m_num_blocks, 0.0);
  
      for (size_t i{0}; i < m_num_blocks; ++i) {
        // Score combines recent improvement and usage (prefer less-used blocks)
        Scalar improvement_score = m_block_improvement_history[i];
        Scalar usage_penalty = static_cast<Scalar>(m_block_usage[i]) /
                               (1.0 + *std::max_element(m_block_usage.begin(), m_block_usage.end()));
        block_scores[i] = improvement_score * (1.0 - 0.5 * usage_penalty);
      }
  
      // Select block with highest score
      size_t best_block = std::distance(block_scores.begin(),
                                        std::max_element(block_scores.begin(), block_scores.end()));
  
      auto blocks = generate_sequential_blocks();
      return blocks[best_block];
    }
  
    /**
     * @brief Update block improvement history
     */
    void
    update_block_history(size_t block_idx, Scalar improvement) {
      // Exponential moving average
      Scalar alpha = 0.3;
      m_block_improvement_history[block_idx] =
        alpha * improvement + (1.0 - alpha) * m_block_improvement_history[block_idx];
    }

  public:
    explicit
    NelderMead_BlockCoordinate(Options const & opts = Options())
    : m_options(opts)
    , m_subspace_optimizer(opts.subspace_options)
    , m_rng(std::random_device{}())
    {}
  
    void
    set_bounds(Vector const & lower, Vector const & upper) {
      m_lower      = lower;
      m_upper      = upper;
      m_use_bounds = true;
      m_subspace_optimizer.set_bounds(lower, upper);
    }
  
    /**
     * @brief Minimize high-dimensional function using block coordinate descent
     */
    Result
    minimize(Vector const & x0, Callback const & callback) {
      m_callback     = &callback;
      m_dim          = x0.size();
      m_global_evals = 0;

      // Initialize blocks
      auto sequential_blocks = generate_sequential_blocks();
      m_num_blocks = sequential_blocks.size();
    
      m_block_improvement_history.assign(m_num_blocks, 0.0);
      m_block_usage.assign(m_num_blocks, 0);
    
      Result result;
      result.solution = x0;
      result.initial_function_value = safe_evaluate(x0);
      result.block_usage_count.resize(m_num_blocks, 0);
    
      Scalar current_value    = result.initial_function_value;
      Scalar previous_value   = current_value;
      size_t stagnation_count = 0;
    
      if (m_options.verbose) {
        fmt::print(
          "╔════════════════════════════════════════════════════════════════╗\n"
          "║           Block Coordinate Nelder-Mead Optimization            ║\n"
          "╠════════════════════════════════════════════════════════════════╣\n"
          "║  Dimension          : {:<39}  ║\n"
          "║  Block Size         : {:<39}  ║\n"
          "║  Number of Blocks   : {:<39}  ║\n"
          "║  Strategy           : {:<39}  ║\n"
          "║  Max Outer Iters    : {:<39}  ║\n"
          "║  Tolerance          : {:<39.6e}  ║\n"
          "╚════════════════════════════════════════════════════════════════╝\n\n",
          m_dim, m_options.block_size, m_num_blocks,
          strategy_to_string(m_options.strategy),
          m_options.max_outer_iterations, m_options.tolerance
        );
      }
    
      for (size_t outer_iter{0}; outer_iter < m_options.max_outer_iterations; ++outer_iter) {
        result.outer_iterations = outer_iter + 1;
      
        // Select block based on strategy
        vector<size_t> active_block;
        size_t block_idx = 0;
      
        switch (m_options.strategy) {
        case BlockStrategy::SEQUENTIAL:
          block_idx = outer_iter % m_num_blocks;
          active_block = sequential_blocks[block_idx];
          break;
      
        case BlockStrategy::RANDOM:
          active_block = generate_random_block();
          block_idx = outer_iter % m_num_blocks;  // For tracking purposes
          break;
      
        case BlockStrategy::GRADIENT_BASED:
          active_block    = generate_gradient_based_block(result.solution, current_value);
          m_global_evals += m_options.gradient_sample_size;  // Gradient estimation cost
          block_idx       = outer_iter % m_num_blocks;
          break;
      
        case BlockStrategy::ADAPTIVE:
          active_block = generate_adaptive_block();
          // Find which block this is
          for (size_t i{0}; i < m_num_blocks; ++i) {
            if (active_block == sequential_blocks[i]) {
              block_idx = i;
              break;
            }
          }
          break;
        }
      
        ++m_block_usage[block_idx];
        ++result.block_usage_count[block_idx];
        
        if (m_options.verbose && outer_iter % m_options.progress_frequency == 0) {
          string tmp;
          for (size_t i{0}; i < std::min(size_t(10), active_block.size()); ++i) {
            tmp += fmt::format("{}", active_block[i]);
            if (i < active_block.size() - 1) tmp += ",";
          }
          if (active_block.size() > 10) tmp += "...";
          fmt::print(
            "  [Outer {:>3}] Block {:>2} (dims: {}) | F={:<12.6e}\n",
            outer_iter, block_idx, tmp, current_value
          );
        }
        
        // Optimize over selected block
        auto subresult = m_subspace_optimizer.minimize( result.solution, active_block, callback );
        
        // Aggiungi stampe per le sotto-iterazioni:
        if ( m_options.verbose &&
             (outer_iter % m_options.progress_frequency) == 0 ) {
          string tmp;
          for (size_t i{0}; i < std::min(size_t(10), active_block.size()); ++i) {
            tmp += fmt::format("{}", active_block[i]);
            if (i < active_block.size() - 1) tmp += ",";
          }
          if (active_block.size() > 10) tmp += "...";
          fmt::print(
            "  [Outer {:>3}] Block {:>2} (dims: {}) | SubIter: {:>4} | F={:<12.6e} | Evals: {}\n",
            outer_iter, block_idx, tmp,
            subresult.iterations,
            current_value,
            subresult.function_evaluations
          );
        }
        
        
        result.solution                   = m_subspace_optimizer.get_full_solution();
        m_global_evals                   += subresult.function_evaluations;
        result.total_subspace_iterations += subresult.iterations;
      
        Scalar new_value            = subresult.final_function_value;
        Scalar improvement          = current_value - new_value;
        Scalar relative_improvement = improvement / (1.0 + std::abs(current_value));
        
        // Update adaptive history
        if (m_options.strategy == BlockStrategy::ADAPTIVE) {
          update_block_history(block_idx, std::max(Scalar(0), relative_improvement));
        }
        
        current_value = new_value;
        
        // Check for convergence
        if (relative_improvement < m_options.tolerance) {
          ++stagnation_count;
          if (stagnation_count >= m_options.stagnation_threshold) {
            result.status = Status::STAGNATED;
            break;
          }
        } else {
          stagnation_count = 0;
        }
      
        // Check global convergence (compare to previous cycle)
        if (outer_iter > 0 && outer_iter % m_num_blocks == 0) {
          Scalar cycle_improvement = std::abs(current_value - previous_value);
          Scalar cycle_relative = cycle_improvement / (1.0 + std::abs(previous_value));
      
          if (cycle_relative < m_options.tolerance) {
            result.status = Status::CONVERGED;
            break;
          }
          previous_value = current_value;
        }
      
        // Check evaluation budget
        if ( m_global_evals >= m_options.max_function_evaluations ) {
          result.status = Status::MAX_FUNCTION_EVALUATIONS;
          break;
        }
      }
    
      if (result.status == Status::FAILED) {
        result.status = Status::MAX_OUTER_ITERATIONS;
      }
      
      result.final_function_value = current_value;
      result.function_evaluations = m_global_evals;
      
      if (m_options.verbose) {

        string tmp;
        for (size_t i{0}; i < std::min(size_t(10), result.block_usage_count.size()); ++i) {
          tmp += fmt::format("{}", result.block_usage_count[i]);
          if (i < result.block_usage_count.size() - 1) tmp += ",";
        }
        if (result.block_usage_count.size() > 10) tmp += "...";

        fmt::print(
          "\n"
          "╔════════════════════════════════════════════════════════════════╗\n"
          "║                    Optimization Finished                       ║\n"
          "╠════════════════════════════════════════════════════════════════╣\n"
          "║  Final Status       : {:<39}  ║\n"
          "║  Final Value        : {:<39.6e}  ║\n"
          "║  Improvement        : {:<39.6e}  ║\n"
          "║  Outer Iterations   : {:<39}  ║\n"
          "║  Subspace Iters     : {:<39}  ║\n"
          "║  Total Evals        : {:<39}  ║\n"
          "║  Block Usage        : {:<39}  ║\n"
          "╚════════════════════════════════════════════════════════════════╝\n",
          status_to_string(result.status),
          result.final_function_value,
          result.initial_function_value - result.final_function_value,
          result.outer_iterations,
          result.total_subspace_iterations,
          result.function_evaluations,
          tmp
        );
      }
      return result;
    }
  
    static string status_to_string(Status status) {
      switch (status) {
      case Status::CONVERGED:                return "CONVERGED";
      case Status::MAX_OUTER_ITERATIONS:     return "MAX_OUTER_ITERATIONS";
      case Status::MAX_FUNCTION_EVALUATIONS: return "MAX_FUN_EVALUATIONS";
      case Status::STAGNATED:                return "STAGNATED";
      case Status::FAILED:                   return "FAILED";
      default:                               return "UNKNOWN";
      }
    }
    
    static string strategy_to_string(BlockStrategy strategy) {
      switch (strategy) {
      case BlockStrategy::SEQUENTIAL:     return "Sequential";
      case BlockStrategy::RANDOM:         return "Random";
      case BlockStrategy::GRADIENT_BASED: return "Gradient-Based";
      case BlockStrategy::ADAPTIVE:       return "Adaptive";
      default:                            return "Unknown";
      }
    }
  };

  /**
   * @class NelderMead_Hybrid
   * @brief Hybrid approach combining full-space and block-coordinate optimization
   *
   * For very high-dimensional problems, this combines:
   * 1. Initial block-coordinate descent to get close to optimum
   * 2. Full-space Nelder-Mead refinement on promising subspaces
   * 3. Adaptive switching between strategies
   */
  template <typename Scalar = double>
  class NelderMead_Hybrid {
  public:
    using Vector        = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Callback      = std::function<Scalar(Vector const &)>;
    using BlockStrategy = typename NelderMead_BlockCoordinate<Scalar>::BlockStrategy;

    enum class Status {
      CONVERGED,
      MAX_ITERATIONS,
      MAX_FUNCTION_EVALUATIONS,
      FAILED
    };
   
    struct Options {
      size_t max_iterations{500};
      size_t max_function_evaluations{200000};
      Scalar tolerance{1e-6};
   
      // Phase 1: Block coordinate descent
      size_t block_phase_iterations{100};
      size_t block_size{10};
      BlockStrategy block_strategy{BlockStrategy::ADAPTIVE};
   
      // Phase 2: Full-space or large-subspace refinement
      size_t refinement_phase_iterations{50};
      size_t refinement_subspace_size{20};  // 0 = full space
      bool use_most_important_dims{true};
   
      bool verbose{false};
    };
  
  struct Result {
    Vector solution;
    Status status{Status::FAILED};
    size_t total_iterations{0};
    size_t function_evaluations{0};
    Scalar final_function_value{0};
    Scalar initial_function_value{0};
    size_t block_phase_iterations{0};
    size_t refinement_phase_iterations{0};
  };

private:
  Options m_options;
  Vector  m_lower;
  Vector  m_upper;
  bool    m_use_bounds{false};
  
  /**
   * @brief Identify most important dimensions based on variance/gradient
   */
  vector<size_t>
  identify_important_dimensions(
    Vector   const & x,
    Callback const & callback,
    size_t           num_dims
  ) {
    size_t total_dim = x.size();
    num_dims = std::min(num_dims, total_dim);
    
    vector<std::pair<Scalar, size_t>> importance;
    
    // Estimate importance via finite differences
    Scalar eps = 1e-6;
    Scalar f_x = callback(x);
    
    for (size_t i{0}; i < total_dim; ++i) {
      Vector x_pert = x;
      x_pert(i) += eps;
      
      if (m_use_bounds) {
        x_pert(i) = std::clamp(x_pert(i), m_lower(i), m_upper(i));
      }
      
      Scalar f_pert = callback(x_pert);
      Scalar sensitivity = std::abs(f_pert - f_x) / eps;
      
      importance.emplace_back(sensitivity, i);
    }
    
    // Sort by importance
    std::sort(importance.begin(), importance.end(),
              [](auto const & a, auto const & b) { return a.first > b.first; });
    
    vector<size_t> important_dims;
    for (size_t i{0}; i < num_dims; ++i) {
      important_dims.push_back(importance[i].second);
    }
    
    return important_dims;
  }

public:
  explicit
  NelderMead_Hybrid(Options const & opts = Options())
  : m_options(opts) {}
  
  void
  set_bounds(Vector const & lower, Vector const & upper) {
    m_lower      = lower;
    m_upper      = upper;
    m_use_bounds = true;
  }
  
  Result
  minimize(Vector const & x0, Callback const & callback) {
    Result result;
    result.solution = x0;
    size_t total_evals = 0;
    
    if (m_options.verbose) {
      fmt::print(
        "╔════════════════════════════════════════════════════════════════╗\n"
        "║              Hybrid Nelder-Mead Optimization                   ║\n"
        "╠════════════════════════════════════════════════════════════════╣\n"
        "║  Dimension          : {:<39}  ║\n"
        "║  Strategy           : Block → Refinement                       ║\n"
        "╚════════════════════════════════════════════════════════════════╝\n\n",
        x0.size()
      );
    }
    
    // PHASE 1: Block Coordinate Descent
    if (m_options.verbose) {
      fmt::print("  [Phase 1] Block Coordinate Descent\n");
    }
    
    typename NelderMead_BlockCoordinate<Scalar>::Options block_opts;
    block_opts.block_size               = m_options.block_size;
    block_opts.max_outer_iterations     = m_options.block_phase_iterations;
    block_opts.max_function_evaluations = m_options.max_function_evaluations / 2;
    block_opts.strategy                 = m_options.block_strategy;
    block_opts.tolerance                = m_options.tolerance;
    block_opts.verbose                  = m_options.verbose;
    
    NelderMead_BlockCoordinate<Scalar> block_optimizer(block_opts);
    if (m_use_bounds) {
      block_optimizer.set_bounds(m_lower, m_upper);
    }
    
    auto block_result = block_optimizer.minimize(result.solution, callback);
    result.solution               = block_result.solution;
    result.block_phase_iterations = block_result.total_subspace_iterations;
    total_evals                  += block_result.function_evaluations;
    
    if (m_options.verbose) {
      fmt::print("  [Phase 1] Complete: F={:.6e}, Evals={}\n\n",
                 block_result.final_function_value,
                 block_result.function_evaluations);
    }
    
    // PHASE 2: Refinement on important subspace or full space
    if (m_options.verbose) {
      fmt::print("  [Phase 2] Refinement\n");
    }
    
    if (m_options.refinement_subspace_size > 0 && 
        m_options.refinement_subspace_size < x0.size()) {
      // Subspace refinement
      vector<size_t> important_dims = identify_important_dimensions( result.solution, callback, m_options.refinement_subspace_size );
      total_evals += x0.size();  // Cost of importance estimation
      
      if (m_options.verbose) {
        fmt::print("  [Phase 2] Optimizing {} most important dimensions\n", important_dims.size());
      }
      
      typename NelderMead_Subspace<Scalar>::Options sub_opts;
      sub_opts.max_iterations           = m_options.refinement_phase_iterations;
      sub_opts.max_function_evaluations = m_options.max_function_evaluations - total_evals;
      sub_opts.tolerance                = m_options.tolerance;
      sub_opts.verbose                  = m_options.verbose;
      
      NelderMead_Subspace<Scalar> subspace_optimizer(sub_opts);
      if (m_use_bounds) {
        subspace_optimizer.set_bounds(m_lower, m_upper);
      }
      
      auto sub_result = subspace_optimizer.minimize(
        result.solution, important_dims, callback
      );
      
      result.solution                    = subspace_optimizer.get_full_solution();
      result.refinement_phase_iterations = (m_options.refinement_subspace_size > 0 && 
                                            m_options.refinement_subspace_size < x0.size()) ?
                                            sub_result.iterations : 0;
      result.total_iterations = result.block_phase_iterations + result.refinement_phase_iterations;
      total_evals                       += sub_result.function_evaluations;
      result.final_function_value        = sub_result.final_function_value;
      
    } else {
      // Full-space refinement (for moderate dimensions)
      if (m_options.verbose) {
        fmt::print("  [Phase 2] Full-space refinement not implemented\n");
        fmt::print("            (would use original NelderMead_classic)\n");
      }
        result.final_function_value = callback(result.solution);
        ++total_evals;
      }
  
      result.function_evaluations = total_evals;
      result.total_iterations = result.block_phase_iterations +
                                result.refinement_phase_iterations;
  
      // Determine final status
      if (total_evals >= m_options.max_function_evaluations) {
        result.status = Status::MAX_FUNCTION_EVALUATIONS;
      } else if (result.total_iterations >= m_options.max_iterations) {
        result.status = Status::MAX_ITERATIONS;
      } else {
        Scalar improvement = result.initial_function_value - result.final_function_value;
        Scalar relative = improvement / (1.0 + std::abs(result.initial_function_value));
        result.status = (relative > m_options.tolerance) ?
          Status::CONVERGED : Status::CONVERGED;
      }
  
      if (m_options.verbose) {
        fmt::print(
          "\n"
          "╔════════════════════════════════════════════════════════════════╗\n"
          "║                    Optimization Complete                       ║\n"
          "╠════════════════════════════════════════════════════════════════╣\n"
          "║  Final Value        : {:<39.6e}  ║\n"
          "║  Improvement        : {:<39.6e}  ║\n"
          "║  Total Evals        : {:<39}  ║\n"
          "║  Block Phase Iters  : {:<39}  ║\n"
          "║  Refine Phase Iters : {:<39}  ║\n"
          "╚════════════════════════════════════════════════════════════════╝\n",
          result.final_function_value,
          result.initial_function_value - result.final_function_value,
          result.function_evaluations,
          result.block_phase_iterations,
          result.refinement_phase_iterations
        );
      }
  
      return result;
    }
  
    static string status_to_string(Status status) {
      switch (status) {
        case Status::CONVERGED: return "CONVERGED";
        case Status::MAX_ITERATIONS: return "MAX_ITERATIONS";
        case Status::MAX_FUNCTION_EVALUATIONS: return "MAX_FUN_EVALUATIONS";
        case Status::FAILED: return "FAILED";
        default: return "UNKNOWN";
      }
    }
  };
  
  
  // ============================================================================
  // USAGE EXAMPLES
  // ============================================================================
  
  /*
  
  // Example 1: Optimize on a subspace
  NelderMead_Subspace<double>::Options sub_opts;
  sub_opts.tolerance = 1e-6;
  sub_opts.verbose = true;
  
  NelderMead_Subspace<double> optimizer(sub_opts);
  
  Vector x_full = Vector::Random(100);  // 100-dimensional problem
  vector<size_t> active_dims = {0, 5, 10, 15, 20};  // Optimize these 5 dimensions
  
  auto result = optimizer.minimize(x_full, active_dims,
    [](Vector const & x) { return x.squaredNorm(); });
  
  // Get full solution with updated active dimensions
  Vector solution = optimizer.get_full_solution();
  
  
  // Example 2: Block coordinate descent
  NelderMead_BlockCoordinate<double>::Options block_opts;
  block_opts.block_size = 10;
  block_opts.max_outer_iterations = 100;
  block_opts.strategy = NelderMead_BlockCoordinate<double>::BlockStrategy::ADAPTIVE;
  block_opts.verbose = true;
  
  NelderMead_BlockCoordinate<double> block_optimizer(block_opts);
  
  Vector x0 = Vector::Random(100);
  auto block_result = block_optimizer.minimize(x0,
    [](Vector const & x) { return x.squaredNorm(); });
  
  
  // Example 3: Hybrid optimization for very large problems
  NelderMead_Hybrid<double>::Options hybrid_opts;
  hybrid_opts.block_size = 15;
  hybrid_opts.block_phase_iterations = 50;
  hybrid_opts.refinement_subspace_size = 30;
  hybrid_opts.verbose = true;
  
  NelderMead_Hybrid<double> hybrid_optimizer(hybrid_opts);
  
  Vector x0_large = Vector::Random(500);
  auto hybrid_result = hybrid_optimizer.minimize(x0_large,
    [](Vector const & x) { return x.squaredNorm(); });
  
  */

} // namespace Utils

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

//
// eof: Utils_NelderMead.hh
//
