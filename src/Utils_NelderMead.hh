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

  /**
   * @brief Implementation of the Nelder-Mead simplex algorithm for multidimensional optimization
   * 
   * This class provides an implementation of the Nelder-Mead simplex algorithm with extensions
   * for high-dimensional optimization problems. It supports various strategies including:
   * - Standard Nelder-Mead for low-dimensional problems
   * - Random Subspace Method (RSM) for high-dimensional problems
   * - Block Coordinate Descent (BCD) for structured high-dimensional problems
   * - Adaptive subspace selection
   * - Mixed strategies combining different approaches
   * 
   * @tparam Scalar The floating-point type used for calculations (default: double)
   */
  template <typename Scalar = double>
  class NelderMead_minimizer {
  public:
    /// Vector type (Eigen dynamic vector)
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    /// Matrix type (Eigen dynamic matrix)
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    /// Callback function type for objective function evaluation
    using Callback = std::function<Scalar(Vector const &)>;

    /// Index vector type
    using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    /**
     * @brief Optimization strategies for different problem types
     */
    enum class Strategy {
      STANDARD,           ///< Standard Nelder-Mead (works well for N < 50)
      RANDOM_SUBSPACE,    ///< Random Subspace Method for high-dimensional problems
      BLOCK_COORDINATE,   ///< Block Coordinate Descent for structured problems
      ADAPTIVE_SUBSPACE,  ///< Automatically selects best strategy based on problem dimension
      MIXED_STRATEGY      ///< Combines RSM and BCD for robust high-dimensional optimization
    };

    /**
     * @brief Optimization status codes
     */
    enum class Status {
      CONVERGED,           ///< Successfully converged to tolerance
      MAX_ITERATIONS,      ///< Reached maximum iteration limit
      MAX_FUN_EVALUATIONS, ///< Reached maximum function evaluation limit
      SIMPLEX_TOO_SMALL,   ///< Simplex became too small to continue
      SUBSPACE_CONVERGED,  ///< Converged in subspace optimization
      FAILED               ///< Optimization failed
    };

    /**
     * @brief Subspace selection methods for high-dimensional optimization
     */
    enum class SubspaceMethod {
      RANDOM,              ///< Random subspace selection
      VARIABLE_IMPORTANCE, ///< Based on variable importance estimates
      CORRELATION_BASED,   ///< Based on correlation structure
      ADAPTIVE_SIZE        ///< Adaptively adjusts subspace size
    };

    /**
     * @brief Structure containing optimization results
     */
    struct Result {
      Status status{Status::FAILED};     ///< Optimization status
      size_t iterations{0};              ///< Total iterations performed
      size_t function_evaluations{0};    ///< Total function evaluations
      Scalar final_function_value{0};    ///< Best function value found
      Scalar initial_function_value{0};  ///< Initial function value
      Vector solution;                   ///< Best solution found
      Scalar simplex_volume{0};          ///< Final simplex volume
      Scalar simplex_diameter{0};        ///< Final simplex diameter
      size_t subspace_iterations{0};     ///< Subspace optimization iterations
      vector<size_t> active_subspace;    ///< Indices of active subspace variables
    };

    /**
     * @brief Configuration options for the optimizer
     */
    struct Options {
      // Basic options
      size_t max_iterations{1000};                   ///< Maximum number of iterations
      size_t max_function_evaluations{5000};         ///< Maximum function evaluations
      Scalar tolerance{1e-8};                        ///< Function value tolerance for convergence
      Scalar simplex_tolerance{1e-12};               ///< Simplex size tolerance
      Scalar volume_tolerance{0.01};                 ///< Volume change tolerance
      
      // Nelder-Mead parameters
      Scalar rho{1.0};                               ///< Reflection coefficient
      Scalar chi{2.0};                               ///< Expansion coefficient
      Scalar gamma{0.5};                             ///< Contraction coefficient
      Scalar sigma{0.5};                             ///< Shrink coefficient
      Scalar initial_step{0.1};                      ///< Initial step size for simplex initialization
      
      // High-dimensional strategy
      Strategy strategy{Strategy::ADAPTIVE_SUBSPACE}; ///< Optimization strategy
      size_t   max_dimension_standard{50};            ///< Max dimension for standard Nelder-Mead
      
      // Random Subspace Method options
      SubspaceMethod subspace_method{SubspaceMethod::ADAPTIVE_SIZE}; ///< Subspace selection method
      size_t subspace_min_size{5};                   ///< Minimum subspace dimension
      size_t subspace_max_size{20};                  ///< Maximum subspace dimension
      size_t subspace_iterations{10};                ///< Iterations per subspace optimization
      Scalar subspace_tolerance{1e-6};               ///< Tolerance for subspace optimization
      bool   cyclic_subspaces{true};                 ///< Use cyclic subspace selection
      
      // Block Coordinate Descent options
      size_t block_size{10};                         ///< Size of each block
      size_t blocks_per_iteration{3};                ///< Number of blocks to optimize per iteration
      bool   random_block_order{true};               ///< Randomize block order
      
      // Adaptive parameters
      bool   adaptive_parameters{true};              ///< Adapt parameters based on dimension
      bool   adaptive_subspace_size{true};           ///< Adapt subspace size during optimization
      Scalar subspace_size_reduction{0.95};          ///< Subspace size reduction factor
      
      // Monitoring and output
      bool verbose{false};                           ///< Enable verbose output
      bool monitor_subspaces{false};                 ///< Monitor subspace optimization
      size_t progress_frequency{100};                ///< Progress report frequency
    };

  private:
    Options m_options;                               ///< Optimization options
    Vector  m_lower;                                 ///< Lower bounds
    Vector  m_upper;                                 ///< Upper bounds
    bool    m_use_bounds{false};                     ///< Whether bounds are active

    /// Simplex vertices stored as vector of vectors
    vector<Vector> m_simplex;

    /// Function values at simplex vertices
    vector<Scalar> m_values;

    /// Centroid of simplex (excluding worst point)
    Vector m_centroid;

    /// Trial point for reflection/expansion/contraction
    Vector m_trial_point;

    // High-dimensional optimization state
    Vector m_current_solution;                  ///< Current best solution
    Vector m_variable_importance;               ///< Variable importance estimates
    Matrix m_correlation_estimate;              ///< Correlation matrix estimate
    vector<vector<size_t>> m_blocks;            ///< Variable blocks for BCD
    
    // Random number generation
    std::mt19937 m_rng;                              ///< Random number generator
    std::uniform_real_distribution<Scalar> m_uniform_dist{0, 1}; ///< Uniform distribution
    
    // Internal state
    size_t m_dim{0};                                 ///< Problem dimension
    Scalar m_eps{std::numeric_limits<Scalar>::epsilon()}; ///< Machine epsilon
    size_t m_subspace_size{0};                       ///< Current subspace size
    size_t m_current_iteration{0};                   ///< Current iteration counter

    /**
     * @brief Project a point to feasible bounds
     * @param x Point to project (modified in-place)
     */
    void
    project_point( Vector & x ) const {
      if (m_use_bounds) x = x.cwiseMax(m_lower).cwiseMin(m_upper);
    }

    /**
     * @brief Initialize the Nelder-Mead simplex around initial point
     * @param x0 Initial point
     * @param callback Objective function
     */
    void
    initialize_simplex( Vector const & x0, Callback const & callback ) {
      m_dim = x0.size();
      
      // Resize simplex storage
      m_simplex.resize(m_dim + 1);
      m_values.resize(m_dim + 1);
      m_centroid.resize(m_dim);
      m_trial_point.resize(m_dim);

      // Initialize all vectors with proper size
      for ( auto & vec : m_simplex ) vec.resize(m_dim);

      // Set first vertex to initial point
      m_simplex[0] = x0;
      project_point(m_simplex[0]);
      m_values[0] = callback(m_simplex[0]);
      
      // Create other vertices by perturbing each coordinate
      Vector x_plus  { x0 };
      Vector x_minus { x0 };
      for ( size_t i{0}; i < m_dim; ++i ) {
      
        auto & xp{ x_plus.coeffRef(i)  };
        auto & xm{ x_minus.coeffRef(i) };
        
        xp += m_options.initial_step;
        xm -= m_options.initial_step;

        if ( m_use_bounds ) {
          xp = min( xp, m_upper.coeff(i) );
          xm = max( xm, m_lower.coeff(i) );
        }

        Scalar f_plus  = callback(x_plus);
        Scalar f_minus = callback(x_minus);

        // Keep the better perturbation
        if ( f_plus < f_minus ) {
          m_simplex[i + 1] = x_plus;
          m_values[i + 1] = f_plus;
        } else {
          m_simplex[i + 1] = x_minus;
          m_values[i + 1] = f_minus;
        }

        xp = xm = x0.coeff(i);
      }
    }

    /**
     * @brief Update centroid (average of all but worst point)
     * @param worst_index Index of worst point to exclude
     */
    void
    update_centroid( size_t worst_index ) {
      m_centroid.setZero();
      for ( size_t i{0}; i <= m_dim; ++i ) {
        if ( i != worst_index )
          m_centroid += m_simplex[i];
      }
      m_centroid /= static_cast<Scalar>(m_dim);
    }

    /**
     * @brief Compute maximum distance between any two simplex vertices
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
     * @brief Compute approximate simplex volume
     * @return Simplex volume
     */
    Scalar
    compute_volume() const {
      if ( m_dim == 0 ) return 0;
      Matrix basis(m_dim, m_dim);
      for ( size_t i{0}; i < m_dim; ++i )
        basis.col(i) = m_simplex[i+1] - m_simplex[0];
      // Volume = |det(basis)| / n!
      return abs(basis.determinant()) / std::tgamma(m_dim + 1);
    }

    /**
     * @brief Sort vertices by function value and return indices
     * @return Vector of indices sorted by function value (best first)
     */
    vector<size_t>
    sort_vertices() const {
      vector<size_t> indices(m_dim + 1);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [this](size_t i, size_t j) { return m_values[i] < m_values[j]; });
      return indices;
    }

    /**
     * @brief Perform one iteration of Nelder-Mead algorithm
     * @param callback Objective function
     * @param function_evaluations Reference to function evaluation counter
     * @return true if converged, false otherwise
     */
    bool
    nelder_mead_iteration( Callback const & callback, size_t & function_evaluations ) {
      auto indices = sort_vertices();
      size_t best         = indices[0];
      size_t second_worst = indices[m_dim - 1];
      size_t worst        = indices[m_dim];

      // Check convergence
      Scalar value_range = m_values[worst] - m_values[best];
      if ( value_range < m_options.tolerance ) return true;

      update_centroid( worst );

      // Reflection
      Scalar f_reflect = reflect_point(callback, worst); ++function_evaluations;

      if ( f_reflect < m_values[best] ) {
        // Expansion
        Scalar f_expand = expand_point(callback); ++function_evaluations;
        m_simplex[worst] = m_trial_point;
        m_values[worst]  = min( f_expand, f_reflect );
      } else if (f_reflect < m_values[second_worst]) {
        // Accept reflection
        m_simplex[worst] = m_trial_point;
        m_values[worst]  = f_reflect;
      } else {
        if (f_reflect < m_values[worst]) {
          // Outside contraction
          Scalar f_contract = contract_point(callback, worst, true);
          ++function_evaluations;

          if (f_contract <= f_reflect) {
            m_simplex[worst] = m_trial_point;
            m_values[worst]  = f_contract;
          } else {
            shrink_simplex(callback, best);
            function_evaluations += m_dim;
          }
        } else {
          // Inside contraction
          Scalar f_contract = contract_point(callback, worst, false);
          ++function_evaluations;

          if (f_contract < m_values[worst]) {
            m_simplex[worst] = m_trial_point;
            m_values[worst] = f_contract;
          } else {
            shrink_simplex(callback, best);
            function_evaluations += m_dim;
          }
        }
      }

      return false;
    }

    /**
     * @brief Perform reflection operation
     * @param callback Objective function
     * @param worst_index Index of worst point
     * @return Function value at reflected point
     */
    Scalar
    reflect_point( Callback const & callback, size_t worst_index ) {
      m_trial_point = m_centroid + m_options.rho * (m_centroid - m_simplex[worst_index]);
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform expansion operation
     * @param callback Objective function
     * @return Function value at expanded point
     */
    Scalar
    expand_point( Callback const & callback ) {
      m_trial_point = m_centroid + m_options.chi * (m_trial_point - m_centroid);
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform contraction operation
     * @param callback Objective function
     * @param worst_index Index of worst point
     * @param outside Whether to perform outside contraction (true) or inside contraction (false)
     * @return Function value at contracted point
     */
    Scalar
    contract_point( Callback const & callback, size_t worst_index, bool outside = true ) {
      if (outside) m_trial_point = m_centroid + m_options.gamma * (m_trial_point - m_centroid);
      else         m_trial_point = m_centroid - m_options.gamma * (m_centroid - m_simplex[worst_index]);
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform shrink operation (contract all points toward best)
     * @param callback Objective function
     * @param best_index Index of best point
     */
    void
    shrink_simplex( Callback const & callback, size_t best_index ) {
      Vector best = m_simplex[best_index];
      // Shrink all points except the best one
      for ( size_t i{0}; i <= m_dim; ++i ) {
        if ( i != best_index ) {
          m_simplex[i] = best + m_options.sigma * (m_simplex[i] - best);
          project_point(m_simplex[i]);
          m_values[i] = callback(m_simplex[i]);
        }
      }
    }

    /**
     * @brief Select subspace for Random Subspace Method
     * @return Vector of indices representing selected subspace
     */
    vector<size_t>
    select_subspace() {
      vector<size_t> all_indices(m_dim);
      std::iota( all_indices.begin(), all_indices.end(), 0 );
      
      auto & min_sz{ m_options.subspace_min_size };
      auto & max_sz{ m_options.subspace_max_size };
      
      if ( m_options.adaptive_subspace_size ) {
        // Adaptive subspace size: decreases with iterations
        m_subspace_size = static_cast<size_t>(
          min_sz + (max_sz - min_sz) * std::exp(-static_cast<Scalar>(m_current_iteration) / 100.0)
        );
        m_subspace_size = std::clamp( m_subspace_size, min_sz, max_sz );
      } else {
        m_subspace_size = max_sz;
      }

      // Ensure subspace size doesn't exceed problem dimension
      m_subspace_size = min(m_subspace_size, m_dim);

      std::shuffle( all_indices.begin(), all_indices.end(), m_rng );
      return vector<size_t>(all_indices.begin(), all_indices.begin() + m_subspace_size);
    }

    /**
     * @brief Create blocks for Block Coordinate Descent
     */
    void
    initialize_blocks() {
      m_blocks.clear();
      size_t num_blocks = (m_dim + m_options.block_size - 1) / m_options.block_size;
      
      for ( size_t i{0}; i < num_blocks; ++i ) {
        vector<size_t> block;
        size_t start = i * m_options.block_size;
        size_t end   = min((i + 1) * m_options.block_size, m_dim );
        
        for ( size_t j{start}; j < end; ++j ) block.push_back(j);
        m_blocks.push_back(block);
      }
    }

    /**
     * @brief Optimize in selected subspace using standard Nelder-Mead
     * @param subspace Indices of variables in the subspace
     * @param full_x Current full-dimensional solution
     * @param callback Objective function
     * @return Optimization result for the subspace
     */
    Result
    optimize_subspace( vector<size_t> const & subspace, Vector const & full_x, Callback const & callback ) {
      size_t subspace_dim = subspace.size();
      
      // Create subspace callback that maps subspace coordinates to full space
      auto subspace_callback = [&](Vector const & subspace_x) -> Scalar {
        Vector full_point = full_x;
        for (size_t i{0}; i < subspace_dim; ++i) full_point(subspace[i]) = subspace_x(i);
        return callback(full_point);
      };

      // Extract subspace initial point
      Vector subspace_x0(subspace_dim);
      for (size_t i{0}; i < subspace_dim; ++i) subspace_x0(i) = full_x(subspace[i]);

      // Extract subspace bounds if available
      Vector subspace_lower, subspace_upper;
      if ( m_use_bounds ) {
        subspace_lower.resize(subspace_dim);
        subspace_upper.resize(subspace_dim);
        for (size_t i{0}; i < subspace_dim; ++i) {
          subspace_lower.coeffRef(i) = m_lower(subspace[i]);
          subspace_upper.coeffRef(i) = m_upper(subspace[i]);
        }
      }

      // Configure subspace optimizer
      Options subspace_opts = m_options;
      subspace_opts.max_iterations = m_options.subspace_iterations;
      subspace_opts.tolerance      = m_options.subspace_tolerance;
      subspace_opts.strategy       = Strategy::STANDARD;  // Always use standard NM in subspace
      subspace_opts.verbose        = m_options.monitor_subspaces;

      NelderMead_minimizer<Scalar> subspace_optimizer(subspace_opts);
      if ( m_use_bounds ) subspace_optimizer.set_bounds(subspace_lower, subspace_upper);

      return subspace_optimizer.minimize(subspace_x0, subspace_callback);
    }

    /**
     * @brief Standard Nelder-Mead optimization for low-dimensional problems
     * @param x0 Initial point
     * @param callback Objective function
     * @return Optimization result
     */
    Result
    minimize_standard( Vector const & x0, Callback const & callback ) {
      Result result;
      result.initial_function_value = callback(x0);

      m_dim = x0.size();
      
      // Handle trivial case: zero-dimensional problem
      if ( m_dim == 0 ) {
        result.status               = Status::CONVERGED;
        result.solution             = x0;
        result.final_function_value = result.initial_function_value;
        return result;
      }

      initialize_simplex(x0, callback);
      result.function_evaluations = m_dim + 1;

      // Adapt parameters based on problem dimension
      if ( m_options.adaptive_parameters && m_dim > 0 ) {
        Scalar n = static_cast<Scalar>(m_dim);
        m_options.rho   = 1.0;
        m_options.chi   = 1.0 + 2.0 / n;
        m_options.gamma = 0.75 - 1.0 / (2.0 * n);
        m_options.sigma = 1.0 - 1.0 / n;
        if ( m_options.verbose )
          fmt::print(
            "[NM] Adaptive parameters: rho={:.3f}, chi={:.3f}, gamma={:.3f}, sigma={:.3f}\n",
            m_options.rho, m_options.chi, m_options.gamma, m_options.sigma
          );
      }
      
      if ( m_options.verbose )
        fmt::print(
          "[NM] Starting Standard Nelder-Mead (dimension={})\n"
          "[NM] Max iterations: {}, Max evaluations: {}\n",
          m_dim, m_options.max_iterations, m_options.max_function_evaluations
        );

      // Main optimization loop
      auto & iter     = result.iterations;
      auto & max_iter = m_options.max_iterations;
      for ( iter = 0; iter < max_iter; ++iter ) {

        if ( result.function_evaluations >= m_options.max_function_evaluations ) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        if ( nelder_mead_iteration(callback, result.function_evaluations) ) {
          result.status = Status::CONVERGED;
          break;
        }

        result.simplex_diameter = compute_diameter();
        if ( result.simplex_diameter < m_options.simplex_tolerance ) {
          result.status = Status::SIMPLEX_TOO_SMALL;
          break;
        }
        
        // Progress reporting
        if ( m_options.verbose && (iter % m_options.progress_frequency) == 0 ) {
          auto indices = sort_vertices();
          Scalar progress = (100.0 * iter) / max_iter;
          fmt::print(
            "[NM] Iter {:4d} ({:5.1f}%): F_best = {:<12.6g}, diameter = {:<8.2e}, evals = {}\n",
            iter, progress, m_values[indices[0]], result.simplex_diameter, result.function_evaluations
          );
        }
      }

      // Extract best solution
      auto indices = sort_vertices();
      result.solution             = m_simplex[indices[0]];
      result.final_function_value = m_values[indices[0]];
      result.simplex_volume       = compute_volume();
      result.simplex_diameter     = compute_diameter();
      
      // Set status if maximum iterations reached
      if ( iter >= max_iter && result.status == Status::FAILED )
        result.status = Status::MAX_ITERATIONS;
      
      // Final report
      if ( m_options.verbose ) {
        fmt::print(
          "\n"
          "[NM] --- Standard Nelder-Mead Complete ---\n"
          "[NM] Status: {}\n"
          "[NM] Function value: {:12.6g} (initial: {:12.6g})\n"
          "[NM] Iterations: {}, Function evaluations: {}\n"
          "[NM] Simplex diameter: {:8.2e}, volume: {:8.2e}\n"
          "[NM] --------------------------------------\n",
          status_to_string(result.status),
          result.final_function_value, result.initial_function_value,
          iter, result.function_evaluations,
          result.simplex_diameter, result.simplex_volume
        );
      }

      return result;
    }

    /**
     * @brief Random Subspace Method for high-dimensional optimization
     * @param x0 Initial point
     * @param callback Objective function
     * @return Optimization result
     */
    Result
    minimize_random_subspace( Vector const & x0, Callback const & callback ) {
      Result result;
      m_dim = x0.size(); 
      
      result.initial_function_value = callback(x0);
      result.function_evaluations = 1;
      
      m_current_solution = x0;
      Scalar best_value = result.initial_function_value;

      std::random_device rd;
      m_rng.seed(rd());
      
      if ( m_options.verbose )
        fmt::print(
          "[RSM] Starting Random Subspace Method\n"
          "[RSM] Problem dimension: {}, Max subspace size: {}\n"
          "[RSM] Subspace iterations: {}, Tolerance: {:8.2e}\n",
          m_dim, m_options.subspace_max_size, m_options.subspace_iterations, m_options.tolerance
        );

      auto & iter     = result.iterations;
      auto & max_iter = m_options.max_iterations;
      for ( iter = 0; iter < max_iter; ++iter ) {

        m_current_iteration = result.iterations;

        if ( result.function_evaluations >= m_options.max_function_evaluations ) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        auto subspace = select_subspace();
        result.active_subspace = subspace;

        // Optimize in selected subspace
        auto subspace_result = optimize_subspace(subspace, m_current_solution, callback);
        result.function_evaluations += subspace_result.function_evaluations;
        result.subspace_iterations  += subspace_result.iterations;

        // Update full solution with subspace results
        for ( size_t i{0}; i < subspace.size(); ++i )
          m_current_solution(subspace[i]) = subspace_result.solution(i);

        // Evaluate full solution
        Scalar new_value = callback(m_current_solution);
        result.function_evaluations++;

        // Check convergence
        if ( abs(best_value - new_value) < m_options.tolerance) {
          result.status = Status::SUBSPACE_CONVERGED;
          break;
        }

        best_value = new_value;

        // Progress reporting
        if ( m_options.verbose && ( iter % m_options.progress_frequency ) == 0 ) {
          Scalar progress = (100.0 * iter) / m_options.max_iterations;
          fmt::print(
            "[RSM] Iter {:4d} ({:5.1f}%): F_best = {:<12.6g} subspace = {}/{}, evals = {}\n",
            iter, progress, best_value, subspace.size(), m_dim, result.function_evaluations
          );
        }
      }
      
      if ( iter >= max_iter && result.status == Status::FAILED ) {
        result.status = Status::MAX_ITERATIONS;
      }

      result.solution = m_current_solution;
      result.final_function_value = best_value;
      
      if ( m_options.verbose ) {
        fmt::print(
          "\n"
          "[RSM] --- Random Subspace Method Complete ---\n"
          "[RSM] Status: {}\n"
          "[RSM] Final function value: {:12.6g}\n"
          "[RSM] Total iterations: {}, function evaluations: {}\n"
          "[RSM] Subspace iterations: {}\n"
          "[RSM] ----------------------------------------\n",
          status_to_string(result.status), result.final_function_value,
          iter, result.function_evaluations, result.subspace_iterations
        );
      }
      return result;
    }

    /**
     * @brief Block Coordinate Descent for structured high-dimensional problems
     * @param x0 Initial point
     * @param callback Objective function
     * @return Optimization result
     */
    Result
    minimize_block_coordinate( Vector const & x0, Callback const & callback ) {
      Result result;
      m_dim = x0.size();
      
      result.initial_function_value = callback(x0);
      result.function_evaluations = 1;
      
      m_current_solution = x0;
      Scalar best_value = result.initial_function_value;

      initialize_blocks();
      
      if ( m_options.verbose )
        fmt::print(
          "[BCD] Starting Block Coordinate Descent\n"
          "[BCD] Problem dimension: {}, Block size: {}, Number of blocks: {}\n"
          "[BCD] Blocks per iteration: {}\n",
          m_dim, m_options.block_size, m_blocks.size(), m_options.blocks_per_iteration
        );

      auto & iter     = result.iterations;
      auto & max_iter = m_options.max_iterations;
      for ( iter = 0; iter < max_iter; ++iter ) {

        if ( result.function_evaluations >= m_options.max_function_evaluations ) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        vector<size_t> block_indices(m_blocks.size());
        std::iota( block_indices.begin(), block_indices.end(), 0 );
        
        if ( m_options.random_block_order )
          std::shuffle(block_indices.begin(), block_indices.end(), m_rng);

        size_t blocks_to_optimize = min( m_options.blocks_per_iteration, m_blocks.size() );

        // Optimize selected blocks
        for ( size_t i{0}; i < blocks_to_optimize; ++i ) {
          auto & block = m_blocks[block_indices[i]];
          
          auto block_result = optimize_subspace(block, m_current_solution, callback);
          result.function_evaluations += block_result.function_evaluations;
          result.subspace_iterations  += block_result.iterations;

          for ( size_t j{0}; j < block.size(); ++j )
            m_current_solution(block[j]) = block_result.solution(j);
        }

        Scalar new_value = callback(m_current_solution);
        result.function_evaluations++;

        if ( abs(best_value - new_value) < m_options.tolerance ) {
          result.status = Status::SUBSPACE_CONVERGED;
          break;
        }

        best_value = new_value;

        // Progress reporting
        if ( m_options.verbose && (iter % m_options.progress_frequency) == 0 ) {
          Scalar progress = (100.0 * iter) / max_iter;
          fmt::print(
            "[BCD] Iter {:4d} ({:5.1f}%): F_best = {:<12.6g}, blocks = {}/{}, evals = {}\n",
            iter, progress, best_value, blocks_to_optimize, m_blocks.size(), result.function_evaluations
          );
        }
      }
      
      if ( iter >= max_iter && result.status == Status::FAILED ) result.status = Status::MAX_ITERATIONS;

      result.solution             = m_current_solution;
      result.final_function_value = best_value;
      
      if ( m_options.verbose )
        fmt::print(
          "\n"
          "[BCD] --- Block Coordinate Descent Complete ---\n"
          "[BCD] Status: {}\n"
          "[BCD] Final function value: {:12.6g}\n"
          "[BCD] Total iterations: {}, function evaluations: {}\n"
          "[BCD] Subspace iterations: {}\n"
          "[BCD] ------------------------------------------\n",
          status_to_string(result.status), result.final_function_value,
          iter, result.function_evaluations, result.subspace_iterations
        );
      return result;
    }

  public:
    /**
     * @brief Construct a new Nelder-Mead minimizer object
     * @param opts Optimization options
     */
    explicit
    NelderMead_minimizer( Options const & opts = Options() ) : m_options(opts) {
      std::random_device rd;
      m_rng.seed(rd());
    }

    /**
     * @brief Get the current best solution
     * @return Current solution vector
     */
    Vector const & solution() const { return m_current_solution; }

    /**
     * @brief Set variable bounds for constrained optimization
     * @param lower Lower bounds vector
     * @param upper Upper bounds vector
     */
    void
    set_bounds(Vector const & lower, Vector const & upper) {
      UTILS_ASSERT(
        lower.size() == upper.size(),
        "NelderMead_minimizer::set_bounds: lower and upper must have same size\n"
      );
      UTILS_ASSERT(
        (lower.array() <= upper.array()).all(),
        "NelderMead_minimizer::set_bounds: lower must be <= upper for all components\n"
      );

      m_lower      = lower;
      m_upper      = upper;
      m_use_bounds = true;
    }

    /**
     * @brief Set variable bounds using raw arrays
     * @param n Number of variables
     * @param lower Array of lower bounds
     * @param upper Array of upper bounds
     */
    void
    set_bounds( size_t n, Scalar const lower[], Scalar const upper[] ) {
      m_lower.resize(n);
      m_upper.resize(n);
      std::copy_n(lower, n, m_lower.data());
      std::copy_n(upper, n, m_upper.data());
      m_use_bounds = true;
    }

    /**
     * @brief Main optimization routine
     * @param x0 Initial point
     * @param callback Objective function to minimize
     * @return Optimization result
     */
    Result
    minimize( Vector const & x0, Callback const & callback ) {
      m_dim = x0.size();
      m_current_solution  = x0;
      m_current_iteration = 0;
      
      // Handle trivial case: zero-dimensional problem
      if ( m_dim == 0 ) {
        Result result;
        result.status                 = Status::CONVERGED;
        result.solution               = x0;
        result.initial_function_value = callback(x0);
        result.final_function_value   = result.initial_function_value;
        result.function_evaluations   = 1;
        return result;
      }

      // Strategy selection for adaptive mode
      Strategy selected_strategy = m_options.strategy;
      if ( selected_strategy == Strategy::ADAPTIVE_SUBSPACE ) {
        if ( m_dim <= m_options.max_dimension_standard ) {
          selected_strategy = Strategy::STANDARD;
        } else if ( m_dim <= 100 ) {
          selected_strategy = Strategy::BLOCK_COORDINATE;
        } else {
          selected_strategy = Strategy::RANDOM_SUBSPACE;
        }
      }

      if ( m_options.verbose )
        fmt::print(
          "\n"
          "[OPT] --- Nelder-Mead Optimization Start ---\n"
          "[OPT] Problem dimension: {}\n"
          "[OPT] Strategy: {} ({})\n"
          "[OPT] Max iterations: {}, Max function evaluations: {}\n"
          "[OPT] Tolerance: {:8.2e}, Simplex tolerance: {:8.2e}, Bounds: {}\n"
          "[OPT] ---------------------------------------\n",
          m_dim, strategy_to_string(selected_strategy),
          (m_options.strategy == Strategy::ADAPTIVE_SUBSPACE ? "auto-selected" : "user-specified"),
          m_options.max_iterations, m_options.max_function_evaluations,
          m_options.tolerance, m_options.simplex_tolerance,
          ( m_use_bounds ? "enabled" : "disabled" )
        );

      // Dispatch to appropriate optimization method
      switch (selected_strategy) {
      case Strategy::STANDARD:
        return minimize_standard(x0, callback);
      case Strategy::RANDOM_SUBSPACE:
        return minimize_random_subspace(x0, callback);
      case Strategy::BLOCK_COORDINATE:
        return minimize_block_coordinate(x0, callback);
      case Strategy::MIXED_STRATEGY: {
        // Mixed strategy: RSM followed by BCD refinement
        Options original_options = m_options;

        auto result = minimize_random_subspace(x0, callback);
        if ( result.status == Status::SUBSPACE_CONVERGED)  {
          if ( m_options.verbose ) fmt::print("[MIXED] RSM converged, starting BCD refinement...\n");
          m_options.strategy = Strategy::BLOCK_COORDINATE;
          auto bcd_result = minimize_block_coordinate(result.solution, callback);
          // Combine results
          bcd_result.iterations           += result.iterations;
          bcd_result.function_evaluations += result.function_evaluations;
          m_options = original_options;
          return bcd_result;
        }
        m_options = original_options;
        return result;
      }
      case Strategy::ADAPTIVE_SUBSPACE:
      default:
        // Fallback to standard method
        if ( m_options.verbose ) fmt::print("[OPT] Using fallback strategy: STANDARD\n");
        return minimize_standard(x0, callback);
      }
    }

    /**
     * @brief Convert strategy enum to string
     * @param strategy Strategy enum value
     * @return String representation
     */
    static
    string
    strategy_to_string( Strategy strategy ) {
      switch ( strategy ) {
      case Strategy::STANDARD:          return "STANDARD";
      case Strategy::RANDOM_SUBSPACE:   return "RANDOM_SUBSPACE";
      case Strategy::BLOCK_COORDINATE:  return "BLOCK_COORDINATE";
      case Strategy::ADAPTIVE_SUBSPACE: return "ADAPTIVE_SUBSPACE";
      case Strategy::MIXED_STRATEGY:    return "MIXED_STRATEGY";
      default:                          return "UNKNOWN";
      }
    }

    /**
     * @brief Convert status enum to descriptive string
     * @param status Status enum value
     * @return Descriptive string
     */
    static
    string status_to_string( Status status ) {
      switch (status) {
      case Status::CONVERGED:           return "CONVERGED (tolerance met)";
      case Status::MAX_ITERATIONS:      return "MAX_ITERATIONS (iteration limit reached)";
      case Status::MAX_FUN_EVALUATIONS: return "MAX_FUN_EVALUATIONS (function evaluation limit reached)";
      case Status::SIMPLEX_TOO_SMALL:   return "SIMPLEX_TOO_SMALL (simplex became degenerate)";
      case Status::SUBSPACE_CONVERGED:  return "SUBSPACE_CONVERGED (converged in subspace optimization)";
      case Status::FAILED:              return "FAILED (unknown error)";
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
//
