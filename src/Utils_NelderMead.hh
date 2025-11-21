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
#endif  __clang__  __clang__ 

namespace Utils {

  template <typename Scalar = double>
  class NelderMead_minimizer {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Callback = std::function<Scalar(Vector const &)>;
    using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

    enum class Strategy {
      STANDARD,
      RANDOM_SUBSPACE,
      BLOCK_COORDINATE,
      ADAPTIVE_SUBSPACE,
      MIXED_STRATEGY
    };

    enum class Status {
      CONVERGED,
      MAX_ITERATIONS,
      MAX_FUN_EVALUATIONS,
      SIMPLEX_TOO_SMALL,
      SUBSPACE_CONVERGED,
      FAILED
    };

    enum class SubspaceMethod {
      RANDOM,
      VARIABLE_IMPORTANCE,
      CORRELATION_BASED,
      ADAPTIVE_SIZE
    };

    struct Result {
      Status status{Status::FAILED};
      size_t iterations{0};
      size_t function_evaluations{0};
      Scalar final_function_value{0};
      Scalar initial_function_value{0};
      Vector solution;
      Scalar simplex_volume{0};
      Scalar simplex_diameter{0};
      size_t subspace_iterations{0};
      std::vector<size_t> active_subspace;
    };

    struct Options {
      // Basic options
      size_t max_iterations{1000};
      size_t max_function_evaluations{5000};
      Scalar tolerance{1e-8};
      Scalar simplex_tolerance{1e-12};
      Scalar volume_tolerance{0.01};
      
      // Nelder-Mead parameters
      Scalar rho{1.0};
      Scalar chi{2.0};
      Scalar gamma{0.5};
      Scalar sigma{0.5};
      Scalar initial_step{0.1};
      
      // High-dimensional strategy
      Strategy strategy{Strategy::ADAPTIVE_SUBSPACE};
      size_t max_dimension_standard{50};
      
      // Random Subspace Method options
      SubspaceMethod subspace_method{SubspaceMethod::ADAPTIVE_SIZE};
      size_t subspace_min_size{5};
      size_t subspace_max_size{20};
      size_t subspace_iterations{10};
      Scalar subspace_tolerance{1e-6};
      bool cyclic_subspaces{true};
      
      // Block Coordinate Descent options
      size_t block_size{10};
      size_t blocks_per_iteration{3};
      bool random_block_order{true};
      
      // Adaptive parameters
      bool adaptive_parameters{true};
      bool adaptive_subspace_size{true};
      Scalar subspace_size_reduction{0.95};
      
      // Monitoring and output
      bool verbose{false};
      bool monitor_subspaces{false};
      size_t progress_frequency{100};
    };

  private:
    Options m_options;
    Vector  m_lower;
    Vector  m_upper;
    bool    m_use_bounds{false};

    // CHANGED: Use std::vector<Vector> for simplex
    std::vector<Vector> m_simplex;                ///< Simplex vertices as vector of vectors
    std::vector<Scalar> m_values;                 ///< Function values at vertices
    Vector m_centroid;
    Vector m_trial_point;

    // High-dimensional optimization state
    Vector m_current_solution;
    Vector m_variable_importance;
    Matrix m_correlation_estimate;
    std::vector<std::vector<size_t>> m_blocks;
    
    // Random number generation
    std::mt19937 m_rng;
    std::uniform_real_distribution<Scalar> m_uniform_dist{0, 1};
    
    // Internal state
    size_t m_dim{0};
    Scalar m_eps{std::numeric_limits<Scalar>::epsilon()};
    size_t m_subspace_size{0};
    size_t m_current_iteration{0};

    /**
     * @brief Project point to feasible bounds
     */
    void project_point(Vector & x) const {
      if (m_use_bounds) {
        for (Eigen::Index i = 0; i < x.size(); ++i) {
          if (x(i) < m_lower(i)) x(i) = m_lower(i);
          else if (x(i) > m_upper(i)) x(i) = m_upper(i);
        }
      }
    }

    /**
     * @brief Initialize standard Nelder-Mead simplex using std::vector<Vector>
     */
    void initialize_simplex(Vector const & x0, Callback const & callback) {
      m_dim = x0.size();
      
      // CHANGED: Resize as vector of vectors
      m_simplex.resize(m_dim + 1);
      m_values.resize(m_dim + 1);
      m_centroid.resize(m_dim);
      m_trial_point.resize(m_dim);

      // Initialize all vectors with proper size
      for (auto& vec : m_simplex) {
        vec.resize(m_dim);
      }

      // Set first vertex to initial point
      m_simplex[0] = x0;
      project_point(m_simplex[0]);
      m_values[0] = callback(m_simplex[0]);
      
      // BUG FIX: La logica di testing delle perturbazioni è stata corretta per non 
      // riutilizzare il vettore, cosa che causava un simplex malformato.
      // Create other vertices by perturbing each coordinate
      for (size_t i = 0; i < m_dim; ++i) {
        
        Vector x_plus = x0;
        x_plus(i) += m_options.initial_step;
        project_point(x_plus);
        Scalar f_plus = callback(x_plus);

        Vector x_minus = x0;
        x_minus(i) -= m_options.initial_step;
        project_point(x_minus);
        Scalar f_minus = callback(x_minus);

        // Keep the better perturbation
        if (f_plus < f_minus) {
          m_simplex[i + 1] = x_plus;
          m_values[i + 1] = f_plus;
        } else {
          m_simplex[i + 1] = x_minus;
          m_values[i + 1] = f_minus;
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
          m_centroid += m_simplex[i];
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
          Scalar dist = (m_simplex[i] - m_simplex[j]).norm();
          max_dist = std::max(max_dist, dist);
        }
      }
      return max_dist;
    }

    /**
     * @brief Compute approximate simplex volume
     */
    Scalar compute_volume() const {
      if (m_dim == 0) return 0;
      Matrix basis(m_dim, m_dim);
      for (size_t i = 0; i < m_dim; ++i) {
        basis.col(i) = m_simplex[i + 1] - m_simplex[0];
      }
      // Volume = |det(basis)| / n! 
      // std::tgamma(m_dim + 1) è la funzione Gamma che per interi è (m_dim)!
      return std::abs(basis.determinant()) / std::tgamma(m_dim + 1);
    }

    /**
     * @brief Sort vertices by function value and return indices
     */
    std::vector<size_t> sort_vertices() const {
      std::vector<size_t> indices(m_dim + 1);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [this](size_t i, size_t j) { return m_values[i] < m_values[j]; });
      return indices;
    }

    /**
     * @brief Core Nelder-Mead iteration
     */
    bool nelder_mead_iteration(Callback const & callback, size_t & function_evaluations) {
      auto indices = sort_vertices();
      size_t best = indices[0];
      size_t second_worst = indices[m_dim - 1];
      size_t worst = indices[m_dim]; // L'indice del punto peggiore nel vettore m_simplex

      // Check convergence
      Scalar value_range = m_values[worst] - m_values[best];
      if (value_range < m_options.tolerance) return true;

      update_centroid(worst);

      // Reflection
      // BUG FIX: Passato l'indice corretto del punto peggiore
      Scalar f_reflect = reflect_point(callback, worst);
      ++function_evaluations;

      if (f_reflect < m_values[best]) {
        // Expansion
        Scalar f_expand = expand_point(callback);
        ++function_evaluations;

        if (f_expand < f_reflect) {
          m_simplex[worst] = m_trial_point;
          m_values[worst] = f_expand;
        } else {
          m_simplex[worst] = m_trial_point;
          m_values[worst] = f_reflect;
        }
      } else if (f_reflect < m_values[second_worst]) {
        // Accept reflection
        m_simplex[worst] = m_trial_point;
        m_values[worst] = f_reflect;
      } else {
        if (f_reflect < m_values[worst]) {
          // Outside contraction
          // BUG FIX: Passato l'indice corretto del punto peggiore
          Scalar f_contract = contract_point(callback, worst, true);
          ++function_evaluations;

          if (f_contract <= f_reflect) {
            m_simplex[worst] = m_trial_point;
            m_values[worst] = f_contract;
          } else {
            // BUG FIX: Passato l'indice corretto del punto migliore
            shrink_simplex(callback, best);
            function_evaluations += m_dim;
          }
        } else {
          // Inside contraction
          // BUG FIX: Passato l'indice corretto del punto peggiore
          Scalar f_contract = contract_point(callback, worst, false);
          ++function_evaluations;

          if (f_contract < m_values[worst]) {
            m_simplex[worst] = m_trial_point;
            m_values[worst] = f_contract;
          } else {
            // BUG FIX: Passato l'indice corretto del punto migliore
            shrink_simplex(callback, best);
            function_evaluations += m_dim;
          }
        }
      }

      return false;
    }

    /**
     * @brief Perform reflection operation
     */
    // BUG FIX: Aggiunto worst_index per accedere correttamente al punto peggiore
    Scalar reflect_point(Callback const & callback, size_t worst_index) {
      m_trial_point = m_centroid + m_options.rho * (m_centroid - m_simplex[worst_index]);
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
    // BUG FIX: Aggiunto worst_index per accedere correttamente al punto peggiore
    Scalar contract_point(Callback const & callback, size_t worst_index, bool outside = true) {
      if (outside) {
        m_trial_point = m_centroid + m_options.gamma * (m_trial_point - m_centroid);
      } else {
        m_trial_point = m_centroid - m_options.gamma * (m_centroid - m_simplex[worst_index]);
      }
      project_point(m_trial_point);
      return callback(m_trial_point);
    }

    /**
     * @brief Perform shrink operation
     */
    // BUG FIX: Aggiunto best_index per accedere correttamente al punto migliore
    void shrink_simplex(Callback const & callback, size_t best_index) {
      Vector best = m_simplex[best_index];
      // Si restringono tutti i punti TRANNE il punto migliore
      for (size_t i = 0; i < m_dim + 1; ++i) {
        if (i != best_index) {
          m_simplex[i] = best + m_options.sigma * (m_simplex[i] - best);
          project_point(m_simplex[i]);
          m_values[i] = callback(m_simplex[i]);
        }
      }
    }

    /**
     * @brief Select subspace for Random Subspace Method
     */
    std::vector<size_t> select_subspace() {
      std::vector<size_t> all_indices(m_dim);
      std::iota(all_indices.begin(), all_indices.end(), 0);
      
      if (m_options.adaptive_subspace_size) {
        m_subspace_size = static_cast<size_t>(
          m_options.subspace_min_size + 
          (m_options.subspace_max_size - m_options.subspace_min_size) *
          std::exp(-static_cast<Scalar>(m_current_iteration) / 100.0)
        );
        m_subspace_size = std::max(m_options.subspace_min_size, 
                                 std::min(m_options.subspace_max_size, m_subspace_size));
      } else {
        m_subspace_size = m_options.subspace_max_size;
      }

      // Gestisce il caso in cui la dimensione del sottospazio è maggiore della dimensione del problema
      m_subspace_size = std::min(m_subspace_size, m_dim);

      std::shuffle(all_indices.begin(), all_indices.end(), m_rng);
      return std::vector<size_t>(all_indices.begin(), 
                               all_indices.begin() + m_subspace_size);
    }

    /**
     * @brief Create blocks for Block Coordinate Descent
     */
    void initialize_blocks() {
      m_blocks.clear();
      size_t num_blocks = (m_dim + m_options.block_size - 1) / m_options.block_size;
      
      for (size_t i = 0; i < num_blocks; ++i) {
        std::vector<size_t> block;
        size_t start = i * m_options.block_size;
        size_t end = std::min((i + 1) * m_options.block_size, m_dim);
        
        for (size_t j = start; j < end; ++j) {
          block.push_back(j);
        }
        m_blocks.push_back(block);
      }
    }

    /**
     * @brief Optimize in selected subspace
     */
    Result optimize_subspace(const std::vector<size_t>& subspace, 
                           Vector const & full_x, 
                           Callback const & callback) {
      size_t subspace_dim = subspace.size();
      
      auto subspace_callback = [&](Vector const & subspace_x) -> Scalar {
        Vector full_point = full_x;
        for (size_t i = 0; i < subspace_dim; ++i) {
          full_point(subspace[i]) = subspace_x(i);
        }
        return callback(full_point);
      };

      Vector subspace_x0(subspace_dim);
      for (size_t i = 0; i < subspace_dim; ++i) {
        subspace_x0(i) = full_x(subspace[i]);
      }

      Vector subspace_lower, subspace_upper;
      if (m_use_bounds) {
        subspace_lower.resize(subspace_dim);
        subspace_upper.resize(subspace_dim);
        for (size_t i = 0; i < subspace_dim; ++i) {
          subspace_lower(i) = m_lower(subspace[i]);
          subspace_upper(i) = m_upper(subspace[i]);
        }
      }

      Options subspace_opts = m_options;
      subspace_opts.max_iterations = m_options.subspace_iterations;
      subspace_opts.tolerance = m_options.subspace_tolerance;
      
      // IMPORTANTE: Usa la strategia STANDARD all'interno del sottospazio 
      subspace_opts.strategy = Strategy::STANDARD;
      // Disattiva il verbose per il sottoproblema, a meno che non sia specificato un monitor più granulare
      subspace_opts.verbose = m_options.monitor_subspaces;

      NelderMead_minimizer<Scalar> subspace_optimizer(subspace_opts);
      if (m_use_bounds) {
        subspace_optimizer.set_bounds(subspace_lower, subspace_upper);
      }

      return subspace_optimizer.minimize(subspace_x0, subspace_callback);
    }

    /**
     * @brief Standard Nelder-Mead optimization
     */
    Result minimize_standard(Vector const & x0, Callback const & callback) {
      Result result;
      result.initial_function_value = callback(x0);

      m_dim = x0.size(); // Re-imposta m_dim se minimize_standard è chiamato da minimize_subspace
      
      // Gestisce il caso banale: se la dimensione del problema è 0
      if (m_dim == 0) {
        result.status = Status::CONVERGED;
        result.solution = x0;
        result.final_function_value = result.initial_function_value;
        return result;
      }


      initialize_simplex(x0, callback);
      result.function_evaluations = m_dim + 1;

      if (m_options.adaptive_parameters && m_dim > 0) {
        m_options.rho = 1.0;
        // Utilizza m_dim come numero di dimensioni per i parametri adattivi
        Scalar n = static_cast<Scalar>(m_dim);
        m_options.chi = 1.0 + 2.0 / n;
        m_options.gamma = 0.75 - 1.0 / (2.0 * n);
        m_options.sigma = 1.0 - 1.0 / n;
        if (m_options.verbose) {
          fmt::print("[NM] Adaptive parameters set (rho={:.2f}, chi={:.2f}, gamma={:.2f}, sigma={:.2f})\n",
                    m_options.rho, m_options.chi, m_options.gamma, m_options.sigma);
        }
      }
      
      if (m_options.verbose) {
        fmt::print("[NM] Starting Standard Nelder-Mead (N={})\n", m_dim);
      }

      for (result.iterations = 0; 
           result.iterations < m_options.max_iterations; 
           ++result.iterations) {

        if (result.function_evaluations >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        if (nelder_mead_iteration(callback, result.function_evaluations)) {
          result.status = Status::CONVERGED;
          break;
        }

        result.simplex_diameter = compute_diameter();
        if (result.simplex_diameter < m_options.simplex_tolerance) {
          result.status = Status::SIMPLEX_TOO_SMALL;
          break;
        }
        
        if (m_options.verbose && result.iterations % m_options.progress_frequency == 0) {
          auto indices = sort_vertices();
          Scalar progress = 100.0 * result.iterations / m_options.max_iterations;
          fmt::print("[NM] Progress: {:5.1f}% | Iter={:4d} | F_best={:12.6g} | Diameter={:e} | Evals={}\n",
                      progress, result.iterations, m_values[indices[0]], result.simplex_diameter,
                      result.function_evaluations);
        }
      }

      auto indices = sort_vertices();
      result.solution = m_simplex[indices[0]];
      result.final_function_value = m_values[indices[0]];
      result.simplex_volume = compute_volume();
      result.simplex_diameter = compute_diameter();
      
      if (result.iterations >= m_options.max_iterations && result.status == Status::FAILED) {
        result.status = Status::MAX_ITERATIONS;
      }
      
      if (m_options.verbose) {
        fmt::print("\n--- Standard Nelder-Mead Optimization Results ---\n");
        fmt::print("[INFO] Status: {}\n", status_to_string(result.status));
        fmt::print("[INFO] Final F: {:12.6g} (Initial: {:12.6g})\n",
                    result.final_function_value, result.initial_function_value);
        fmt::print("[INFO] Iterations: {} | Function Evals: {}\n",
                    result.iterations, result.function_evaluations);
        fmt::print("[INFO] Simplex Diameter: {:e} | Volume: {:e}\n",
                    result.simplex_diameter, result.simplex_volume);
        fmt::print("---------------------------------------------------\n");
      }


      return result;
    }

    /**
     * @brief Random Subspace Method optimization
     */
    Result minimize_random_subspace(Vector const & x0, Callback const & callback) {
      Result result;
      // Il minimizer esterno (RSM) deve usare la dimensione completa
      m_dim = x0.size(); 
      
      result.initial_function_value = callback(x0);
      result.function_evaluations = 1;
      
      m_current_solution = x0;
      Scalar best_value = result.initial_function_value;

      std::random_device rd;
      m_rng.seed(rd());
      
      if (m_options.verbose) {
        fmt::print("[RSM] Starting Random Subspace Method (N={}, Max Subspace Size={})\n", 
                    m_dim, m_options.subspace_max_size);
      }

      for (result.iterations = 0; 
           result.iterations < m_options.max_iterations; 
           ++result.iterations) {
        
        m_current_iteration = result.iterations; // Aggiorna per l'adaptive size

        if (result.function_evaluations >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        auto subspace = select_subspace();
        result.active_subspace = subspace;

        // Salva la soluzione corrente prima di ottimizzare il sottospazio
        Vector solution_before_subspace = m_current_solution;

        auto subspace_result = optimize_subspace(subspace, m_current_solution, callback);
        result.function_evaluations += subspace_result.function_evaluations;
        result.subspace_iterations += subspace_result.iterations;

        // Aggiorna la soluzione completa con i risultati del sottospazio
        for (size_t i = 0; i < subspace.size(); ++i) {
          m_current_solution(subspace[i]) = subspace_result.solution(i);
        }

        // Riesegui il callback sulla soluzione completa (necessario per aggiornare best_value)
        Scalar new_value = callback(m_current_solution);
        result.function_evaluations++;

        // Criterio di convergenza basato sulla differenza assoluta del valore della funzione
        if (std::abs(best_value - new_value) < m_options.tolerance) {
          result.status = Status::SUBSPACE_CONVERGED;
          break;
        }

        best_value = new_value;

        if (m_options.verbose && result.iterations % m_options.progress_frequency == 0) {
          Scalar progress = 100.0 * result.iterations / m_options.max_iterations;
          fmt::print("[RSM] Progress: {:5.1f}% | Iter={:4d} | F_best={:12.6g} | Subspace={}/{} | Evals={}\n",
                      progress, result.iterations, best_value, subspace.size(), m_dim,
                      result.function_evaluations);
        }
      }
      
      if (result.iterations >= m_options.max_iterations && result.status == Status::FAILED) {
        result.status = Status::MAX_ITERATIONS;
      }

      result.solution = m_current_solution;
      result.final_function_value = best_value;
      
      if (m_options.verbose) {
        fmt::print("\n--- RSM Optimization Results ---\n");
        fmt::print("[INFO] Status: {}\n", status_to_string(result.status));
        fmt::print("[INFO] Final F: {:12.6g} (Initial: {:12.6g})\n",
                    result.final_function_value, result.initial_function_value);
        fmt::print("[INFO] Total Iterations: {} | Total Function Evals: {}\n",
                    result.iterations, result.function_evaluations);
        fmt::print("[INFO] Subspace Iterations: {}\n", result.subspace_iterations);
        fmt::print("--------------------------------\n");
      }
      return result;
    }

    /**
     * @brief Block Coordinate Descent optimization
     */
    Result minimize_block_coordinate(Vector const & x0, Callback const & callback) {
      Result result;
      // Il minimizer esterno (BCD) deve usare la dimensione completa
      m_dim = x0.size();
      
      result.initial_function_value = callback(x0);
      result.function_evaluations = 1;
      
      m_current_solution = x0;
      Scalar best_value = result.initial_function_value;

      initialize_blocks();
      
      if (m_options.verbose) {
        fmt::print("[BCD] Starting Block Coordinate Descent (N={}, Block Size={}, Num Blocks={})\n", 
                    m_dim, m_options.block_size, m_blocks.size());
      }

      for (result.iterations = 0; 
           result.iterations < m_options.max_iterations; 
           ++result.iterations) {

        if (result.function_evaluations >= m_options.max_function_evaluations) {
          result.status = Status::MAX_FUN_EVALUATIONS;
          break;
        }

        std::vector<size_t> block_indices(m_blocks.size());
        std::iota(block_indices.begin(), block_indices.end(), 0);
        
        if (m_options.random_block_order) {
          std::shuffle(block_indices.begin(), block_indices.end(), m_rng);
        }

        size_t blocks_to_optimize = std::min(m_options.blocks_per_iteration, 
                                           m_blocks.size());

        for (size_t i = 0; i < blocks_to_optimize; ++i) {
          auto& block = m_blocks[block_indices[i]];
          
          auto block_result = optimize_subspace(block, m_current_solution, callback);
          result.function_evaluations += block_result.function_evaluations;
          result.subspace_iterations += block_result.iterations;

          for (size_t j = 0; j < block.size(); ++j) {
            m_current_solution(block[j]) = block_result.solution(j);
          }
        }

        Scalar new_value = callback(m_current_solution);
        result.function_evaluations++;

        if (std::abs(best_value - new_value) < m_options.tolerance) {
          result.status = Status::SUBSPACE_CONVERGED;
          break;
        }

        best_value = new_value;

        if (m_options.verbose && result.iterations % m_options.progress_frequency == 0) {
          Scalar progress = 100.0 * result.iterations / m_options.max_iterations;
          fmt::print("[BCD] Progress: {:5.1f}% | Iter={:4d} | F_best={:12.6g} | Blocks={}/{} | Evals={}\n",
                      progress, result.iterations, best_value, blocks_to_optimize, m_blocks.size(),
                      result.function_evaluations);
        }
      }
      
      if (result.iterations >= m_options.max_iterations && result.status == Status::FAILED) {
        result.status = Status::MAX_ITERATIONS;
      }


      result.solution = m_current_solution;
      result.final_function_value = best_value;
      
      if (m_options.verbose) {
        fmt::print("\n--- BCD Optimization Results ---\n");
        fmt::print("[INFO] Status: {}\n", status_to_string(result.status));
        fmt::print("[INFO] Final F: {:12.6g} (Initial: {:12.6g})\n",
                    result.final_function_value, result.initial_function_value);
        fmt::print("[INFO] Total Iterations: {} | Total Function Evals: {}\n",
                    result.iterations, result.function_evaluations);
        fmt::print("[INFO] Subspace Iterations: {}\n", result.subspace_iterations);
        fmt::print("--------------------------------\n");
      }
      return result;
    }

  public:
    explicit NelderMead_minimizer(Options const & opts = Options())
    : m_options(opts) {
      std::random_device rd;
      m_rng.seed(rd());
    }

    Vector const & solution() const { return m_current_solution; }

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

    void set_bounds(size_t n, Scalar const lower[], Scalar const upper[]) {
      m_lower.resize(n);
      m_upper.resize(n);
      std::copy_n(lower, n, m_lower.data());
      std::copy_n(upper, n, m_upper.data());
      m_use_bounds = true;
    }

    Result minimize(Vector const & x0, Callback const & callback) {
      m_dim = x0.size();
      m_current_solution = x0;
      m_current_iteration = 0;
      
      // Gestisce il caso banale prima di selezionare la strategia
      if (m_dim == 0) {
        Result result;
        result.status = Status::CONVERGED;
        result.solution = x0;
        result.initial_function_value = callback(x0);
        result.final_function_value = result.initial_function_value;
        result.function_evaluations = 1;
        return result;
      }

      Strategy selected_strategy = m_options.strategy;
      if (selected_strategy == Strategy::ADAPTIVE_SUBSPACE) {
        if (m_dim <= m_options.max_dimension_standard) {
          selected_strategy = Strategy::STANDARD;
        } else if (m_dim <= 100) {
          selected_strategy = Strategy::BLOCK_COORDINATE;
        } else {
          selected_strategy = Strategy::RANDOM_SUBSPACE;
        }
      }

      if (m_options.verbose) {
        fmt::print("--- Nelder-Mead Optimization Start ---\n");
        fmt::print("[INFO] Problem dimension (N): {}\n", m_dim);
        fmt::print("[INFO] Selected strategy: {} (Adaptive: {})\n", 
                    strategy_to_string(selected_strategy), 
                    m_options.strategy == Strategy::ADAPTIVE_SUBSPACE ? "Yes" : "No");
        fmt::print("[INFO] Max Iterations: {} | Max Evals: {}\n",
                    m_options.max_iterations, m_options.max_function_evaluations);
        fmt::print("[INFO] Tolerance (F): {:e} | Tolerance (Simplex): {:e}\n",
                    m_options.tolerance, m_options.simplex_tolerance);
        fmt::print("--------------------------------------\n");
      }

      switch (selected_strategy) {
        case Strategy::STANDARD:
          return minimize_standard(x0, callback);
        case Strategy::RANDOM_SUBSPACE:
          return minimize_random_subspace(x0, callback);
        case Strategy::BLOCK_COORDINATE:
          return minimize_block_coordinate(x0, callback);
        case Strategy::MIXED_STRATEGY: {
          // Salvataggio/ripristino dello stato prima e dopo la modifica delle opzioni
          Options original_options = m_options; 

          auto result = minimize_random_subspace(x0, callback);
          if (result.status == Status::SUBSPACE_CONVERGED) {
            if (m_options.verbose) {
              fmt::print("[MIXED] RSM converged (Subspace). Starting BCD refinement...\n");
            }
            // Modifica locale per il sottoproblema
            m_options.strategy = Strategy::BLOCK_COORDINATE;
            // La soluzione RSM diventa il punto di partenza per BCD
            auto bcd_result = minimize_block_coordinate(result.solution, callback);
            // Consolidare i conteggi totali di iterazioni e chiamate funzione
            bcd_result.iterations += result.iterations;
            bcd_result.function_evaluations += result.function_evaluations;
            // Ripristino delle opzioni
            m_options = original_options; 
            return bcd_result;
          }
          // Ripristino delle opzioni nel caso in cui RSM non sia convergente
          m_options = original_options; 
          return result;
        }
        case Strategy::ADAPTIVE_SUBSPACE: // Se la strategia era ADAPTIVE ma non è stata mappata sopra
        default:
          // Dovrebbe essere coperto sopra, ma in caso di errore logico
          if (m_options.verbose) {
            fmt::print("[ERROR] Unhandled or adaptive strategy fallback to standard.\n");
          }
          return minimize_standard(x0, callback);
      }

      return Result{};
    }

    static std::string strategy_to_string(Strategy strategy) {
      switch (strategy) {
        case Strategy::STANDARD:          return "STANDARD";
        case Strategy::RANDOM_SUBSPACE:   return "RANDOM_SUBSPACE (RSM)";
        case Strategy::BLOCK_COORDINATE:  return "BLOCK_COORDINATE (BCD)";
        case Strategy::ADAPTIVE_SUBSPACE: return "ADAPTIVE_SUBSPACE (Auto-select)";
        case Strategy::MIXED_STRATEGY:    return "MIXED_STRATEGY (RSM -> BCD)";
        default:                          return "UNKNOWN";
      }
    }

    static std::string status_to_string(Status status) {
      switch (status) {
        case Status::CONVERGED:           return "CONVERGED (Tolerance Met)";
        case Status::MAX_ITERATIONS:      return "MAX_ITERATIONS (Limit Reached)";
        case Status::MAX_FUN_EVALUATIONS: return "MAX_FUN_EVALUATIONS (Limit Reached)";
        case Status::SIMPLEX_TOO_SMALL:   return "SIMPLEX_TOO_SMALL (Degenerate Simplex)";
        case Status::SUBSPACE_CONVERGED:  return "SUBSPACE_CONVERGED (High-Dim Convergence)";
        case Status::FAILED:              return "FAILED (Unknown Error)";
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
