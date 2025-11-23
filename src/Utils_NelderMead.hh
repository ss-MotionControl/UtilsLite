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
    CONVERGED,              // Tolleranza sulla funzione soddisfatta
    CONVERGED_TOL_X,        // Tolleranza sulla dimensione del simplesso
    MAX_ITERATIONS,         // Max iterazioni raggiunte
    MAX_FUN_EVALUATIONS,    // Max valutazioni raggiunte
    FAIL_NAN,               // Rilevato NaN durante il calcolo
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

    // Statistiche
    size_t outer_iterations{0};
    size_t inner_iterations{0};
    size_t total_iterations{0};

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

      // Parametri standard Nelder-Mead
      const Scalar rho = 1.0;  // Reflection
      const Scalar chi = 2.0;  // Expansion
      const Scalar gam = 0.5;  // Contraction
      const Scalar sig = 0.5;  // Shrink

      // Allocazione Simplesso
      std::vector<Vector> simplex(dim + 1);
      std::vector<Scalar> f_values(dim + 1);
      std::vector<size_t> idx(dim + 1); 

      // 1. Costruzione Simplesso
      simplex[0] = x0;
      f_values[0] = safe_evaluate(simplex[0], callback);

      for (size_t i = 0; i < dim; ++i) {
        Vector next = x0;
        // euristica passo iniziale: evita passo nullo su x=0
        Scalar step = (std::abs(x0(i)) < 1e-6) ? 0.00025 : m_options.initial_step * x0(i);
        next(i) += step;
        simplex[i+1] = next;
        f_values[i+1] = safe_evaluate(next, callback);
      }

      // Indici per ordinamento indiretto
      std::iota(idx.begin(), idx.end(), 0);
      auto sort_simplex = [&]() {
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { 
            // Gestione robusta NaN/Inf in ordinamento: mettili in fondo
            if (std::isnan(f_values[a])) return false;
            if (std::isnan(f_values[b])) return true;
            return f_values[a] < f_values[b]; 
        });
      };
      sort_simplex();

      res.initial_function_value = f_values[idx[0]];
      size_t iter = 0;

      for (; iter < m_options.max_iterations; ++iter) {
        // Check budget
        if (m_evals_count >= m_options.max_function_evaluations) {
          res.status = NelderMeadStatus::MAX_FUN_EVALUATIONS;
          break;
        }

        // Check Convergenza (Differenza valori funzione)
        Scalar f_best  = f_values[idx[0]];
        Scalar f_worst = f_values[idx.back()];
        if (std::abs(f_worst - f_best) < m_options.tolerance) {
          res.status = NelderMeadStatus::CONVERGED;
          break;
        }

        // Centroide (escluso peggiore)
        Vector centroid = Vector::Zero(dim);
        for (size_t i = 0; i < dim; ++i) centroid += simplex[idx[i]];
        centroid /= static_cast<Scalar>(dim);

        // --- REFLECTION ---
        Vector xr = centroid + rho * (centroid - simplex[idx.back()]);
        Scalar fr = safe_evaluate(xr, callback);

        if (fr < f_values[idx[0]]) {
          // --- EXPANSION ---
          Vector xe = centroid + chi * (xr - centroid);
          Scalar fe = safe_evaluate(xe, callback);
          if (fe < fr) { 
            simplex[idx.back()] = xe; f_values[idx.back()] = fe; 
          } else { 
            simplex[idx.back()] = xr; f_values[idx.back()] = fr; 
          }
        } else if (fr < f_values[idx[dim - 1]]) {
          // --- ACCEPT REFLECTION ---
          simplex[idx.back()] = xr; f_values[idx.back()] = fr;
        } else {
          // --- CONTRACTION ---
          // Qui siamo peggio del second-worst.
          bool contraction_ok = false;

          if (fr < f_values[idx.back()]) {
            // Outside Contraction (meglio del worst, ma peggio del second-worst)
            Vector xc = centroid + gam * (xr - centroid);
            Scalar fc = safe_evaluate(xc, callback);
            if (fc <= fr) {
              simplex[idx.back()] = xc; f_values[idx.back()] = fc;
              contraction_ok = true;
            }
          } else {
            // Inside Contraction (peggio anche del worst)
            Vector xc = centroid + gam * (simplex[idx.back()] - centroid);
            Scalar fc = safe_evaluate(xc, callback);
            if (fc < f_values[idx.back()]) {
              simplex[idx.back()] = xc; f_values[idx.back()] = fc;
              contraction_ok = true;
            }
          }

          if (!contraction_ok) {
            // --- SHRINK ---
            Vector x_best = simplex[idx[0]];
            for (size_t i = 1; i <= dim; ++i) {
               size_t k = idx[i]; // Usa indice reale
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

    enum class BlockStrategy { SEQUENTIAL, RANDOM, ADAPTIVE };

    struct Options {
      size_t block_size{10};
      size_t max_outer_iterations{100};
      size_t max_function_evaluations{100000};
      Scalar tolerance{1e-6};
      bool   verbose{true};
      BlockStrategy strategy{BlockStrategy::ADAPTIVE};
      typename NelderMead_classic<Scalar>::Options sub_options;
    };

  private:
    Options m_options;
    NelderMead_classic<Scalar> m_solver;
    Vector m_lower, m_upper;
    bool m_use_bounds{false};

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

      // --- SHORTCUT PER DIMENSIONI PICCOLE (< 10) ---
      // Se la dimensione è piccola o il blocco copre tutto, usa direttamente il classico
      if (n < 10 || n <= m_options.block_size) {
          if (m_options.verbose) {
              fmt::print("   [Info] Dim < 10 o Dim <= BlockSize. Switching to DIRECT CLASSIC solver.\n");
          }
          // Configura il solver interno con budget GLOBALE
          auto full_opts = m_options.sub_options;
          full_opts.max_function_evaluations = m_options.max_function_evaluations;
          full_opts.max_iterations = m_options.max_function_evaluations; // Rimuoviamo limiti artificiali di iter
          full_opts.tolerance = m_options.tolerance;
          full_opts.verbose = false; // Evita spam interno
          
          m_solver.set_options(full_opts);
          if (m_use_bounds) m_solver.set_bounds(m_lower, m_upper);

          // Esegui diretto
          return m_solver.minimize(x, global_callback);
      }
      
      // --- LOGICA A BLOCCHI (per Dim >= 10) ---
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

      if (m_options.verbose) {
         fmt::print("   {:<5} | {:<10} | {:<14} | {:<10} | {:<10}\n", 
                    "Iter", "BlockDim", "F Value", "In.Iter", "In.Eval");
         fmt::print("   {:-<65}\n", "");
      }

      while (outer_iter < m_options.max_outer_iterations && !converged) {
        outer_iter++;
        if (count_outer_evals + count_inner_evals >= m_options.max_function_evaluations) {
          res.status = NelderMeadStatus::MAX_FUN_EVALUATIONS;
          break;
        }

        vector<size_t> idxs = select_block(n, outer_iter - 1);
        size_t k = idxs.size();

        // Setup Subspace
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

        auto sub_res = m_solver.minimize(x_sub, sub_cb);
        count_inner_iters += sub_res.inner_iterations;

        Scalar improvement = current_f - sub_res.final_function_value;
        if (sub_res.final_function_value < current_f) {
           current_f = sub_res.final_function_value;
           for (size_t i = 0; i < k; ++i) x(idxs[i]) = sub_res.solution(i);
        }

        if (m_options.verbose) {
           fmt::print("   {:<5} | {:<10} | {:<14.6e} | {:<10} | {:<10}\n", 
                      outer_iter, k, current_f, sub_res.inner_iterations, sub_res.inner_evaluations);
        }

        if (improvement < m_options.tolerance && improvement >= 0) {
            // Per ora non facciamo early exit aggressivo nel block coordinate
            // per dare chance ad altri blocchi
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
          res.status = (outer_iter >= m_options.max_outer_iterations) ? 
                       NelderMeadStatus::MAX_ITERATIONS : NelderMeadStatus::CONVERGED;
      }
      return res;
    }
  };
}

#endif
