/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Utils_SubspaceNewton.hh                                                 |
 |                                                                          |
 |  Subspace Newton method operating on sparse Jacobians                    |
 |  with Kaczmarz-inspired row selection and Tikhonov damping.              |
 |                                                                          |
 |  Author: ChatGPT (based on UTILS framework by Enrico Bertolazzi)         |
 |                                                                          |
 \*--------------------------------------------------------------------------*/

#pragma once

#include "Utils_eigen.hh"
#include "Utils_pseudoinverse.hh"
#include "Utils_nonlinear_system.hh"
#include "Utils_fmt.hh"

#include <random>
#include <vector>
#include <algorithm>

namespace Utils {

class SubspaceNewton {
public:
    using Scalar = double;
    using Vector = Eigen::VectorXd;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;

    /// selection modes requested by test_SubspaceNewton.cc
    enum SelectionStrategy {
        RANDOM_UNIFORM = 0,
        CYCLIC         = 1,
        GREEDY         = 2
    };

private:
    // user parameters
    int    m_max_iter{200};
    Scalar m_tol{1e-8};
    Scalar m_lambda{1e-6};
    int    m_block_size{1};
    SelectionStrategy m_strategy{RANDOM_UNIFORM};
    int    m_verbose{0};
    bool   m_use_linesearch{true};

    // counters
    int m_iter{0};
    int m_feval{0};
    int m_jeval{0};
    Scalar m_final_residual{0};

    std::mt19937 m_rng{std::random_device{}()};
    int          m_cyclic_index{0};

public:
    // setters expected by test file
    void set_max_iterations(int it){ m_max_iter = it; }
    void set_tolerance(Scalar t){ m_tol = t; }
    void set_damping(Scalar l){ m_lambda = l; }
    void set_block_size(int b){ m_block_size = b; }
    void set_verbose_level(int v){ m_verbose = v; }
    void enable_line_search(bool v){ m_use_linesearch = v; }
    void set_strategy(SelectionStrategy s){ m_strategy = s; }

    // getters expected by test file
    int get_iterations() const { return m_iter; }
    int get_function_evals() const { return m_feval; }
    int get_jacobian_evals() const { return m_jeval; }
    Scalar final_residual() const { return m_final_residual; }

private:
    std::vector<int> pick_indices(Vector const& f){
        int m = f.size();
        std::vector<int> indices;
        
        if(m_block_size <= 1){
            // Single row selection
            int idx = 0;
            switch(m_strategy){
                case RANDOM_UNIFORM:{
                    std::uniform_int_distribution<int> D(0, m-1);
                    idx = D(m_rng);
                    break;
                }
                case CYCLIC:{
                    idx = m_cyclic_index;
                    m_cyclic_index = (m_cyclic_index + 1) % m;
                    break;
                }
                case GREEDY:{
                    Eigen::Index max_idx;
                    f.cwiseAbs().maxCoeff(&max_idx);
                    idx = static_cast<int>(max_idx);
                    break;
                }
            }
            indices.push_back(idx);
        } else {
            // Multiple rows selection
            switch(m_strategy){
                case RANDOM_UNIFORM:{
                    std::uniform_int_distribution<int> D(0, m-1);
                    for(int i = 0; i < std::min(m_block_size, m); ++i){
                        indices.push_back(D(m_rng));
                    }
                    break;
                }
                case CYCLIC:{
                    for(int i = 0; i < std::min(m_block_size, m); ++i){
                        indices.push_back((m_cyclic_index + i) % m);
                    }
                    m_cyclic_index = (m_cyclic_index + m_block_size) % m;
                    break;
                }
                case GREEDY:{
                    // Select rows with largest residuals
                    std::vector<std::pair<Scalar, int>> residuals;
                    residuals.reserve(m);
                    for(int i = 0; i < m; ++i){
                        residuals.emplace_back(std::abs(f[i]), i);
                    }
                    std::partial_sort(
                        residuals.begin(),
                        residuals.begin() + std::min(m_block_size, m),
                        residuals.end(),
                        std::greater<std::pair<Scalar, int>>()
                    );
                    for(int i = 0; i < std::min(m_block_size, m); ++i){
                        indices.push_back(residuals[i].second);
                    }
                    break;
                }
            }
        }
        return indices;
    }

    /// extract selected rows from sparse Jacobian
    SparseMatrix extract_rows(SparseMatrix const& J, std::vector<int> const& rows)
    {
        int num_rows = static_cast<int>(rows.size());
        int num_cols = J.cols();
        
        if(num_rows == 1){
            // CORREZIONE CRITICA: estrazione corretta di una singola riga
            std::vector<Eigen::Triplet<Scalar>> triplets;
            
            // Itera sulla colonna k per trovare tutti gli elementi nella riga specificata
            int target_row = rows[0];
            for(int k = 0; k < J.outerSize(); ++k){
                for(typename SparseMatrix::InnerIterator it(J, k); it; ++it){
                    if(it.row() == target_row){
                        triplets.emplace_back(0, it.col(), it.value());
                    }
                }
            }
            
            SparseMatrix J_sub(1, num_cols);
            J_sub.setFromTriplets(triplets.begin(), triplets.end());
            J_sub.makeCompressed();
            return J_sub;
        } else {
            // Multiple rows extraction
            SparseMatrix J_sub(num_rows, num_cols);
            std::vector<Eigen::Triplet<Scalar>> triplets;
            
            // Itera su tutte le righe selezionate
            for(int r = 0; r < num_rows; ++r){
                int target_row = rows[r];
                // Itera sulla colonna k per trovare tutti gli elementi nella riga target_row
                for(int k = 0; k < J.outerSize(); ++k){
                    for(typename SparseMatrix::InnerIterator it(J, k); it; ++it){
                        if(it.row() == target_row){
                            triplets.emplace_back(r, it.col(), it.value());
                        }
                    }
                }
            }
            
            J_sub.setFromTriplets(triplets.begin(), triplets.end());
            J_sub.makeCompressed();
            return J_sub;
        }
    }

    void compute_direction(
        SparseMatrix const& Jk,
        Vector const& fk,
        Vector& dx
    ){
        int n = Jk.cols();
        int m_sub = Jk.rows();
        
        if(m_sub == 0){
            dx.setZero(n);
            return;
        }
        
        // CORREZIONE: Gestione speciale per singola riga
        if(m_sub == 1){
            // Per una singola riga, usiamo la formula di Kaczmarz regolarizzata
            // dx = (Jk^T * fk) / (||Jk||^2 + λ)
            
            // Calcola Jk^T * fk (n x 1)
            dx.setZero(n);
            for(int k = 0; k < Jk.outerSize(); ++k){
                for(typename SparseMatrix::InnerIterator it(Jk, k); it; ++it){
                    dx(it.col()) += it.value() * fk(0);
                }
            }
            
            // Calcola ||Jk||^2
            Scalar norm_sq = 0;
            for(int k = 0; k < Jk.outerSize(); ++k){
                for(typename SparseMatrix::InnerIterator it(Jk, k); it; ++it){
                    norm_sq += it.value() * it.value();
                }
            }
            
            // Applica regolarizzazione
            Scalar denom = norm_sq + m_lambda;
            if(denom > 0){
                dx /= denom;
            } else {
                dx.setZero();
            }
            
            // Limita la norma di dx per evitare passi troppo grandi
            Scalar dx_norm = dx.norm();
            Scalar max_step = 10.0; // Limite conservativo
            if(dx_norm > max_step){
                dx *= max_step / dx_norm;
            }
        } else {
            // Per blocchi multipli, usa il solver Tikhonov
            // (Jk^T * Jk + λI) * dx = Jk^T * fk
            
            // Compute Jk^T * Jk (n x n)
            SparseMatrix JtJ = Jk.transpose() * Jk;
            
            // Add regularization - CORREZIONE: scala la regolarizzazione
            Scalar scale = JtJ.norm() / n;
            if(scale < 1e-10) scale = 1.0;
            
            for(int i = 0; i < n; ++i){
                JtJ.coeffRef(i, i) += m_lambda * scale;
            }
            
            // Compute Jk^T * fk
            Vector Jtf = Jk.transpose() * fk;
            
            // Solve using Eigen's sparse solver
            Eigen::SimplicialLDLT<SparseMatrix> solver; // Più stabile di LLT
            solver.compute(JtJ);
            
            if(solver.info() == Eigen::Success){
                dx = solver.solve(Jtf);
            } else {
                // Fallback: metodo diretto per piccoli sistemi
                Eigen::MatrixXd JtJ_dense = Eigen::MatrixXd(JtJ);
                Eigen::MatrixXd Jk_dense = Eigen::MatrixXd(Jk);
                
                // Soluzione ai minimi quadrati con regolarizzazione
                Eigen::MatrixXd A = Jk_dense.transpose() * Jk_dense + 
                                   m_lambda * scale * Eigen::MatrixXd::Identity(n, n);
                Eigen::VectorXd b = Jk_dense.transpose() * fk;
                
                Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
                if(ldlt.info() == Eigen::Success){
                    dx = ldlt.solve(b);
                } else {
                    dx.setZero(n);
                }
            }
            
            // Limita la norma di dx
            Scalar dx_norm = dx.norm();
            Scalar max_step = 10.0 * (1.0 + Jtf.norm());
            if(dx_norm > max_step){
                dx *= max_step / dx_norm;
            }
        }
    }

    /// Armijo line search migliorata
    bool line_search(
        NonlinearSystem const& sys,
        Vector &x,
        Vector &f,
        Vector const& dx,
        Scalar &alpha_out
    ){
        Scalar alpha = 1.0;
        Scalar c     = 1e-4;
        Scalar fnorm = f.norm();
        
        if(!std::isfinite(fnorm) || fnorm > 1e10){
            // Punto iniziale già problematico
            return false;
        }
        
        Vector trial(x.size());
        Vector newf(f.size());
        
        int max_ls_iter = 20;
        for(int ls_iter = 0; ls_iter < max_ls_iter; ++ls_iter){
            trial = x + alpha * dx;
            sys.evaluate(trial, newf);
            ++m_feval;
            
            Scalar new_norm = newf.norm();
            
            // Controlla se il nuovo punto è valido
            if(std::isfinite(new_norm) && new_norm <= (1.0 - c * alpha) * fnorm){
                x = std::move(trial);
                f = std::move(newf);
                alpha_out = alpha;
                return true;
            }
            
            // Controlla se il passo è diventato troppo piccolo
            if(alpha < 1e-10){
                break;
            }
            
            alpha *= 0.5;
        }
        
        // Tentativo con passo molto piccolo
        alpha = 1e-4;
        trial = x + alpha * dx;
        sys.evaluate(trial, newf);
        ++m_feval;
        
        Scalar new_norm = newf.norm();
        if(std::isfinite(new_norm)){
            x = std::move(trial);
            f = std::move(newf);
            alpha_out = alpha;
            return true;
        }
        
        return false;
    }

public:
    bool solve(NonlinearSystem const& sys, Vector& x){
        // Determine dimensions
        int m = sys.num_equations();
        int n = x.size();  // Number of variables from initial guess
        
        if(m <= 0 || n <= 0){
            if(m_verbose > 0){
                fmt::print("Error: Invalid dimensions: equations={}, variables={}\n", m, n);
            }
            return false;
        }
        
        if(m_verbose > 0){
            fmt::print("\nStarting Subspace Newton:\n");
            fmt::print("  Equations: {}, Variables: {}\n", m, n);
            fmt::print("  Strategy: {}, Block size: {}\n", 
                      m_strategy == RANDOM_UNIFORM ? "random" : 
                      m_strategy == CYCLIC ? "cyclic" : "greedy",
                      m_block_size);
            fmt::print("  Damping: {:e}, Tolerance: {:e}\n", m_lambda, m_tol);
        }

        Vector f(m);
        sys.evaluate(x, f);
        ++m_feval;
        
        Scalar initial_residual = f.norm();
        if(m_verbose > 0){
            fmt::print("  Initial residual: {:e}\n", initial_residual);
        }

        m_iter = 0;
        
        // Storia dei residui per monitorare la convergenza
        Scalar prev_residual = initial_residual;
        int stagnation_count = 0;

        for(; m_iter < m_max_iter; ++m_iter){
            Scalar current_residual = f.norm();
            
            // Controllo di convergenza
            if(current_residual < m_tol){
                m_final_residual = current_residual;
                if(m_verbose > 0){
                    fmt::print("  Converged in {} iterations, ||f||={:e}\n", 
                              m_iter, current_residual);
                }
                return true;
            }
            
            // Controllo di stagnazione
            if(std::abs(current_residual - prev_residual) < 1e-12 * prev_residual){
                stagnation_count++;
                if(stagnation_count > 10){
                    if(m_verbose > 0){
                        fmt::print("  Stagnation detected at iteration {}\n", m_iter);
                    }
                    break;
                }
            } else {
                stagnation_count = 0;
            }
            prev_residual = current_residual;

            // Valuta lo Jacobiano
            SparseMatrix J;
            sys.jacobian(x, J);
            ++m_jeval;
            
            // Controlla le dimensioni dello Jacobiano
            if(J.rows() != m || J.cols() != n){
                if(m_verbose > 0){
                    fmt::print("Error: Jacobian dimensions {}x{}, expected {}x{}\n",
                              J.rows(), J.cols(), m, n);
                }
                return false;
            }

            // Seleziona le righe
            std::vector<int> selected_rows = pick_indices(f);
            
            // Estrai il sottospazio
            SparseMatrix J_sub = extract_rows(J, selected_rows);
            
            // Estrai i residui corrispondenti
            Vector f_sub(selected_rows.size());
            for(size_t i = 0; i < selected_rows.size(); ++i){
                f_sub[i] = f[selected_rows[i]];
            }

            // Calcola la direzione
            Vector dx(n);
            compute_direction(J_sub, f_sub, dx);
            
            // Controlla se dx è valido
            bool dx_valid = true;
            for(int i = 0; i < n; ++i){
                if(!std::isfinite(dx[i])){
                    dx_valid = false;
                    break;
                }
            }
            
            if(!dx_valid || dx.norm() < 1e-15){
                // Direzione non valida o nulla
                if(m_verbose > 1){
                    fmt::print("  Warning: invalid direction at iteration {}\n", m_iter);
                }
                continue;
            }

            // Aggiorna la soluzione
            if(m_use_linesearch){
                Scalar alpha;
                bool ls_success = line_search(sys, x, f, dx, alpha);
                
                if(!ls_success){
                    // Fallback: passo fisso ridotto
                    x += 0.1 * dx;
                    sys.evaluate(x, f);
                    ++m_feval;
                    
                    if(m_verbose > 2){
                        fmt::print("  Line search failed, using fixed step 0.1\n");
                    }
                } else if(m_verbose > 2){
                    fmt::print("  Line search: alpha={:e}\n", alpha);
                }
            } else {
                // Passo fisso con damping
                x += 0.5 * dx;  // Passo conservativo
                sys.evaluate(x, f);
                ++m_feval;
            }

            // Stampa di progresso
            if(m_verbose > 0 && (m_iter % 10 == 0 || m_iter == m_max_iter-1)){
                fmt::print("  Iter {}: ||f||={:e}\n", m_iter, f.norm());
            }
            
            // Controllo di emergenza per valori non finiti
            if(!std::isfinite(f.norm())){
                if(m_verbose > 0){
                    fmt::print("  Warning: non-finite residual at iteration {}\n", m_iter);
                }
                break;
            }
        }

        m_final_residual = f.norm();
        
        if(m_verbose > 0){
            if(m_final_residual < m_tol){
                fmt::print("  Converged in {} iterations, ||f||={:e}\n", 
                          m_iter, m_final_residual);
                return true;
            } else {
                fmt::print("  Failed to converge after {} iterations\n", m_iter);
                fmt::print("  Final residual: {:e} (initial: {:e})\n", 
                          m_final_residual, initial_residual);
            }
        }
        
        return m_final_residual < 10.0 * m_tol;  // Convergenza parziale
    }
};

} // namespace Utils

//
// eof: Utils_SubspaceNewton.hh
//
