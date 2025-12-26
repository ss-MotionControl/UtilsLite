/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2025                                                      |
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
// file: Utils_minimize_Newton.hh
//
/*--------------------------------------------------------------------------*\
 |  Improved Newton Minimizer with Enhanced Recovery Mechanisms             |
 |  Based on: Nocedal & Wright, "Numerical Optimization", 2nd Ed.           |
\*--------------------------------------------------------------------------*/

#pragma once

#ifndef UTILS_MINIMIZE_NEWTON_dot_HH
#define UTILS_MINIMIZE_NEWTON_dot_HH

#include "Utils_minimize.hh"

namespace Utils
{

  // ===========================================================================
  template <typename Scalar = double>
  class HessianRegularizer
  {
  public:
    using Vector       = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;

    struct Options
    {
      Scalar lambda_min{ 1e-8 };
      Scalar lambda_max{ 1e8 };
      Scalar lambda_factor{ 10.0 };
      size_t max_attempts{ 20 };
    };

  private:
    Options m_opts;
    size_t  m_modification_count{ 0 };
    Scalar  m_current_lambda{ 0 };
    Scalar  m_epsi{ std::numeric_limits<Scalar>::epsilon() };

    bool
    is_valid_descent( Vector const & p, Vector const & g ) const
    {
      if ( !p.allFinite() ) return false;

      Scalar pg = p.dot( g );
      if ( !std::isfinite( pg ) ) return false;

      return pg < -m_epsi * ( Scalar( 1 ) + std::abs( pg ) );
    }

    void
    add_diagonal_regularization( SparseMatrix & H, Scalar lambda ) const
    {
      std::vector<Eigen::Triplet<Scalar>> triplets;
      triplets.reserve( H.nonZeros() + H.rows() );

      std::set<int> diagonal_entries;

      for ( int k = 0; k < H.outerSize(); ++k )
      {
        for ( typename SparseMatrix::InnerIterator it( H, k ); it; ++it )
        {
          if ( it.row() == it.col() )
          {
            triplets.emplace_back( it.row(), it.col(), it.value() + lambda );
            diagonal_entries.insert( it.row() );
          }
          else
          {
            triplets.emplace_back( it.row(), it.col(), it.value() );
          }
        }
      }

      for ( int i = 0; i < H.rows(); ++i )
      {
        if ( diagonal_entries.find( i ) == diagonal_entries.end() ) { triplets.emplace_back( i, i, lambda ); }
      }

      H.setFromTriplets( triplets.begin(), triplets.end() );
    }

  public:
    HessianRegularizer( Options opts = Options() ) : m_opts( opts ) {}

    bool
    compute_direction( SparseMatrix H, Vector const & g, Vector & p )
    {
      Scalar lambda  = m_current_lambda > 0 ? m_current_lambda : m_opts.lambda_min;
      bool   success = false;

      for ( size_t attempt = 0; attempt < m_opts.max_attempts; ++attempt )
      {
        if ( attempt > 0 || m_current_lambda > 0 ) { add_diagonal_regularization( H, lambda ); }

        Eigen::SimplicialLLT<SparseMatrix> llt;
        llt.compute( H );

        if ( llt.info() == Eigen::Success )
        {
          p = llt.solve( -g );

          if ( is_valid_descent( p, g ) )
          {
            m_current_lambda = lambda;
            success          = true;
            break;
          }
        }

        if ( attempt == 0 ) { m_modification_count++; }

        lambda = ( attempt < 3 ) ? m_opts.lambda_min * std::pow( Scalar( 2 ), attempt + 1 )
                                 : lambda * m_opts.lambda_factor;

        if ( lambda > m_opts.lambda_max ) break;
      }

      return success;
    }

    // NEW: Adaptive lambda management
    void
    decrease_lambda()
    {
      m_current_lambda = std::max( m_current_lambda / Scalar( 2 ), m_opts.lambda_min );
    }

    void
    increase_lambda()
    {
      m_current_lambda = std::min( m_current_lambda * Scalar( 2 ), m_opts.lambda_max );
    }

    size_t
    modification_count() const
    {
      return m_modification_count;
    }
    void
    reset_count()
    {
      m_modification_count = 0;
    }
    Scalar
    current_lambda() const
    {
      return m_current_lambda;
    }
  };

  // ===========================================================================
  template <typename Scalar = double>
  class Newton_minimizer
  {
  public:
    using Vector       = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using SparseMatrix = Eigen::SparseMatrix<Scalar>;
    using Callback     = std::function<Scalar( Vector const &, Vector *, SparseMatrix * )>;

    enum class Status
    {
      CONVERGED          = 0,
      MAX_ITERATIONS     = 1,
      LINE_SEARCH_FAILED = 2,
      GRADIENT_TOO_SMALL = 3,
      FAILED             = 4,
      STAGNATION         = 5
    };

    static string
    to_string( Status status )
    {
      switch ( status )
      {
        case Status::CONVERGED:
          return "CONVERGED";
        case Status::MAX_ITERATIONS:
          return "MAX_ITER";
        case Status::LINE_SEARCH_FAILED:
          return "LINE_SEARCH_FAILED";
        case Status::GRADIENT_TOO_SMALL:
          return "GRAD_SMALL";
        case Status::STAGNATION:
          return "STAGNATION";
        case Status::FAILED:
          return "FAILED";
        default:
          return "UNKNOWN";
      }
    }

    struct Result
    {
      Status status{ Status::FAILED };
      size_t total_iterations{ 0 };
      size_t total_evaluations{ 0 };
      size_t hessian_evaluations{ 0 };
      size_t hessian_modifications{ 0 };
      Scalar final_gradient_norm{ 0 };
      Scalar final_function_value{ 0 };
      Scalar initial_function_value{ 0 };
      size_t line_search_evaluations{ 0 };
    };

    struct Options
    {
      size_t max_iter{ 100 };

      Scalar g_tol{ 1e-8 };
      Scalar g_tol_weak{ 1e-4 };
      Scalar f_tol{ 1e-12 };
      Scalar x_tol{ 1e-10 };

      Scalar step_max{ 10.0 };
      Scalar step_min{ 1e-20 };  // NEW: Absolute minimum step
      Scalar very_small_step{ 1e-8 };

      typename HessianRegularizer<Scalar>::Options hessian_opts;

      size_t verbosity_level{ 1 };

      // NEW: Stagnation detection
      size_t max_stagnation_iter{ 10 };
      Scalar stagnation_f_tol{ 1e-14 };
      Scalar stagnation_x_tol{ 1e-14 };

      // NEW: Recovery strategies
      bool   use_gradient_when_stagnant{ true };
      bool   use_hessian_recomputation{ true };
      size_t hessian_recompute_interval{ 10 };
    };

  private:
    Options                      m_options;
    BoxConstraintHandler<Scalar> m_constraints;
    HessianRegularizer<Scalar>   m_hessian_reg;

    mutable Vector       m_x, m_g, m_p, m_x_new, m_g_new;
    mutable SparseMatrix m_H;
    mutable size_t       m_function_evaluations{ 0 };
    mutable size_t       m_hessian_evaluations{ 0 };
    mutable size_t       m_line_search_evaluations{ 0 };

    Scalar m_epsi{ std::numeric_limits<Scalar>::epsilon() };
    string m_indent{ "" };

    // NEW: Stagnation tracking
    size_t m_stagnation_counter{ 0 };
    Scalar m_best_f{ std::numeric_limits<Scalar>::infinity() };
    Vector m_best_x;

    bool
    validate_callback_result( Scalar f, Vector const & g, bool check_grad ) const
    {
      if ( !std::isfinite( f ) ) return false;
      if ( check_grad && !g.allFinite() ) return false;
      return true;
    }

    // NEW: Enhanced line search with shorter initial step when stagnating
    template <typename Linesearch>
    std::optional<std::pair<Scalar, size_t>>
    adaptive_line_search(
      Scalar             f,
      Vector const &     x,
      Vector const &     p,
      Vector const &     g,
      Callback const &   callback,
      Linesearch const & linesearch,
      bool               use_short_step )
    {
      Scalar pg = p.dot( g );

      // Limit initial step based on constraints
      Scalar alpha_max = m_constraints.max_step_length( x, p );
      alpha_max        = std::min( alpha_max, m_options.step_max );

      // If stagnating, start with much shorter step
      if ( use_short_step ) { alpha_max = std::min( alpha_max, Scalar( 0.1 ) ); }

      auto ls_callback = [&]( Vector const & x_test, Vector * g_test ) -> Scalar
      { return callback( x_test, g_test, nullptr ); };

      return linesearch( f, pg, x, p, ls_callback, alpha_max );
    }

    // NEW: Gradient descent step as recovery
    bool
    try_gradient_step(
      Vector const &   x,
      Vector const &   g,
      Scalar           f,
      Vector &         x_new,
      Vector &         g_new,
      Scalar &         f_new,
      Callback const & callback )
    {
      Vector grad_dir = -g;
      m_constraints.project_direction( x, grad_dir );

      Scalar gnorm = grad_dir.norm();
      if ( gnorm < m_epsi ) return false;

      grad_dir /= gnorm;  // Normalize

      // Try exponentially decreasing steps
      const std::array<Scalar, 8> steps = { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9 };

      for ( Scalar alpha : steps )
      {
        x_new = x + alpha * grad_dir;
        m_constraints.project( x_new );

        f_new = callback( x_new, &g_new, nullptr );
        m_function_evaluations++;

        if ( validate_callback_result( f_new, g_new, true ) )
        {
          // Accept if we get ANY improvement
          if ( f_new < f - m_epsi * std::abs( f ) )
          {
            if ( m_options.verbosity_level >= 2 )
            {
              fmt::print(
                fmt::fg( fmt::color::yellow ),
                "{}[Newton] Gradient recovery: α={:.2e}, Δf={:.2e}\n",
                m_indent,
                alpha,
                f_new - f );
            }
            return true;
          }
        }
      }

      return false;
    }

    void
    print_header( size_t n, Scalar f0 ) const
    {
      if ( m_options.verbosity_level < 1 ) return;

      fmt::print(
        fmt::fg( fmt::color::light_blue ),
        "\n"
        "╔════════════════════════════════════════════════════════════════╗\n"
        "║                   Newton Optimization                          ║\n"
        "╠════════════════════════════════════════════════════════════════╣\n"
        "║ Dimension: {:<51} ║\n"
        "║ Max Iterations: {:<46} ║\n"
        "║ Gradient Tolerance: {:<42.2e} ║\n"
        "║ Bounds: {:<54} ║\n"
        "╚════════════════════════════════════════════════════════════════╝\n"
        "Initial F = {:.6e}\n\n",
        n,
        m_options.max_iter,
        m_options.g_tol,
        m_constraints.is_active() ? "Active" : "None",
        f0 );
    }

    void
    print_iteration( size_t iter, Scalar f, Scalar gnorm, Scalar step, Scalar pg, string const & mode ) const
    {
      if ( m_options.verbosity_level < 2 ) return;

      auto color = ( mode == "GRAD" || mode == "STAG" ) ? fmt::fg( fmt::color::yellow ) : fmt::fg( fmt::color::white );

      fmt::print(
        color,
        "[{:4d}] F={:.6e} | ‖pg‖={:.2e} | α={:.2e} | p·g={:.2e} {}\n",
        iter,
        f,
        gnorm,
        step,
        pg,
        mode );
    }

  public:
    Newton_minimizer( Options opts = Options() ) : m_options( opts ), m_hessian_reg( opts.hessian_opts ) {}

    void
    set_indent( string const & indent )
    {
      m_indent = indent;
    }
    Vector const &
    solution() const
    {
      return m_x;
    }

    void
    set_bounds( Vector const & lower, Vector const & upper )
    {
      m_constraints.set_bounds( lower, upper );
    }

    template <typename Linesearch>
    Result
    minimize(
      Vector const &     x0,
      Callback const &   callback,
      Linesearch const & linesearch = MoreThuenteLineSearch<Scalar>() )
    {
      m_function_evaluations    = 0;
      m_hessian_evaluations     = 0;
      m_line_search_evaluations = 0;
      m_hessian_reg.reset_count();
      m_stagnation_counter = 0;

      size_t n = x0.size();
      m_x.resize( n );
      m_g.resize( n );
      m_p.resize( n );
      m_x_new.resize( n );
      m_g_new.resize( n );
      m_best_x.resize( n );
      m_H.resize( n, n );

      m_x = x0;
      m_constraints.project( m_x );

      Scalar f = callback( m_x, &m_g, &m_H );
      m_function_evaluations++;
      m_hessian_evaluations++;

      if ( !validate_callback_result( f, m_g, true ) )
      {
        return { Status::FAILED, 0, m_function_evaluations, m_hessian_evaluations, 0, 0, f, f, 0 };
      }

      Scalar f_initial = f;
      m_best_f         = f;
      m_best_x         = m_x;

      print_header( n, f_initial );

      Status status = Status::MAX_ITERATIONS;
      Scalar f_prev = f;
      Vector x_prev = m_x;

      for ( size_t iter = 0; iter < m_options.max_iter; ++iter )
      {
        Scalar gnorm = m_constraints.projected_gradient_norm( m_x, m_g );

        // Check convergence
        if ( gnorm <= m_options.g_tol )
        {
          status = Status::CONVERGED;
          if ( m_options.verbosity_level >= 1 )
          {
            fmt::print(
              fmt::fg( fmt::color::green ),
              "{}[Newton] Converged: ‖pg‖ = {:.2e} ≤ {:.2e}\n",
              m_indent,
              gnorm,
              m_options.g_tol );
          }
          break;
        }

        // Recompute Hessian periodically or when stagnating
        bool need_hessian = ( iter % m_options.hessian_recompute_interval == 0 ) ||
                            ( m_stagnation_counter > 3 && m_options.use_hessian_recomputation );

        if ( need_hessian && iter > 0 )
        {
          callback( m_x, nullptr, &m_H );
          m_hessian_evaluations++;
        }

        // Try Newton direction
        bool direction_ok = m_hessian_reg.compute_direction( m_H, m_g, m_p );

        if ( direction_ok )
        {
          m_constraints.project_direction( m_x, m_p );
          Scalar pg = m_p.dot( m_g );

          if ( pg >= -m_epsi * ( Scalar( 1 ) + std::abs( pg ) ) ) { direction_ok = false; }
        }

        // Fallback to gradient if Newton fails
        if ( !direction_ok )
        {
          m_p = -m_g;
          m_constraints.project_direction( m_x, m_p );

          if ( m_options.verbosity_level >= 2 )
          {
            fmt::print( fmt::fg( fmt::color::yellow ), "{}[Newton] Using gradient descent direction\n", m_indent );
          }
        }

        Scalar pg = m_p.dot( m_g );

        // Try line search
        bool use_short_step = m_stagnation_counter > 2;
        auto step_opt       = adaptive_line_search( f, m_x, m_p, m_g, callback, linesearch, use_short_step );

        bool   step_success = false;
        Scalar step_size    = 0;

        if ( step_opt.has_value() )
        {
          auto [s, n_evals] = *step_opt;
          step_size         = s;
          m_function_evaluations += n_evals;
          m_line_search_evaluations += n_evals;

          if ( s > m_options.step_min )
          {
            m_x_new = m_x + s * m_p;
            m_constraints.project( m_x_new );

            Scalar f_new = callback( m_x_new, &m_g_new, &m_H );
            m_function_evaluations++;
            m_hessian_evaluations++;

            if ( validate_callback_result( f_new, m_g_new, true ) )
            {
              m_x.swap( m_x_new );
              m_g.swap( m_g_new );
              f            = f_new;
              step_success = true;
            }
          }
        }

        // If line search failed, try gradient recovery
        if ( !step_success && m_options.use_gradient_when_stagnant )
        {
          if ( try_gradient_step( m_x, m_g, f, m_x_new, m_g_new, f, callback ) )
          {
            m_x.swap( m_x_new );
            m_g.swap( m_g_new );
            step_success = true;
            step_size    = ( m_x - x_prev ).norm();

            // Recompute Hessian after recovery
            callback( m_x, nullptr, &m_H );
            m_hessian_evaluations++;
          }
        }

        // Update stagnation counter
        Scalar f_change = std::abs( f - f_prev );
        Scalar x_change = ( m_x - x_prev ).norm();

        if ( step_success )
        {
          if ( f_change < m_options.stagnation_f_tol && x_change < m_options.stagnation_x_tol )
          {
            m_stagnation_counter++;
          }
          else
          {
            m_stagnation_counter = 0;
            m_hessian_reg.decrease_lambda();  // Reduce regularization on progress
          }

          if ( f < m_best_f )
          {
            m_best_f = f;
            m_best_x = m_x;
          }

          f_prev = f;
          x_prev = m_x;
        }
        else
        {
          m_stagnation_counter++;
          m_hessian_reg.increase_lambda();  // Increase regularization on failure
        }

        // Check for terminal stagnation
        if ( m_stagnation_counter >= m_options.max_stagnation_iter )
        {
          status = Status::STAGNATION;
          if ( m_options.verbosity_level >= 1 )
          {
            fmt::print(
              fmt::fg( fmt::color::red ),
              "{}[Newton] Stagnation detected ({} iterations)\n",
              m_indent,
              m_stagnation_counter );
          }

          // Return best point found
          m_x = m_best_x;
          f   = m_best_f;
          callback( m_x, &m_g, nullptr );
          break;
        }

        string mode = step_success ? "OK" : "FAIL";
        if ( m_stagnation_counter > 0 ) mode += fmt::format( "(S:{})", m_stagnation_counter );

        print_iteration( iter, f, gnorm, step_size, pg, mode );
      }

      Scalar final_gnorm = m_constraints.projected_gradient_norm( m_x, m_g );

      if ( m_options.verbosity_level >= 1 )
      {
        fmt::print(
          fmt::fg( fmt::color::cyan ),
          "\n{}[Newton] Final: F={:.6e}, ‖pg‖={:.2e}, Status={}\n",
          m_indent,
          f,
          final_gnorm,
          to_string( status ) );
      }

      return { status,
               std::min( m_options.max_iter, m_function_evaluations ),
               m_function_evaluations,
               m_hessian_evaluations,
               m_hessian_reg.modification_count(),
               final_gnorm,
               f,
               f_initial,
               m_line_search_evaluations };
    }
  };

}  // namespace Utils

#endif
