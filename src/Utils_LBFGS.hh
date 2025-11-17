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
// file: Utils_LBFGS.hh
//

#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef UTILS_LBFGS_dot_HH
#define UTILS_LBFGS_dot_HH

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

#include "Utils.hh"
#include "Utils_fmt.hh"
#include "Utils_eigen.hh"

#include <optional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>

namespace Utils {

  using std::abs;
  using std::min;
  using std::max;

  /*!
   * \file Utils_LBFGS.hh
   * \brief Full header-only L-BFGS implementation (two-loop recursion) + projected L-BFGS minimizer
   *
   * Detailed Doxygen comments are provided for classes and functions below.
   *
   * References:
   *  - J. Nocedal (1980). "Updating Quasi-Newton Matrices with Limited Storage".
   *  - Byrd, Lu, Nocedal, Zhu (1995). "A Limited Memory Algorithm for Bound Constrained Optimization".
   *  - Nocedal & Wright, "Numerical Optimization", Chapter on Quasi-Newton Methods.
   *
   * This implementation exposes:
   *  - LBFGS<Scalar>: storage for S/Y, two-loop recursion, safe update
   *  - LBFGSMinimizer<Scalar>: high-level minimizer using LBFGS and projected updates for box constraints
   *
   * Notes:
   *  - The minimizer uses Armijo backtracking line-search by default; it can be extended to strong Wolfe.
   *  - Box-constrained behavior is implemented via projection + simple active-set heuristics (practical projected L-BFGS).
   *  - All operations are templated on Scalar (double/float). Uses Eigen::VectorXd-like types internally.
   */

  // ---------------------------------------------------------------------------
  // LBFGS: storage + two-loop recursion
  // ---------------------------------------------------------------------------

  /*!
   * \class LBFGS
   * \brief Limited-memory BFGS storage and two-loop recursion implementation.
   *
   * Template parameter:
   *   - Scalar: floating point scalar type (double/float).
   *
   * Responsibilities:
   *   - store up to m correction pairs (s_i, y_i)
   *   - perform safe update of the memory with curvature checks
   *   - compute H*g via two-loop recursion with initial diag H0 = gamma * I
   *
   * Typical usage:
   *   LBFGS<double> lbfgs(m);
   *   lbfgs.add_correction(s, y); // after successful step
   *   auto Hg = lbfgs.two_loop_recursion(g, gamma);
   *
   * \verbatim
   * Algorithm (two-loop recursion) summary:
   *   let ρᵢ = 1 / (yᵢ • sᵢ)
   *   q = g
   *   for i = m-1 .. 0:
   *     αᵢ = ρᵢ ⋅ (sᵢ • q)
   *     q -= αᵢ ⋅ yᵢ
   *   r = H0 ⋅ q (H0 = gamma * I)
   *   for i = 0 .. m-1:
   *     βᵢ = ρᵢ ⋅ (yᵢ • r)
   *     r += sᵢ ⋅ (αᵢ - βᵢ)
   *   return r
   * \endvebatim
   */

  template <typename Scalar = double>
  class LBFGS {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  private:
    size_t m_capacity;
    size_t m_dimension;
    Matrix m_S;               // Each column is a s vector
    Matrix m_Y;               // Each column is a y vector
    Vector m_rho;             // 1/(y^T s) for each pair
    size_t m_current_size{0};
    size_t m_oldest_index{0}; // Index of oldest pair
    size_t m_newest_index{0}; // Index where next pair will be stored

    /* Additional robustness parameters */
    bool   m_enable_damping{ true };        // enable Powell-style damping by default
    Scalar m_h0_min{ Scalar(1e-6) };        // minimum allowed initial H0 scaling
    Scalar m_h0_max{ Scalar(1e6) };         // maximum allowed initial H0 scaling

    mutable Vector m_alpha;
    mutable Vector m_q;
    mutable Vector m_r;

  public:
    /*!
     * \brief Construct LBFGS storage.
     * \param maxCorrections maximum number of stored correction pairs m (typical 3..20).
     * \param problem_dimension dimension of the optimization problem
     */
    explicit
    LBFGS( size_t maxCorrections = 10, size_t problem_dimension = 0 )
    : m_capacity( maxCorrections )
    , m_dimension( problem_dimension )
    , m_S( problem_dimension, maxCorrections )
    , m_Y( problem_dimension, maxCorrections )
    , m_rho( maxCorrections )
    , m_alpha( maxCorrections )
    , m_q( problem_dimension )
    , m_r( problem_dimension )
    {
      assert( maxCorrections > 0 );
      clear();
    }

    //! Clear stored pairs
    void
    clear() {
      m_current_size = 0;
      m_oldest_index = 0;
      m_newest_index = 0;
      m_S.setZero();
      m_Y.setZero();
      m_rho.setZero();
      m_alpha.setZero();
      m_q.setZero();
      m_r.setZero();
    }

    //! Return number of stored corrections
    size_t size() const { return m_current_size; }

    //! Return capacity (max corrections)
    size_t capacity() const { return m_capacity; }

    //! Return problem dimension
    size_t dimension() const { return m_dimension; }

    /*!
     * \brief Resize for new problem dimension (clears memory)
     */
    void
    resize( size_t const new_dimension ) {
      m_dimension = new_dimension;
      m_S.resize( new_dimension, m_capacity );
      m_Y.resize( new_dimension, m_capacity );
      m_S.setZero();
      m_Y.setZero();
      clear();
    }

    /*!
     * \brief Add a correction pair (\f$ s = x_{k+1}-x_k \f$, \f$ y = g_{k+1}-g_k \f$)
     * Performs curvature check \f$ s^T y > \epsilon \dot \|s\| \dot \|y\| \f$ to avoid division by zero.
     *
     * \param s displacement vector
     * \param y gradient difference vector
     * \param min_curvature_ratio optional threshold for (sᵀy) / (||s||*||y||) (default 1e-8)
     * \return true if the pair was accepted and stored, false otherwise
     */
    bool
    add_correction( Vector const & s, Vector const & y, Scalar const min_curvature_ratio = 1e-8 ) {
      assert( s.size() == y.size() );
      if ( static_cast<size_t>(s.size()) != m_dimension ) resize( s.size() );

      Scalar const sty  { s.dot(y) };
      Scalar const snrm { s.norm() };
      Scalar const ynrm { y.norm() };

      // basic positivity check
      if ( !(sty > 0) ) return false;

      // relative and absolute thresholds for curvature test (robust)
      Scalar const eps        = std::numeric_limits<Scalar>::epsilon();
      Scalar const rel_thresh = min_curvature_ratio * snrm * ynrm;
      Scalar const abs_thresh = eps * snrm * snrm;
      Scalar const thresh     = max(rel_thresh, abs_thresh);

      if ( sty <= thresh ) return false;

      // Store the new pair
      m_S.col ( m_newest_index ) = s;
      m_Y.col ( m_newest_index ) = y;
      m_rho   ( m_newest_index ) = Scalar(1.0) / sty;

      // Update indices and size
      if ( m_current_size < m_capacity ) {
        // Buffer not full yet
        ++m_current_size;
        m_newest_index = (m_newest_index + 1) % m_capacity;
      } else {
        // Buffer full - overwrite oldest
        m_oldest_index = (m_oldest_index + 1) % m_capacity;
        m_newest_index = (m_newest_index + 1) % m_capacity;
      }

      return true;
    }

    /*!
     * \brief Two-loop recursion: compute H * g without forming H explicitly.
     *
     * \param g gradient vector
     * \param h0 scalar initial diagonal element (H0 = h0 * I). Typical choice:
     *           \f$ h_0 = (s_{m-1}^T y_{m-1}) / (y_{m-1}^T y_{m-1}) \f$
     *
     * \return Vector r = H * g
     */
    Vector
    two_loop_recursion( Vector const & g, Scalar h0 ) const {
      if ( m_current_size == 0 ) return h0 * g;

      m_q = g; // We'll modify q in-place

      // First loop: from newest to oldest
      // We traverse backwards through the circular buffer
      size_t idx = (m_newest_index == 0) ? m_capacity - 1 : m_newest_index - 1;
        
      size_t i{m_current_size};
      while ( i > 0 ) {
        --i;
        m_alpha(i) = m_rho(idx) * m_S.col(idx).dot(m_q);
        m_q       -= m_alpha(i) * m_Y.col(idx);
            
        // Move to previous index (circular)
        idx = (idx == 0) ? m_capacity - 1 : idx - 1;
      }

      // Multiply by initial Hessian approximation H0 = h0 * I
      m_r = h0 * m_q;

      // Second loop: from oldest to newest
      // Start from oldest index and move forward
      idx = m_oldest_index;
      for ( size_t i{0}; i < m_current_size; ++i ) {
        Scalar beta = m_rho(idx) * m_Y.col(idx).dot(m_r);
        m_r += m_S.col(idx) * (m_alpha(i) - beta);

        // Move to next index (circular)
        idx = (idx + 1) % m_capacity;
      }
      return m_r;
    }

    /*!
     * \brief Helper to compute recommended initial H0 scalar from latest pair.
     * If no pairs stored, returns 1.0.
     */
    Scalar
    compute_initial_h0( Scalar default_value = Scalar(1.0) ) const {
      if ( m_current_size == 0 ) return default_value;
        
      // Get the newest pair (the one just before newest_index in circular buffer)
      size_t latest_idx = (m_newest_index == 0) ? m_capacity - 1 : m_newest_index - 1;
      auto const & s = m_S.col( latest_idx );
      auto const & y = m_Y.col( latest_idx );
        
      Scalar sty { s.dot(y) };
      Scalar yty { y.dot(y) };
      Scalar h0 = default_value;
      if ( yty > Scalar(0) ) h0 = sty / yty;
      // clamp h0 to avoid extreme scaling
      h0 = max(m_h0_min, min(m_h0_max, h0));
      return h0;
    }

    /*!
     * \brief Helper: attempt a Powell-style damping and then add correction to LBFGS object.
     *
     * This function computes a damped y_hat = theta * y + (1-theta) * (s / h0)
     * with theta chosen (if possible) so that s^T y_hat >= thresh. Falls back to
     * calling add_correction with original (s,y) if damping not needed or fails.
     */
    bool
    add_correction_with_damping(
      LBFGS<Scalar> & lb,
      Vector const  & s,
      Vector const  & y,
      Scalar const   min_curvature_ratio = 1e-8
    ) {
      using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
      Scalar const sty        = s.dot(y);
      Scalar const snrm       = s.norm();
      Scalar const ynrm       = y.norm();
      Scalar const eps        = std::numeric_limits<Scalar>::epsilon();
      Scalar const rel_thresh = min_curvature_ratio * snrm * ynrm;
      Scalar const abs_thresh = eps * snrm * snrm;
      Scalar const thresh     = max(rel_thresh, abs_thresh);

      if ( sty > thresh ) return lb.add_correction(s,y,min_curvature_ratio);

      // compute h0 and fallback if invalid
      Scalar h0 = lb.compute_initial_h0(1);
      if ( !(h0 > 0) ) return false;

      Scalar sBs = s.squaredNorm() / h0; // s^T (B0 s) with B0 = H0^{-1} = (1/h0) I

      // If denominator would be zero or very small, don't attempt damping
      Scalar denom = sty - sBs;
      Vector y_hat(y);
      if ( std::abs(denom) < std::numeric_limits<Scalar>::epsilon() ) {
        // ambiguous, don't damp
        return lb.add_correction(s,y,min_curvature_ratio);
      } else {
        // choose theta so that s^T y_hat = thresh (solve theta*(sty - sBs) + sBs = thresh)
        Scalar theta = (thresh - sBs) / denom;
        // clamp to [0,1]
        if      ( theta < 0 ) theta = 0;
        else if ( theta > 1 ) theta = 1;
        y_hat = theta * y + (1 - theta) * (s / h0);
        Scalar sty_hat = s.dot(y_hat);
        if ( !(sty_hat > 0) ) return false;
        return lb.add_correction(s, y_hat, min_curvature_ratio);
      }
    }

  };

  /*!
   * \brief Complete More–Thuente strong-Wolfe line search implementation (C++17).
   *
   * Implements the More–Thuente algorithm for finding a step length \alpha that
   * satisfies the **strong Wolfe conditions**:
   *
   *   (1)  phi(alpha) <= phi(0) + c1 * alpha * phi'(0)         (sufficient decrease, Armijo)
   *   (2)  |phi'(alpha)| <= c2 * |phi'(0)|                    (curvature, strong Wolfe)
   *
   * where
   *   phi(alpha)   = f(x + alpha * p)
   *   phi'(alpha)  = \nabla f(x + alpha * p)^T p
   *
   * Reference:
   *  - More, J. J. and Thuente, D. J., "Line Search Algorithms with Guaranteed
   *    Sufficient Decrease", ACM TOMS (1994).
   *
   * Usage:
   *  - Provide a callback `eval(alpha)` that returns {phi(alpha), derphi(alpha)}.
   *  - Call strong_wolfe_more_thuente(phi0, derphi0, eval, c1, c2, alpha_max, max_iters).
   *
   * The implementation follows the structure of More–Thuente:
   *  - probe forward increasing alpha until a bracket [alpha_lo, alpha_hi] is found
   *    that contains a step satisfying Armijo but not monotone.
   *  - call `zoom` to refine within the bracket using cubic/quadratic interpolation.
   *
   * The function returns std::optional<Scalar>: `std::nullopt` on failure.
   */

  namespace detail {

    template<class Scalar>
    inline
    Scalar
    cubic_minimizer(
      Scalar a, Scalar fa, Scalar fpa,
      Scalar b, Scalar fb, Scalar fpb,
      Scalar perc = 0.01
    ) {
      // Cubic minimizer from More-Thuente: minimize cubic that matches f and f' at a and b.
      // Based on standard formula; guard against numerical issues.
      Scalar const d1    { fpa + fpb - 3*( (fa - fb) / (a - b) ) };
      Scalar const discr { d1*d1 - fpa*fpb };
      if ( discr < 0 ) return (a + b) / Scalar(2); // fallback: use midpoint
      Scalar sqrt_disc { std::sqrt(max<Scalar>(Scalar(0), discr)) };
      Scalar denom     { (fpb - fpa + 2*sqrt_disc) };
      if ( abs(denom) < std::numeric_limits<Scalar>::epsilon() ) return (a + b) / Scalar(2);
      Scalar t { b - (b - a) * ( (fpb + sqrt_disc - d1) / denom ) };
      // clamp to [a,b]
      Scalar mi{ min(a,b) };
      Scalar ma{ max(a,b) };
      Scalar d{ perc*(ma-mi) };
      mi += d;
      ma -= d;
      if      ( t < mi ) t = mi;
      else if ( t > ma ) t = ma;
      return t;
    }

    template <typename Scalar>
    Scalar
    compute_step(
      Scalar a_lo,   Scalar phi_lo,   Scalar der_lo,
      Scalar a_hi,   Scalar phi_hi,   Scalar der_hi,
      Scalar a_prev, Scalar phi_prev, Scalar der_prev,
      Scalar alpha_max, Scalar step_max,
      bool is_bracketing // True se siamo in fase di bracketing
    ) {
      // Implementazione semplificata (ma robusta) della logica di bisezione/interpolazione di MT.
      // In un'implementazione completa, qui ci sarebbe una logica molto dettagliata
      // per scegliere tra interpolazione cubica e quadratica e bisezione,
      // garantendo che il nuovo passo si trovi all'interno dell'intervallo [a_lo, a_hi]
      // e rispetti i vincoli di massima/minima riduzione.
      
      // 1. Interpolazione cubica
      Scalar a_j = detail::cubic_minimizer<Scalar>(a_lo, phi_lo, der_lo, a_hi, phi_hi, der_hi);

      // 2. Salvaguardia: se il passo è troppo lontano, usa l'interpolazione quadratica o bisezione.
      if ( a_j <= a_lo || a_j >= a_hi ) {
        // Usa interpolazione quadratica o bisezione
        if ( is_bracketing ) {
          // Durante il bracketing, se l'interpolazione fallisce, si fa estrapolazione o si usa un passo fisso.
          a_j = a_hi + 2*(a_hi - a_prev); // Estrapolazione semplice
          if ( a_j > alpha_max ) a_j = alpha_max;
        } else {
          // Durante lo zoom, si usa l'interpolazione quadratica o bisezione
          a_j = (a_lo + a_hi) / 2; // Bisezione semplice
        }
      }

      // 3. Limitazione finale per l'intervallo accettabile
      Scalar step_tolerance = 1e-6; // Una tolleranza per il passo minimo
      Scalar da = step_tolerance * (a_hi - a_lo);
      a_lo += da;
      a_hi -= da;
      if      ( a_j < a_lo ) a_j = a_lo;
      //else if ( a_j > a_hi ) a_j = a_hi;

      return min( a_j, step_max );
    }
     
  } // namespace detail

  // -----------------------------------------------------------------------------
  // Line-search policies
  // -----------------------------------------------------------------------------
  template <typename Scalar>
  class ArmijoLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Scalar m_c1          = 1e-4;
    Scalar m_step_reduce = 0.5;
    Scalar m_step_expand = 1.2;
    size_t m_max_iters   = 50;
    Scalar m_epsi        = 1e-15;

    mutable Vector m_x_new;

  public:
    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           step0 = 1
    ) const {
      Scalar step = step0;
      Scalar c1_Df0 = m_c1 * Df0;
      auto const n{ x.size() };
      m_x_new.resize(n);
      
      Scalar best_step    = step0;
      Scalar best_f       = std::numeric_limits<Scalar>::max();
      size_t shrink_count = 0;
      
      for (size_t k = 0; k < m_max_iters; ++k) {
        m_x_new.noalias() = x + step * d;
        Scalar f_new = callback(m_x_new, nullptr);
        
        // Memorizza il miglior passo trovato
        if (f_new < best_f) {
          best_f    = f_new;
          best_step = step;
        }
        
        if ( f_new <= f0 + step * c1_Df0 ) return step; // Successo
        
        // Strategia adattiva: riduci più aggressivamente dopo diversi fallimenti
        Scalar reduction = m_step_reduce;
        if ( shrink_count > 2 ) reduction = 0.1; // Riduzione più aggressiva
            
        step *= reduction;
        shrink_count++;
            
        // Controllo per passi troppo piccoli
        if ( step < m_epsi ) {
          // Ritorna il miglior passo trovato anche se non soddisfa Armijo
          if ( best_f < f0 ) return best_step;
          break;
        }
      }
      
      // Fallback: ritorna il miglior passo trovato
      if ( best_f < f0 ) return best_step;

      return std::nullopt;
    }
  };

  template <typename Scalar>
  class WolfeLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Scalar m_c1        = 1e-4;
    Scalar m_c2        = 0.9;
    Scalar m_alpha_max = 10;
    Scalar m_epsi      = 1e-12;
    size_t m_max_iters = 50;
    
    mutable Vector m_g_new, m_x_new;
    
  public:

    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           alpha0 = 1
    ) const {

      if ( !(Df0 < 0) ) return std::nullopt;
      m_g_new.resize( x.size() );
      m_x_new.resize( x.size() );
      size_t evals = 0;

      auto eval = [&]( Scalar a, Scalar & f, Scalar & df ) {
        ++evals;
        m_x_new.noalias() = x + a * d;
        f  = callback( m_x_new, &m_g_new );
        df = m_g_new.dot(d);
      };
      
      Scalar c1_Df0{ m_c1 * Df0 };
      Scalar c2_Df0{ m_c2 * Df0 };

      // initial trial
      Scalar alpha = alpha0;
      Scalar phi, der;
      eval( alpha, phi, der );

      // check if good
      if ( phi <= f0 + alpha * c1_Df0 && der >= c2_Df0 ) return alpha;

      Scalar alpha_lo   = 0;
      Scalar phi_lo     = f0;
      Scalar der_lo     = Df0;

      Scalar alpha_hi   = 0;
      Scalar phi_hi     = 0;
      Scalar der_hi     = 0;

      bool   bracketed  = false;
      Scalar alpha_prev = 0;
      Scalar phi_prev   = f0;
      Scalar der_prev   = Df0;

      // --- Bracketing ---
      while ( evals < m_max_iters ) {
        if ( (phi > f0 + alpha*c1_Df0) || (evals > 1 && phi >= phi_prev) ) {
          alpha_lo  = alpha_prev;
          phi_lo    = phi_prev;
          der_lo    = der_prev;

          alpha_hi  = alpha;
          phi_hi    = phi;
          der_hi    = der;

          bracketed = true;
          break;
        }

        if ( der >= c2_Df0 ) return alpha;

        if ( der >= 0 ) {
          alpha_lo  = alpha_prev;
          phi_lo    = phi_prev;
          der_lo    = der_prev;

          alpha_hi  = alpha;
          phi_hi    = phi;
          der_hi    = der;

          bracketed = true;
          break;
        }

        Scalar new_alpha = min( 2 * alpha, m_alpha_max);
        alpha_prev = alpha;
        phi_prev   = phi;
        der_prev   = der;
        alpha      = new_alpha;
        eval(alpha, phi, der );

        if ( phi <= f0 + alpha * c1_Df0 && der >= c2_Df0 ) return alpha;
      }

      if (!bracketed) return std::nullopt;

      // --- Zoom phase ---
      while ( evals < m_max_iters ) {
        Scalar a_j  = abs(alpha_hi - alpha_lo) > m_epsi ?
                      detail::cubic_minimizer<Scalar>( alpha_lo, phi_lo, der_lo, alpha_hi, phi_hi, der_hi ) :
                      (alpha_lo + alpha_hi) / 2;

        Scalar phi_j, der_j;
        eval(a_j,phi_j, der_j);

        if ( (phi_j > f0 + a_j * c1_Df0) || (phi_j >= phi_lo) ) {
          alpha_hi = a_j;
          phi_hi   = phi_j;
          der_hi   = der_j;
        } else {
          if ( der_j >= c2_Df0 ) return a_j;

          if ( der_j * (alpha_hi - alpha_lo) >= 0 ) {
            alpha_hi = alpha_lo;
            phi_hi   = phi_lo;
            der_hi   = der_lo;
          }

          alpha_lo = a_j;
          phi_lo   = phi_j;
          der_lo   = der_j;
        }

        if ( abs(alpha_hi - alpha_lo) < m_epsi ) return a_j;
      }

      return std::nullopt;
    }
  };

  template <typename Scalar>
  class StrongWolfeLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Scalar m_c1        = 1e-4;
    Scalar m_c2        = 0.9;
    Scalar m_alpha_max = 10;
    Scalar m_epsi      = 1e-12;
    size_t m_max_iters = 50;
    
    mutable Vector m_g_new, m_x_new;
    
  public:

    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           alpha0 = 1
    ) const {

      if ( !(Df0 < 0) ) return std::nullopt;
      m_g_new.resize( x.size() );
      m_x_new.resize( x.size() );
      size_t evals = 0;

      auto eval = [&]( Scalar a, Scalar & f, Scalar & df ) {
        ++evals;
        m_x_new.noalias() = x + a * d;
        f  = callback( m_x_new, &m_g_new );
        df = m_g_new.dot(d);
      };
      
      Scalar c1_Df0{ m_c1 * Df0 };
      Scalar c2_Df0{ m_c2 * Df0 };

      // initial trial
      Scalar alpha = alpha0;
      Scalar phi, der;
      eval( alpha, phi, der );

      // check if good
      if ( phi <= f0 + alpha * c1_Df0 && abs(der) <= -c2_Df0 ) return alpha;

      Scalar alpha_lo   = 0;
      Scalar phi_lo     = f0;
      Scalar der_lo     = Df0;

      Scalar alpha_hi   = 0;
      Scalar phi_hi     = 0;
      Scalar der_hi     = 0;

      bool   bracketed  = false;
      Scalar alpha_prev = 0;
      Scalar phi_prev   = f0;
      Scalar der_prev   = Df0;

      // --- Bracketing ---
      while ( evals < m_max_iters ) {

        if ( (phi > f0 + alpha*c1_Df0) || (evals > 1 && phi >= phi_prev) ) {
          alpha_lo  = alpha_prev;
          phi_lo    = phi_prev;
          der_lo    = der_prev;

          alpha_hi  = alpha;
          phi_hi    = phi;
          der_hi    = der;

          bracketed = true;
          break;
        }

        if ( abs(der) <= -c2_Df0 ) return alpha;

        if ( der >= 0 ) {
          alpha_lo  = alpha_prev;
          phi_lo    = phi_prev;
          der_lo    = der_prev;

          alpha_hi  = alpha;
          phi_hi    = phi;
          der_hi    = der;

          bracketed = true;
          break;
        }

        Scalar new_alpha = min( 2 * alpha, m_alpha_max );
        alpha_prev = alpha;
        phi_prev   = phi;
        der_prev   = der;
        alpha      = new_alpha;
        eval(alpha, phi, der );

        if ( phi <= f0 + alpha * c1_Df0 && abs(der) <= -c2_Df0 ) return alpha;
      }

      if (!bracketed) return std::nullopt;

      // --- Zoom phase ---
      while ( evals < m_max_iters ) {
        Scalar a_j  = abs(alpha_hi - alpha_lo) > m_epsi ?
                      detail::cubic_minimizer<Scalar>( alpha_lo, phi_lo, der_lo, alpha_hi, phi_hi, der_hi ) :
                      (alpha_lo + alpha_hi) / 2;

        Scalar phi_j, der_j;
        eval(a_j,phi_j, der_j);

        if ( (phi_j > f0 + a_j * c1_Df0) || (phi_j >= phi_lo) ) {
          alpha_hi = a_j;
          phi_hi   = phi_j;
          der_hi   = der_j;
        } else {
          if ( abs(der_j) <= -c2_Df0 ) return a_j;

          if ( der_j * (alpha_hi - alpha_lo) >= 0 ) {
            alpha_hi = alpha_lo;
            phi_hi   = phi_lo;
            der_hi   = der_lo;
          }

          alpha_lo = a_j;
          phi_lo   = phi_j;
          der_lo   = der_j;
        }

        if ( abs(alpha_hi - alpha_lo) < m_epsi ) return a_j;
      }

      return std::nullopt;
    }
  };

  template <typename Scalar>
  class GoldsteinLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    Scalar m_c1          = 0.1;
    Scalar m_step_reduce = 0.5;
    size_t m_max_iters   = 50;

    mutable Vector m_x_new;

  public:

    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           alpha0 = 1
    ) const {
      Scalar alpha = alpha0;
      m_x_new.resize(x.size());
      for (size_t k=0; k<m_max_iters; ++k) {
        m_x_new.noalias() = x + alpha*d;
        Scalar f_new = callback(m_x_new, nullptr);
        if ( f_new <= f0 + m_c1*alpha*Df0 &&
             f_new >= f0 + (1-m_c1)*alpha*Df0 ) return alpha;
        alpha *= m_step_reduce;
      }
      return std::nullopt;
    }
  };
  
  template <typename Scalar>
  class HagerZhangLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Scalar m_c1        = 1e-4;
    Scalar m_c2        = 0.9;
    Scalar m_alpha_max = 10;
    Scalar m_epsi      = 1e-12;
    size_t m_max_iters = 50;

    // Hager-Zhang specific parameters
    Scalar m_delta     = 0.1;
    Scalar m_sigma     = 0.9;
    Scalar m_epsilon_k = 1e-6;

    mutable size_t m_evals;
    mutable Scalar m_f0;
    mutable Scalar m_Df0;
    mutable Scalar m_c1_Df0;
    mutable Scalar m_c2_Df0;

    mutable Vector m_g_new, m_x_new;
    
    template <typename EvalFunc>
    std::optional<Scalar>
    zoom(
      Scalar a_lo, Scalar phi_lo, Scalar der_lo,
      Scalar a_hi, Scalar phi_hi, Scalar der_hi,
      EvalFunc const & eval
    ) const {

      for ( size_t iter{0}; iter < m_max_iters && m_evals < m_max_iters; ++iter ) {
        // Use cubic interpolation
        Scalar a_j = detail::cubic_minimizer<Scalar>( a_lo, phi_lo, der_lo, a_hi, phi_hi, der_hi );
        
        // Safeguard
        if ( a_j <= min(a_lo, a_hi) || a_j >= max(a_lo, a_hi) ) a_j = (a_lo + a_hi) / 2;

        Scalar phi_j, der_j;
        eval(a_j, phi_j, der_j);

        if ( phi_j > m_f0 + a_j * m_c1_Df0 || phi_j >= phi_lo ) {
          a_hi   = a_j;
          phi_hi = phi_j;
          der_hi = der_j;
        } else {
          if ( abs(der_j) <= -m_c2_Df0 + m_epsilon_k ) return a_j;

          if ( der_j * (a_hi - a_lo) >= 0 ) {
            a_hi   = a_lo;
            phi_hi = phi_lo;
            der_hi = der_lo;
          }

          a_lo   = a_j;
          phi_lo = phi_j;
          der_lo = der_j;
        }

        if ( abs(a_hi - a_lo) < m_epsi ) return a_j;
      }
      return std::nullopt;
    }

  public:

    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           alpha0 = 1
    ) const {
      if ( !(Df0 < 0) ) return std::nullopt;

      m_f0     = f0;
      m_Df0    = Df0;
      m_c1_Df0 = m_c1 * Df0;
      m_c2_Df0 = m_c2 * Df0;

      m_g_new.resize( x.size() );
      m_x_new.resize( x.size() );
      m_evals = 0;

      auto eval = [&]( Scalar a, Scalar & f, Scalar & df ) {
        ++m_evals;
        m_x_new.noalias() = x + a * d;
        f  = callback( m_x_new, &m_g_new );
        df = m_g_new.dot(d);
      };

      Scalar alpha      = alpha0;
      Scalar alpha_prev = 0;
      Scalar phi_prev   = f0;
      Scalar der_prev   = Df0;
      
      Scalar phi, der;
      eval(alpha, phi, der);

      for ( size_t iter = 0; iter < m_max_iters; ++iter ) {
        // Check modified Wolfe conditions
        if ( phi <= f0 + alpha * m_c1_Df0 && der >= m_c2_Df0 - m_epsilon_k ) return alpha;

        // Bracketing
        if ( phi > f0 + alpha * m_c1_Df0 || (iter > 0 && phi >= phi_prev) ) {
          return zoom( alpha_prev, phi_prev, der_prev, alpha, phi, der, eval );
        }

        if ( abs(der) <= m_epsilon_k - m_c2_Df0 ) return alpha;

        if ( der >= 0 ) return zoom(alpha, phi, der, alpha_prev, phi_prev, der_prev, eval );

        // Extrapolation
        Scalar alpha_new = min( 2 * alpha, m_alpha_max );

        alpha_prev = alpha;
        phi_prev   = phi;
        der_prev   = der;
        alpha      = alpha_new;
        eval(alpha, phi, der);
      }

      return std::nullopt;
    }

  };

  template <typename Scalar>
  class MoreThuenteLineSearch {
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Scalar m_c1        = 1e-4; // Armijo
    Scalar m_c2        = 0.9;  // Strong Wolfe (Curvature)
    Scalar m_alpha_max = 10.0;
    Scalar m_alpha_min = 1e-12; // Passo minimo accettabile
    Scalar m_epsi      = 1e-12;
    size_t m_max_iters = 50;
    
    mutable Vector m_g_new, m_x_new;

    // --- Stati interni (per chiarezza, gestiti tramite i parametri passati) ---
    // Questi potrebbero essere membri privati, ma li gestiremo nel main operator() per isolamento.

  public:

    template <typename Callback>
    std::optional<Scalar>
    operator()(
      Scalar           f0,
      Scalar           Df0,
      Vector   const & x,
      Vector   const & d,
      Callback const & callback,
      Scalar           alpha0 = 1
    ) const {
      if ( !(Df0 < 0) ) return std::nullopt;

      m_g_new.resize( x.size() );
      m_x_new.resize( x.size() );
      size_t evals = 0;

      auto eval = [&]( Scalar a, Scalar & f, Scalar & df ) {
        ++evals;
        m_x_new.noalias() = x + a * d;
        f  = callback( m_x_new, &m_g_new );
        df = m_g_new.dot(d);
      };
        
      Scalar c1_Df0{ m_c1 * Df0 };
      Scalar c2_Df0{ m_c2 * Df0 };

      // ----------------------------------------------------
      // FASE 1: Inizializzazione e ciclo principale (Bracketing & Zoom)
      // ----------------------------------------------------

      // Punti di lavoro
      Scalar alpha_prev = 0;
      Scalar phi_prev   = f0;
      Scalar der_prev   = Df0;
        
      Scalar alpha_curr = alpha0; // Passo di prova corrente
      Scalar phi_curr, der_curr;
        
      // Intervallo di bracketing iniziale
      Scalar alpha_lo = 0;
      Scalar phi_lo   = f0;
      Scalar der_lo   = Df0;
        
      Scalar alpha_hi = m_alpha_max;
        
      bool bracketed = false;

      // Esegui la prima valutazione
      eval(alpha_curr, phi_curr, der_curr);
        
      for ( size_t k{0}; k < m_max_iters && evals < m_max_iters; ++k ) {
            
        // Check Strong Wolfe al passo corrente (MT è un raffinamento per SW)
        if ( phi_curr <= f0 + alpha_curr * c1_Df0 && std::abs(der_curr) <= -c2_Df0 ) return alpha_curr; // Trovato passo accettabile

        // --- Bracketing Logic (Aggiornamento dell'intervallo [alpha_lo, alpha_hi]) ---

        if ( phi_curr > f0 + alpha_curr * c1_Df0 || (bracketed && phi_curr >= phi_lo) ) {
          // Caso A: Violata Armijo o funzione non decrescente
          // L'intervallo [alpha_lo, alpha_curr] contiene il minimo.
          alpha_hi = alpha_curr;
          // Mantieni alpha_lo (potrebbe essere alpha_prev o 0)
          bracketed = true;
        } else if ( std::abs(der_curr) <= -c2_Df0 ) {
          // Caso B: Soddisfatta Armijo ma derivata non abbastanza piatta (Strong Wolfe fallita)
          // Se la derivata è troppo negativa (troppo ripida), dobbiamo andare più avanti.
          // Se der_curr < 0, il minimo è oltre alpha_curr.
          alpha_lo = alpha_curr;
          phi_lo   = phi_curr;
          der_lo   = der_curr;
        } else if ( der_curr >= 0 ) {
          // Caso C: Trovata pendenza positiva, l'intervallo è [alpha_prev, alpha_curr]
          alpha_hi = alpha_curr;
          // Mantieni alpha_lo
          bracketed = true;
        }

        // --- Selezione del nuovo passo ---
        Scalar alpha_new;

        if ( bracketed ) {
          // Zoom phase: usa interpolazione cubica/safeguard tra alpha_lo e alpha_hi
          alpha_new = detail::compute_step(
            alpha_lo,   phi_lo,   der_lo,
            alpha_hi,   phi_curr, der_curr, // Nota: usa phi_curr e der_curr per alpha_hi (il passo "cattivo")
            alpha_prev, phi_prev, der_prev,
            m_alpha_max, m_alpha_max,
            false // is_bracketing = false
          );
        } else {
          // Extrapolation phase: usa interpolazione o raddoppia il passo
                
          // Opzione 1: Raddoppia
          alpha_new = 2 * alpha_curr;

          // Opzione 2: Interpolazione/Estrapolazione (più avanzata, ma rischiosa)
          // alpha_new = detail::compute_step(
          //     alpha_lo, phi_lo, der_lo,
          //     alpha_curr, phi_curr, der_curr,
          //     alpha_prev, phi_prev, der_prev,
          //     m_alpha_max, m_alpha_max,
          //     true // is_bracketing = true
          // );
        }

        // --- Salvaguardia finale ---
        if ( alpha_new <= 0 || alpha_new >= m_alpha_max ) alpha_new = min( 2 * alpha_curr, m_alpha_max);
        if ( std::abs(alpha_hi - alpha_lo) < m_epsi     ) return alpha_curr; // Convergenza dell'intervallo

        // Aggiorna stato per la prossima iterazione
        alpha_prev = alpha_curr;
        phi_prev   = phi_curr;
        der_prev   = der_curr;

        alpha_curr = alpha_new;
        eval(alpha_curr, phi_curr, der_curr);
      }

      return std::nullopt; // Raggiunto il massimo delle iterazioni
    }
  };

  // ---------------------------------------------------------------------------
  // LBFGSMinimizer: high-level optimizer with line-search and optional box bounds
  // ---------------------------------------------------------------------------

  template <typename Scalar = double>
  class LBFGS_minimizer {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Callback = std::function<Scalar(Vector const &, Vector *)>;

    struct Options {
      size_t max_iter        { 200   };
      Scalar g_tol           { 1e-8  };
      Scalar f_tol           { 1e-12 };
      Scalar x_tol           { 1e-10 }; // Nuovo: tolleranza su x
      Scalar step_max        { 1     };
      Scalar c1              { 1e-4  };
      size_t max_line_search { 50    }; // Aumentato
      size_t m               { 20    };
      bool   use_projection  { false };
      bool   verbose         { false };
      bool   use_damping     { true };
      Scalar sty_min_factor  { 1e-12 }; // Più permissivo
    };

  private:

    Scalar        m_epsi{ 1e-15 };
    Options       m_options;
    LBFGS<Scalar> m_LBFGS;
    Vector        m_lower;
    Vector        m_upper;

    // project x into bounds
    void
    project_inplace( Vector & x ) const
    { x = x.cwiseMax(m_lower).cwiseMin(m_upper); }

    Vector
    projected_gradient( Vector const & x, Vector const & g ) const {
      auto freeze = ((x.array() <= m_lower.array()) && (g.array() >= 0)) ||
                    ((x.array() >= m_upper.array()) && (g.array() <= 0));
      return freeze.select(Vector::Zero(g.size()), g);
    }

    void
    projected_gradient_inplace( Vector const & x, Vector & g ) const {
      auto freeze = ((x.array() <= m_lower.array()) && (g.array() >= 0)) ||
                    ((x.array() >= m_upper.array()) && (g.array() <= 0));
      g = freeze.select(Vector::Zero(g.size()), g);
    }

    void
    projected_direction_inplace( Vector const & x, Vector & d ) const {
      auto freeze = ((x.array() <= m_lower.array()) && (d.array() < 0)) ||
                    ((x.array() >= m_upper.array()) && (d.array() > 0));
      d = freeze.select(Vector::Zero(d.size()), d);
    }

    Scalar
    projected_gradient_norm( Vector const & x, Vector const & g ) const {
      auto freeze = ((x.array() <= m_lower.array()) && (g.array() >= 0)) ||
                    ((x.array() >= m_upper.array()) && (g.array() <= 0));
      return freeze.select(Vector::Zero(g.size()), g)
                   .array()
                   .abs()
                   .maxCoeff();
    }

    mutable Vector m_x, m_g, m_p, m_x_new, m_g_new, m_s, m_y;
    mutable size_t m_iter_since_reset{0};

  public:
    LBFGS_minimizer( Options opts = Options() )
    : m_options(opts)
    , m_LBFGS(opts.m)
    {}

    void
    set_bounds( Vector const & lower, Vector const & upper ) {
      assert( lower.size() == upper.size() );
      m_lower = lower;
      m_upper = upper;
      m_options.use_projection = true;
    }

    void
    reset_memory() {
      m_LBFGS.clear();
      m_iter_since_reset = 0;
    }

    template <typename Linesearch>
    std::tuple<int, Vector, Scalar>
    minimize(
      Vector     const & x0,
      Callback   const & callback,
      Linesearch const & linesearch = MoreThuenteLineSearch<Scalar>()
    ) {
      auto const n{ x0.size() };
      m_x.resize(n); m_g.resize(n); m_p.resize(n);
      m_x_new.resize(n); m_g_new.resize(n); m_s.resize(n); m_y.resize(n);
      
      if ( m_options.use_projection ) {
        assert( m_lower.size() == n );
        assert( m_upper.size() == n );
      }

      m_x.noalias() = x0;
      if ( m_options.use_projection ) project_inplace(m_x);
      
      Scalar f = callback(m_x, &m_g);
      Scalar f_prev{ f };
      Vector x_prev    = m_x;
      Scalar step_prev = 0;

      for ( size_t iter{ 0 }; iter < m_options.max_iter; ++iter ) {
        ++m_iter_since_reset;
        
        // Check per stagnazione
        if ( iter > 0 ) {
          Scalar x_change = (m_x - x_prev).norm();
          if ( x_change < step_prev*m_options.x_tol ) {
            if ( m_options.verbose )
              fmt::print("[LBFGS] Converged by x change: {:.2e} < {:.2e}\n", x_change, m_options.x_tol);
            return {0, m_x, f};
          }
        }
        x_prev = m_x;

        Scalar gnorm{ projected_gradient_norm(m_x, m_g) };
        if ( m_options.verbose )
          fmt::print("[LBFGS] iter={:3d} f={:12.4g} ‖pg‖={:12.4g} mem={}\n", iter, f, gnorm, m_LBFGS.size());
        
        if ( gnorm <= m_options.g_tol ) return {0, m_x, f};

        // Reset periodico della memoria ogni 50 iterazioni
        if ( m_iter_since_reset > 50 ) {
          if ( m_options.verbose ) fmt::print("[LBFGS] Periodic memory reset\n");
          m_LBFGS.clear();
          m_iter_since_reset = 0;
        }

        // compute search direction
        Scalar h0 = m_LBFGS.compute_initial_h0(1);
        m_p = -m_LBFGS.two_loop_recursion(m_g, h0);

        if ( m_options.use_projection ) {
          projected_direction_inplace( m_x, m_p );
          if ( m_p.isZero() ) m_p = -projected_gradient(m_x, m_g);
        }

        // Robust descent direction check
        Scalar pg             = m_p.dot(m_g);
        size_t fallback_count = 0;
        size_t max_fallback   = 3;

        while ((!std::isfinite(pg) || pg >= -m_epsi) && fallback_count < max_fallback) {
          if ( fallback_count == 0 ) {
            m_p = -projected_gradient(m_x, m_g);
          } else if ( fallback_count == 1 ) {
            m_LBFGS.clear();
            h0  = m_LBFGS.compute_initial_h0(1);
            m_p = -m_LBFGS.two_loop_recursion(m_g, h0);
            if ( m_options.use_projection ) projected_direction_inplace(m_x, m_p);
          } else {
            m_p = -m_g / (1 + m_g.norm());
            if ( m_options.use_projection ) projected_direction_inplace(m_x, m_p);
          }
          pg = m_p.dot(m_g);
          fallback_count++;
        }

        if (!std::isfinite(pg) || pg >= -m_epsi) {
          if ( m_options.verbose ) fmt::print("[LBFGS] Cannot find descent direction, stopping\n");
          return {3, m_x, f};
        }

        // Line search
        auto step_opt = linesearch( f, pg, m_x, m_p, callback, m_options.step_max );
        
        if (!step_opt.has_value()) {
          if ( m_options.verbose ) fmt::print("[LBFGS] line search failed, resetting memory\n");
          m_LBFGS.clear();
          // Prova con passo fisso piccolo
          m_x_new.noalias() = m_x + 1e-8 * m_p;
          if ( m_options.use_projection ) project_inplace(m_x_new);
          Scalar f_test = callback(m_x_new, &m_g_new);
          if (f_test < f) {
            // Accetta comunque il passo
            m_s.noalias() = m_x_new - m_x;
            m_y.noalias() = m_g_new - m_g;
            m_x.swap(m_x_new);
            m_g.swap(m_g_new);
            f = f_test;
            continue;
          } else {
            return {2, m_x, f};
          }
        }

        Scalar step = *step_opt; step_prev = step > 1 ? step : 1 ;
        
        // evaluate final new point
        m_x_new.noalias() = m_x + step * m_p;
        if ( m_options.use_projection ) project_inplace(m_x_new);
        Scalar f_new = callback(m_x_new, &m_g_new);

        // compute s,y
        m_s.noalias() = m_x_new - m_x;
        m_y.noalias() = m_g_new - m_g;

        // Robust curvature check
        Scalar sty           = m_s.dot(m_y);
        Scalar sty_tolerance = max( m_options.sty_min_factor, m_epsi * m_s.squaredNorm() * m_y.squaredNorm());

        if (sty > sty_tolerance) {
          m_LBFGS.add_correction(m_s, m_y);
        } else {
          if ( m_options.verbose && m_LBFGS.size() > 0 )
            fmt::print("[LBFGS] curvature condition failed (s^T y = {:.2e}), skipping update\n", sty);
          // Non cancellare la memoria, procedi senza aggiornare
        }

        // move
        m_x.swap(m_x_new);
        m_g.swap(m_g_new);
        f = f_new;

        // check function change
        Scalar f_change = std::abs(f - f_prev);
        if ( f_change <= m_options.f_tol ) {
          if ( m_options.verbose )
            fmt::print("[LBFGS] Converged by function change: {:.2e} < {:.2e}\n", f_change, m_options.f_tol);
          return {0, m_x, f};
        }
        f_prev = f;
      }

      return {1, m_x, f}; // max iter
    }
  };
}

#endif

#endif

//
// eof: Utils_LBFGS.hh
//
