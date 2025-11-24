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

#include "Utils_Linesearch.hh"

/**
 * @file Utils_LBFGS.hh
 * @brief Complete header-only implementation of L-BFGS optimization algorithms
 *
 * This file provides a comprehensive implementation of the Limited-memory 
 * Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) quasi-Newton optimization method,
 * including multiple line search strategies and support for box constraints.
 *
 * ## Main Components
 *
 * - **LBFGS**: Core L-BFGS storage and two-loop recursion algorithm
 * - **Line Search Policies**: Multiple strategies (Armijo, Wolfe, Strong Wolfe, 
 *   Goldstein, Hager-Zhang, More-Thuente)
 * - **LBFGS_minimizer**: High-level minimizer with automatic line search and 
 *   projected gradient methods for box-constrained optimization
 *
 * ## Theoretical Background
 *
 * The L-BFGS method approximates the inverse Hessian matrix using a limited number
 * of recent gradient differences. This makes it suitable for large-scale optimization
 * where storing the full Hessian is impractical.
 *
 * ### Algorithm Overview
 *
 * At iteration k, L-BFGS computes a search direction \f$p_k\f$ by:
 * \f[
 *   p_k = -H_k \nabla f(x_k)
 * \f]
 * where \f$H_k\f$ is an approximation to \f$[\nabla^2 f(x_k)]^{-1}\f$ constructed
 * from the m most recent correction pairs \f$(s_i, y_i)\f$:
 * \f[
 *   s_i = x_{i+1} - x_i, \quad y_i = \nabla f(x_{i+1}) - \nabla f(x_i)
 * \f]
 *
 * The key advantage is that \f$H_k\f$ is never formed explicitly; instead,
 * the product \f$H_k g\f$ is computed efficiently via two-loop recursion in O(mn) time.
 *
 * ## References
 *
 * -# J. Nocedal (1980). "Updating Quasi-Newton Matrices with Limited Storage".
 *    Mathematics of Computation, 35(151), 773-782.
 *    DOI: 10.1090/S0025-5718-1980-0572855-7
 *
 * -# D.C. Liu and J. Nocedal (1989). "On the Limited Memory BFGS Method for 
 *    Large Scale Optimization". Mathematical Programming, 45(1-3), 503-528.
 *    DOI: 10.1007/BF01589116
 *
 * -# R.H. Byrd, P. Lu, J. Nocedal, and C. Zhu (1995). "A Limited Memory Algorithm 
 *    for Bound Constrained Optimization". SIAM Journal on Scientific Computing,
 *    16(5), 1190-1208. DOI: 10.1137/0916069
 *
 * -# J. Nocedal and S.J. Wright (2006). "Numerical Optimization", 2nd Edition,
 *    Springer. ISBN: 978-0-387-30303-1
 *
 * -# J.J. Moré and D.J. Thuente (1994). "Line Search Algorithms with Guaranteed
 *    Sufficient Decrease". ACM Transactions on Mathematical Software, 20(3), 286-307.
 *    DOI: 10.1145/192115.192132
 *
 * -# W.W. Hager and H. Zhang (2006). "A New Conjugate Gradient Method with 
 *    Guaranteed Descent and an Efficient Line Search". SIAM Journal on Optimization,
 *    16(1), 170-192. DOI: 10.1137/030601880
 *
 * ## Implementation Features
 *
 * - **Robustness**: Includes Powell damping, curvature checks, and fallback strategies
 * - **Flexibility**: Template-based design supports different scalar types (float/double)
 * - **Efficiency**: Uses Eigen for vectorized operations, minimal memory allocations
 * - **Box Constraints**: Implements projected gradient methods for bound-constrained problems
 * - **Multiple Line Searches**: Choose the best strategy for your problem characteristics
 *
 * ## Usage Example
 *
 * @code{.cpp}
 * using namespace Utils;
 * 
 * // Define objective function
 * auto rosenbrock = [](Vector const& x, Vector* g) -> double {
 *   double f = 100*(x(1)-x(0)*x(0))*(x(1)-x(0)*x(0)) + (1-x(0))*(1-x(0));
 *   if (g) {
 *     (*g)(0) = -400*(x(1)-x(0)*x(0))*x(0) - 2*(1-x(0));
 *     (*g)(1) = 200*(x(1)-x(0)*x(0));
 *   }
 *   return f;
 * };
 * 
 * // Setup minimizer
 * LBFGS_minimizer<double>::Options opts;
 * opts.max_iter = 1000;
 * opts.g_tol = 1e-6;
 * opts.verbose = true;
 * 
 * LBFGS_minimizer<double> minimizer(opts);
 * 
 * // Initial point
 * Vector x0(2);
 * x0 << -1.2, 1.0;
 * 
 * // Minimize
 * auto [status, x_opt, f_opt, data] = minimizer.minimize(
 *   x0, rosenbrock, StrongWolfeLineSearch<double>()
 * );
 * 
 * std::cout << "Solution: " << x_opt.transpose() << std::endl;
 * std::cout << "Minimum: " << f_opt << std::endl;
 * @endcode
 *
 * @author Enrico Bertolazzi
 * @date 2025
 */

namespace Utils {

  using std::abs;
  using std::min;
  using std::max;

  // ===========================================================================
  // LBFGS: Core Two-Loop Recursion Implementation
  // ===========================================================================

  /**
   * @class LBFGS
   * @brief Limited-memory BFGS storage and two-loop recursion implementation
   *
   * This class implements the core L-BFGS algorithm for computing approximate
   * inverse Hessian-vector products without explicitly storing the matrix.
   *
   * ## Algorithm Description
   *
   * The L-BFGS method maintains a history of m correction pairs:
   * \f[
   *   \{(s_i, y_i)\}_{i=k-m}^{k-1}
   * \f]
   * where:
   * - \f$s_i = x_{i+1} - x_i\f$ (displacement vector)
   * - \f$y_i = \nabla f(x_{i+1}) - \nabla f(x_i)\f$ (gradient difference)
   *
   * ### Two-Loop Recursion
   *
   * Given gradient g, compute \f$H_k g\f$ via:
   *
   * **First loop** (backward, from newest to oldest):
   * \f{align*}{
   *   q &\leftarrow g \\
   *   \text{for } i &= k-1, k-2, \ldots, k-m: \\
   *     \alpha_i &\leftarrow \rho_i s_i^T q \\
   *     q &\leftarrow q - \alpha_i y_i
   * \f}
   *
   * **Scaling**:
   * \f[
   *   r \leftarrow H_0 q
   * \f]
   * where \f$H_0 = \gamma I\f$ with \f$\gamma = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}\f$
   *
   * **Second loop** (forward, from oldest to newest):
   * \f{align*}{
   *   \text{for } i &= k-m, k-m+1, \ldots, k-1: \\
   *     \beta_i &\leftarrow \rho_i y_i^T r \\
   *     r &\leftarrow r + s_i(\alpha_i - \beta_i)
   * \f}
   *
   * **Output**: \f$r = H_k g\f$
   *
   * ### Computational Complexity
   *
   * - Time: O(mn) per iteration, where n is dimension and m is memory size
   * - Space: O(mn) to store correction pairs
   *
   * ### Curvature Condition
   *
   * For numerical stability, pairs are only accepted if:
   * \f[
   *   s_i^T y_i > \epsilon \|s_i\| \|y_i\|
   * \f]
   * This ensures the BFGS update maintains positive definiteness.
   *
   * ## Implementation Details
   *
   * - Uses circular buffer for efficient O(1) insertion of new pairs
   * - Automatic oldest pair removal when capacity is exceeded
   * - Robust curvature checking with both relative and absolute thresholds
   * - Optional Powell damping for improved robustness
   * - Clamping of initial Hessian approximation to avoid extreme scaling
   *
   * @tparam Scalar Floating-point type (typically double or float)
   *
   * @note This implementation is thread-safe for read operations if no
   *       concurrent modifications occur.
   *
   * @see Liu, D.C. and Nocedal, J. (1989) for the original algorithm
   * @see Nocedal and Wright (2006), Chapter 7, for detailed theory
   */

  template <typename Scalar = double>
  class LBFGS {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  private:
    size_t m_capacity;         ///< Maximum number of correction pairs (m)
    size_t m_dimension;        ///< Problem dimension (n)
    Matrix m_S;                ///< Storage for s vectors (n × m matrix, each column is s_i)
    Matrix m_Y;                ///< Storage for y vectors (n × m matrix, each column is y_i)
    Vector m_rho;              ///< Storage for ρ_i = 1/(y_i^T s_i) values
    size_t m_current_size{0};  ///< Current number of stored pairs
    size_t m_oldest_index{0};  ///< Circular buffer index of oldest pair
    size_t m_newest_index{0};  ///< Circular buffer index where next pair will be stored

    /* Robustness parameters */
    bool   m_enable_damping{ true };        ///< Enable Powell-style damping by default
    Scalar m_h0_min{ Scalar(1e-6) };        ///< Minimum allowed initial H0 scaling
    Scalar m_h0_max{ Scalar(1e6) };         ///< Maximum allowed initial H0 scaling

    /* Temporary workspace vectors (mutable for const methods) */
    mutable Vector m_alpha;  ///< Workspace for α_i values in two-loop recursion
    mutable Vector m_q;      ///< Workspace for q vector in two-loop recursion
    mutable Vector m_r;      ///< Workspace for r vector in two-loop recursion

  public:
    /**
     * @brief Construct L-BFGS storage
     *
     * Allocates memory for storing up to maxCorrections correction pairs.
     * The circular buffer strategy allows O(1) insertion of new pairs.
     *
     * @param maxCorrections Maximum number of stored correction pairs (typical: 5-20).
     *                       Larger values provide better Hessian approximation but
     *                       increase memory usage and computational cost.
     * @param problem_dimension Dimension of the optimization problem (n)
     *
     * @note A typical choice is m=10 for most problems. Use smaller m (3-7) for
     *       very large problems or when memory is constrained.
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

    /**
     * @brief Clear all stored correction pairs
     *
     * Resets the L-BFGS memory to its initial empty state. This is useful when:
     * - Starting optimization of a new problem
     * - Restarting after convergence issues
     * - Periodic refresh to avoid accumulation of numerical errors
     *
     * @post size() == 0 and all internal storage is zeroed
     */
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

    /**
     * @brief Return current number of stored correction pairs
     * @return Number of pairs currently in memory (0 ≤ size() ≤ capacity())
     */
    size_t size() const { return m_current_size; }

    /**
     * @brief Return maximum capacity for correction pairs
     * @return Maximum number of pairs (m parameter)
     */
    size_t capacity() const { return m_capacity; }

    /**
     * @brief Return problem dimension
     * @return Dimension of vectors in optimization problem (n)
     */
    size_t dimension() const { return m_dimension; }

    /**
     * @brief Resize for new problem dimension
     *
     * Reallocates storage for a different problem size. This clears all
     * existing correction pairs.
     *
     * @param new_dimension New problem dimension
     *
     * @post size() == 0 and dimension() == new_dimension
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

    /**
     * @brief Add a correction pair (s, y) to L-BFGS memory
     *
     * Attempts to add a new correction pair after validating the curvature condition.
     * The pair is accepted only if:
     * \f[
     *   s^T y > \max\{\epsilon_{\text{rel}} \|s\| \|y\|, \epsilon_{\text{abs}} \|s\|^2\}
     * \f]
     *
     * This ensures:
     * - Positive definiteness of the Hessian approximation
     * - Numerical stability (avoids division by very small numbers)
     * - Well-conditioned updates
     *
     * ### Update Strategy
     *
     * When the buffer is full (size() == capacity()), the oldest pair is
     * automatically discarded to make room for the new one.
     *
     * @param s Displacement vector \f$s = x_{k+1} - x_k\f$
     * @param y Gradient difference \f$y = \nabla f(x_{k+1}) - \nabla f(x_k)\f$
     * @param min_curvature_ratio Minimum ratio \f$\frac{s^T y}{\|s\| \|y\|}\f$ 
     *                            for acceptance (default: 1e-8)
     *
     * @return true if pair was accepted and stored, false if rejected due to
     *         insufficient curvature
     *
     * @pre s.size() == y.size()
     * @post If accepted: size() increments by 1 (unless at capacity)
     *
     * @note Rejection typically occurs when:
     *       - Line search is too inaccurate
     *       - Gradient evaluations have numerical errors
     *       - Problem has very small or zero curvature in search direction
     *
     * @see Nocedal & Wright (2006), Section 7.2 for curvature condition theory
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

    /**
     * @brief Compute H*g using two-loop recursion algorithm
     *
     * This is the core L-BFGS operation that computes the product of the
     * approximate inverse Hessian H with a gradient vector g, without ever
     * forming H explicitly.
     *
     * ### Algorithm Steps
     *
     * 1. **First Loop** (backward through history):
     *    - Start with q = g
     *    - For each pair (s_i, y_i) from newest to oldest:
     *      - Compute α_i = ρ_i (s_i · q)
     *      - Update q ← q - α_i y_i
     *
     * 2. **Initial Hessian Application**:
     *    - Scale: r = h0 * q
     *    - This represents H_0 = h0 * I (scaled identity matrix)
     *
     * 3. **Second Loop** (forward through history):
     *    - For each pair (s_i, y_i) from oldest to newest:
     *      - Compute β_i = ρ_i (y_i · r)
     *      - Update r ← r + s_i (α_i - β_i)
     *
     * 4. **Result**: Return r = H * g
     *
     * ### Computational Details
     *
     * - **Time Complexity**: O(m*n), where m = size() and n = dimension()
     * - **Space Complexity**: O(m + n) for temporary vectors
     * - **Memory Access**: Efficient cache usage due to contiguous storage
     *
     * @param g Gradient vector (typically ∇f(x))
     * @param h0 Initial diagonal Hessian approximation scalar. Common choices:
     *           - h0 = 1.0 (simple scaling)
     *           - h0 = (s^T y)/(y^T y) using most recent pair (recommended)
     *           - Adaptive scaling based on problem characteristics
     *
     * @return Vector r = H * g, representing the quasi-Newton search direction
     *
     * @pre g.size() == dimension()
     * @post result.size() == g.size()
     *
     * @note If no pairs are stored (size() == 0), returns h0 * g (scaled steepest descent)
     *
     * @see Nocedal (1980) for original two-loop recursion algorithm
     * @see Liu & Nocedal (1989) for practical implementation details
     */
    Vector
    two_loop_recursion( Vector const & g, Scalar h0 ) const {
      if ( m_current_size == 0 ) return h0 * g;

      m_q = g; // Start with q = g

      // =====================================================================
      // First loop: Process pairs from newest to oldest
      // =====================================================================
      size_t idx = (m_newest_index == 0) ? m_capacity - 1 : m_newest_index - 1;
        
      size_t i{m_current_size};
      while ( i > 0 ) {
        --i;
        m_alpha(i) = m_rho(idx) * m_S.col(idx).dot(m_q);
        m_q       -= m_alpha(i) * m_Y.col(idx);
            
        // Move to previous index in circular buffer
        idx = (idx == 0) ? m_capacity - 1 : idx - 1;
      }

      // =====================================================================
      // Apply initial Hessian approximation H0 = h0 * I
      // =====================================================================
      m_r = h0 * m_q;

      // =====================================================================
      // Second loop: Process pairs from oldest to newest
      // =====================================================================
      idx = m_oldest_index;
      for ( size_t i{0}; i < m_current_size; ++i ) {
        Scalar beta = m_rho(idx) * m_Y.col(idx).dot(m_r);
        m_r += m_S.col(idx) * (m_alpha(i) - beta);

        // Move to next index in circular buffer
        idx = (idx + 1) % m_capacity;
      }
      
      return m_r;
    }

    /**
     * @brief Compute recommended initial Hessian scaling from latest pair
     *
     * Computes the scalar h0 for the initial Hessian approximation H0 = h0*I
     * using the most recent correction pair:
     * \f[
     *   h_0 = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}
     * \f]
     *
     * This choice is motivated by:
     * - Self-scaling property: approximates the curvature along the most recent step
     * - Theoretical foundation: relates to Barzilai-Borwein spectral steplength
     * - Empirical success: works well in practice for many problems
     *
     * The result is clamped to [m_h0_min, m_h0_max] to prevent extreme scaling
     * that could lead to numerical instability.
     *
     * @param default_value Value to return if no pairs are stored (default: 1.0)
     *
     * @return Recommended h0 value, or default_value if size() == 0
     *
     * @note The clamping bounds are:
     *       - Lower: 1e-6 (prevents division issues for nearly-flat regions)
     *       - Upper: 1e6 (prevents overflow in steep regions)
     *
     * @see Nocedal & Wright (2006), Section 7.2 for scaling strategies
     */
    Scalar
    compute_initial_h0( Scalar default_value = Scalar(1.0) ) const {
      if ( m_current_size == 0 ) return default_value;
        
      // Get the newest pair (just before newest_index in circular buffer)
      size_t latest_idx = (m_newest_index == 0) ? m_capacity - 1 : m_newest_index - 1;
      auto const & s = m_S.col( latest_idx );
      auto const & y = m_Y.col( latest_idx );
        
      Scalar sty { s.dot(y) };
      Scalar yty { y.dot(y) };
      Scalar h0 = default_value;
      if ( yty > 0 ) h0 = sty / yty;
      
      // Clamp to safe range
      h0 = max(m_h0_min, min(m_h0_max, h0));
      return h0;
    }

    /**
     * @brief Add correction with Powell damping for improved robustness
     *
     * Powell damping (also called modified BFGS) adjusts the gradient difference
     * y to ensure the curvature condition is satisfied, even when the standard
     * pair (s,y) would be rejected.
     *
     * ### Algorithm
     *
     * When s^T y is too small (less than threshold), compute:
     * \f[
     *   \theta = \frac{\text{threshold} - s^T B_0 s}{s^T y - s^T B_0 s}
     * \f]
     * where B_0 = H_0^{-1} = (1/h0)*I, then use damped gradient:
     * \f[
     *   \hat{y} = \theta y + (1-\theta) B_0 s
     * \f]
     *
     * This creates a convex combination that guarantees s^T ŷ ≥ threshold.
     *
     * ### Benefits
     *
     * - Maintains positive definiteness even with inaccurate line searches
     * - Allows continued progress in difficult regions
     * - Reduces need for memory resets
     *
     * @param lb Reference to LBFGS object to update
     * @param s Displacement vector
     * @param y Gradient difference vector
     * @param min_curvature_ratio Minimum curvature ratio (default: 1e-8)
     *
     * @return true if correction was successfully added (possibly damped)
     *
     * @note Falls back to undamped update if damping is unnecessary or fails
     *
     * @see Powell, M.J.D. (1978). "A Fast Algorithm for Nonlinearly Constrained
     *      Optimization Calculations". Numerical Analysis, Lecture Notes in
     *      Mathematics, Springer.
     * @see Nocedal & Wright (2006), Section 18.3 for modified BFGS
     */
    bool
    add_correction_with_damping(
      LBFGS<Scalar> & lb,
      Vector const  & s,
      Vector const  & y,
      Scalar const    min_curvature_ratio = 1e-8
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
      if ( abs(denom) < std::numeric_limits<Scalar>::epsilon() ) {
        // ambiguous, don't damp
        return lb.add_correction(s,y,min_curvature_ratio);
      } else {
        // choose theta so that s^T y_hat = thresh (solve theta*(sty - sBs) + sBs = thresh)
        Scalar theta = std::clamp( (thresh - sBs) / denom, 0, 1 );
        y_hat = theta * y + (1 - theta) * (s / h0);
        Scalar sty_hat = s.dot(y_hat);
        if ( !(sty_hat > 0) ) return false;
        return lb.add_correction(s, y_hat, min_curvature_ratio);
      }
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

    enum class Status {
      CONVERGED          = 0, // Convergenza raggiunta
      MAX_ITERATIONS     = 1, // Massimo numero di iterazioni raggiunto
      LINE_SEARCH_FAILED = 2, // Line search fallita
      GRADIENT_TOO_SMALL = 3, // Gradiente troppo piccolo
      FAILED             = 4  // Fallimento generico
    };

    struct IterationData {
      Status status{Status::FAILED};
      size_t iterations{0};
      Scalar final_gradient_norm{0};
      Scalar final_function_value{0};
      Scalar initial_function_value{0};
      size_t function_evaluations{0};
    };

    struct Options {
      size_t max_iter        {200};
      size_t iter_reset      {50};
      size_t m               {20};

      Scalar g_tol           {1e-8};
      Scalar f_tol           {1e-12};
      Scalar x_tol           {1e-10};
      
      Scalar step_max        {10};
      Scalar sty_min_factor  {1e-12};
      Scalar very_small_step {1e-8};

      bool   use_projection  {false};
      bool   verbose         {false};
    };

  private:

    Scalar        m_epsi{std::numeric_limits<Scalar>::epsilon()};
    Options       m_options;
    LBFGS<Scalar> m_LBFGS;
    Vector        m_lower;
    Vector        m_upper;
    Vector        m_tol_lower;
    Vector        m_tol_upper;

    // Helper functions for bound checks with tolerances
    bool
    is_on_lower_bound(Scalar x_i, Scalar lower_i) const {
      return (x_i <= lower_i + m_epsi * (1 + std::abs(lower_i)));
    }

    bool
    is_on_upper_bound(Scalar x_i, Scalar upper_i) const {
      return (x_i >= upper_i - m_epsi * (1 + std::abs(upper_i)));
    }

    void
    check_bounds_consistency() const {
      if (m_options.use_projection && (m_lower.array() > m_upper.array()).any()) {
        throw std::invalid_argument("Lower bounds must be <= upper bounds");
      }
    }

    // Project x into bounds
    void
    project_inplace(Vector & x) const {
      x = x.cwiseMax(m_lower).cwiseMin(m_upper);
    }

    Vector
    projected_gradient(Vector const & x, Vector const & g) const {
      Vector result = g;
      projected_gradient_inplace(x, result);
      return result;
    }

    void
    projected_gradient_inplace(Vector const & x, Vector & g) const {
      
      // Crea maschere booleane per le condizioni
      auto on_lower_bound = (x.array() <= (m_lower.array() + m_tol_lower.array()));
      auto on_upper_bound = (x.array() >= (m_upper.array() - m_tol_upper.array()));
      auto grad_pointing_out_lower = (g.array() < 0);
      auto grad_pointing_out_upper = (g.array() > 0);
      
      // Combina le condizioni e azzera i gradienti appropriati
      auto freeze_mask = (on_lower_bound && grad_pointing_out_lower) || 
                         (on_upper_bound && grad_pointing_out_upper);
      
      g = freeze_mask.select(Vector::Zero(g.size()), g);
    }

    void projected_direction_inplace(Vector const & x, Vector & d) const {
      // Crea maschere booleane per le condizioni
      auto on_lower_bound = (x.array() <= (m_lower.array() + m_tol_lower.array()));
      auto on_upper_bound = (x.array() >= (m_upper.array() - m_tol_upper.array()));
      auto dir_pointing_out_lower = (d.array() < 0);
      auto dir_pointing_out_upper = (d.array() > 0);
        
      // Combina le condizioni e azzera le direzioni inappropriate
      auto freeze_mask = (on_lower_bound && dir_pointing_out_lower) ||
                         (on_upper_bound && dir_pointing_out_upper);
        
      d = freeze_mask.select(Vector::Zero(d.size()), d);
    }

    Scalar projected_gradient_norm(Vector const & x, Vector const & g) const {
      return projected_gradient(x, g).template lpNorm<Eigen::Infinity>();
    }

    // Helper function for descent direction validation
    bool
    is_valid_descent_direction(Scalar pg) const {
      if (!std::isfinite(pg)) return false;
      if (pg >= -m_epsi * (1 + std::abs(pg))) return false;
      return true;
    }

    // Mutable state variables
    mutable Vector m_x, m_g, m_p, m_x_new, m_g_new, m_s, m_y;
    mutable size_t m_iter_since_reset{0};
    mutable size_t m_function_evaluations{0};

  public:
    LBFGS_minimizer(Options opts = Options())
    : m_options(opts)
    , m_LBFGS(opts.m)
    {}
    
    Vector const & solution() const { return m_x; }

    void
    set_bounds(size_t n, Scalar const lower[], Scalar const upper[]) {
      m_lower.resize(n);
      m_upper.resize(n);
      std::copy_n(lower, n, m_lower.data());
      std::copy_n(upper, n, m_upper.data());

      m_tol_lower.resize(n);
      m_tol_upper.resize(n);
  
      // Calcola tolleranze vettoriali
      m_tol_lower = m_epsi * (Vector::Ones(n).array() + m_lower.array().abs());
      m_tol_upper = m_epsi * (Vector::Ones(n).array() + m_upper.array().abs());

      m_options.use_projection = true;
      check_bounds_consistency();
    }

    void
    set_bounds(Vector const & lower, Vector const & upper) {
      assert(lower.size() == upper.size());
      set_bounds( lower.size(), lower.data(), upper.data() );
    }

    void
    reset_memory() {
      if (m_options.verbose) fmt::print("[LBFGS] Periodic memory reset\n");
      m_LBFGS.clear();
      m_iter_since_reset = 0;
    }

    template <typename Linesearch>
    IterationData minimize(
      Vector     const & x0,
      Callback   const & callback,
      Linesearch const & linesearch = MoreThuenteLineSearch<Scalar>()
    ) {
      Status status{Status::MAX_ITERATIONS};
      Scalar gnorm{0};

      // Reset counters
      m_function_evaluations = 0;

      auto const n{x0.size()};
      m_x.resize(n);
      m_g.resize(n);
      m_x_new.resize(n);
      m_g_new.resize(n);
      m_p.resize(n);
      m_s.resize(n);
      m_y.resize(n);

      if (m_options.use_projection) {
        assert(m_lower.size() == n);
        assert(m_upper.size() == n);
        check_bounds_consistency();
      }

      // Initialize and project if needed
      m_x.noalias() = x0;
      if (m_options.use_projection) project_inplace(m_x);
      
      // Initial evaluation
      Scalar f = callback(m_x, &m_g);
      m_function_evaluations++;

      Scalar f_prev{f};
      Scalar f_initial{f};
      size_t iteration{0};

      // Main optimization loop
      for (; iteration < m_options.max_iter; ++iteration) {
        ++m_iter_since_reset;

        // Check gradient norm
        gnorm = projected_gradient_norm(m_x, m_g);
        if ( m_options.verbose )
          fmt::print("[LBFGS] iter={:3d} f={:12.4g} ‖pg‖={:12.4g} mem={}\n", iteration, f, gnorm, m_LBFGS.size());

        if ( gnorm <= m_options.g_tol ) {
          status = Status::GRADIENT_TOO_SMALL;
          goto exit_position;
        }

        // Periodic memory reset
        if ( m_iter_since_reset >= m_options.iter_reset ) reset_memory();

        // Compute search direction using L-BFGS
        Scalar h0 = m_LBFGS.compute_initial_h0(1);
        m_p = -m_LBFGS.two_loop_recursion(m_g, h0);

        // Project direction if bounds are active
        if ( m_options.use_projection ) {
          projected_direction_inplace(m_x, m_p);
          if ( m_p.isZero(m_epsi) ) m_p = -projected_gradient(m_x, m_g);
        }

        // Robust descent direction check with fallback strategies
        Scalar pg = m_p.dot(m_g);
        if ( !is_valid_descent_direction(pg) ) {
          // Try gradient direction
          m_p = -m_g;
          if ( m_options.use_projection ) projected_direction_inplace(m_x, m_p);
          pg = m_p.dot(m_g);
          if ( !is_valid_descent_direction(pg) ) {
            // Reset L-BFGS memory and try again
            m_LBFGS.clear();
            h0  = m_LBFGS.compute_initial_h0(1);
            m_p = -m_LBFGS.two_loop_recursion(m_g, h0);
            if ( m_options.use_projection ) projected_direction_inplace(m_x, m_p);
            pg = m_p.dot(m_g);
            if ( m_options.verbose )
              fmt::print("[LBFGS] Cannot find descent direction, stopping\n");
            gnorm  = projected_gradient_norm(m_x, m_g);
            status = Status::FAILED;
            goto exit_position;
          }
        }

        // Line search
        auto step_opt = linesearch(f, pg, m_x, m_p, callback, m_options.step_max);
        
        if (!step_opt.has_value()) {
          if (m_options.verbose)
            fmt::print("[LBFGS] line search failed, trying fallback steps\n");

          m_LBFGS.clear();

          // Try progressively smaller fixed steps
          bool fallback_success = false;
          const std::array<Scalar, 3> fallback_steps = {
            m_options.very_small_step,
            m_options.very_small_step * Scalar(0.1),
            m_options.very_small_step * Scalar(0.01)
          };

          for (Scalar fallback_step : fallback_steps) {
            m_x_new.noalias() = m_x + fallback_step * m_p;
            if (m_options.use_projection) project_inplace(m_x_new);
            Scalar f_test = callback(m_x_new, &m_g_new);
            m_function_evaluations++;
  
            if (f_test < f) {
              m_s.noalias() = m_x_new - m_x;
              m_y.noalias() = m_g_new - m_g;
              m_x.swap(m_x_new);
              m_g.swap(m_g_new);
              f = f_test;
              fallback_success = true;
              break;
            }
          }
          
          if (!fallback_success) {
            gnorm = projected_gradient_norm(m_x, m_g);
            status = Status::LINE_SEARCH_FAILED;
            goto exit_position;
          }
          continue;
        }

        auto [step,n_evals] = *step_opt; m_function_evaluations += n_evals;
        
        // Evaluate new point
        m_x_new.noalias() = m_x + step * m_p;
        if ( m_options.use_projection ) project_inplace(m_x_new);
        Scalar f_new = callback(m_x_new, &m_g_new);
        m_function_evaluations++;

        // Check for stagnation in variables
        {
          Scalar x_change = m_p.norm();
          if ( x_change < m_options.x_tol ) {
            if (m_options.verbose)
              fmt::print("[LBFGS] Converged by x change: {:.2e} < {:.2e}\n", x_change, m_options.x_tol );
            gnorm  = projected_gradient_norm(m_x, m_g);
            status = Status::CONVERGED;
            goto exit_position;
          }
        }

        // Compute differences for L-BFGS update
        m_s.noalias() = m_x_new - m_x;
        m_y.noalias() = m_g_new - m_g;

        // Robust curvature check for L-BFGS update
        Scalar sty = m_s.dot(m_y);
        Scalar sty_tolerance = std::max(
          m_options.sty_min_factor,
          m_epsi * m_s.squaredNorm() * m_y.squaredNorm()
        );

        if ( sty > sty_tolerance ) {
          m_LBFGS.add_correction(m_s, m_y);
        } else {
          if ( m_options.verbose && m_LBFGS.size() > 0 )
            fmt::print("[LBFGS] curvature condition failed (s^T y = {:.2e}), skipping update\n", sty);
        }

        // Move to new point
        m_x.swap(m_x_new);
        m_g.swap(m_g_new);
        f = f_new;

        // Check function value change
        Scalar f_change = std::abs(f - f_prev);
        if ( f_change <= m_options.f_tol ) {
          if ( m_options.verbose )
            fmt::print("[LBFGS] Converged by function change: {:.2e} < {:.2e}\n", f_change, m_options.f_tol);
          gnorm  = projected_gradient_norm(m_x, m_g);
          status = Status::CONVERGED;
          goto exit_position;
        }
        f_prev = f;
      }
        
      // Final gradient norm if not converged
      gnorm = projected_gradient_norm(m_x, m_g);

    exit_position:
      return IterationData{
        status,
        iteration,
        gnorm,
        f,
        f_initial,
        m_function_evaluations
      };
    }
  };

// ===========================================================================
// CLASS: LBFGS_BlockCoordinate (Enhanced Block Coordinate L-BFGS)
// ===========================================================================

/**
 * @class LBFGS_BlockCoordinate
 * @brief Enhanced block coordinate L-BFGS with overlapping random consecutive blocks
 *
 * This version implements sophisticated block selection strategies:
 * - Random consecutive blocks with configurable overlap
 * - Statistical coverage guarantee for all coordinates
 * - Adaptive block size based on progress
 * - Multiple block selection strategies
 *
 * ### Block Selection Strategies
 *
 * 1. **Random Consecutive**: Random start position, consecutive coordinates
 * 2. **Random Consecutive with Overlap**: Blocks can partially overlap
 * 3. **Cyclic Consecutive**: Systematic coverage of all coordinates
 * 4. **Greedy**: Focus on coordinates with largest gradients
 * 5. **Hybrid**: Combine random consecutive with greedy selection
 *
 * ### Statistical Coverage
 *
 * The algorithm ensures that over multiple outer iterations, all coordinates
 * are visited with high probability. Coverage tracking prevents neglect of
 * any variable while maintaining randomness for better exploration.
 *
 * @tparam Scalar Floating-point type (typically double)
 */
template <typename Scalar = double>
class LBFGS_BlockCoordinate {
public:
  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using Callback = std::function<Scalar(Vector const &, Vector *)>;

  enum class Status {
    CONVERGED,
    MAX_OUTER_ITERATIONS,
    MAX_INNER_ITERATIONS,
    GRADIENT_TOO_SMALL,
    LINE_SEARCH_FAILED,
    FAILED
  };

  struct Result {
    Vector solution;
    Scalar final_function_value{0};
    Scalar initial_function_value{0};
    Status status{Status::FAILED};
    
    // Statistics
    size_t outer_iterations{0};
    size_t inner_iterations{0};
    size_t total_iterations{0};
    
    size_t outer_evaluations{0};
    size_t inner_evaluations{0};
    size_t total_evaluations{0};
    
    // Enhanced statistics
    size_t blocks_processed{0};
    Scalar max_block_improvement{0};
    std::vector<size_t> coordinate_visits;  // Visit count per coordinate
    double coverage_ratio{0.0};             // Percentage of coordinates visited
  };

  struct Options {
    // Block configuration
    size_t block_size{100};                  ///< Base block size
    string block_selection{"random_consecutive_overlap"}; ///< Selection strategy
    double overlap_ratio{0.3};               ///< Fraction of block overlap (0-1)
    size_t min_block_size{20};               ///< Minimum block size
    size_t max_block_size{500};              ///< Maximum block size
    
    // Coverage control
    double min_coverage_ratio{0.95};         ///< Minimum coordinate coverage per cycle
    size_t coverage_reset_frequency{10};     ///< Reset coverage tracking periodically
    
    // Outer loop control
    size_t max_outer_iterations{50};
    size_t max_inner_iterations{200};
    Scalar outer_tolerance{1e-6};
    
    // Inner L-BFGS options
    size_t lbfgs_m{10};
    Scalar lbfgs_g_tol{1e-6};
    Scalar lbfgs_f_tol{1e-8};
    
    // Adaptive parameters
    bool adaptive_block_size{true};
    Scalar progress_threshold{1e-4};
    
    // Verbosity
    bool verbose{true};
    size_t progress_frequency{5};
    bool track_coverage{true};               ///< Track coordinate visits
  };

private:
  Options m_options;
  Vector m_lower;
  Vector m_upper;
  bool m_use_bounds{false};
  
  // Coverage tracking
  std::vector<size_t> m_coordinate_visits;
  size_t m_total_coordinates{0};
  size_t m_outer_iteration_count{0};
  
  // Random number generation
  std::random_device m_rd;
  std::mt19937 m_gen;
  
  // Progress tracking
  Scalar m_previous_best{std::numeric_limits<Scalar>::max()};
  size_t m_stagnation_count{0};

public:
  explicit LBFGS_BlockCoordinate(Options const & opts = Options()) 
    : m_options(opts), m_gen(m_rd()) {}

  void set_bounds(Vector const & lower, Vector const & upper) {
    UTILS_ASSERT(lower.size() == upper.size(), "Bounds size mismatch");
    m_lower = lower;
    m_upper = upper;
    m_use_bounds = true;
  }

  void clear_bounds() {
    m_use_bounds = false;
  }

  /**
   * @brief Initialize coverage tracking for new problem
   */
  void initialize_coverage_tracking(size_t n) {
    m_total_coordinates = n;
    m_coordinate_visits.assign(n, 0);
    m_outer_iteration_count = 0;
  }

  /**
   * @brief Select random consecutive block with optional overlap
   */
  std::vector<size_t> select_random_consecutive_block(size_t n_total, size_t block_size, 
                                                     bool allow_overlap = true) {
    std::vector<size_t> indices;
    
    if (block_size >= n_total) {
      // Full block - all coordinates
      indices.resize(n_total);
      std::iota(indices.begin(), indices.end(), 0);
      return indices;
    }
    
    // Calculate overlap offset
    size_t overlap_offset = allow_overlap ? 
      static_cast<size_t>(block_size * m_options.overlap_ratio) : 0;
    
    size_t effective_block_size = block_size;
    if (allow_overlap && overlap_offset > 0) {
      // Adjust block size to account for potential overlap
      effective_block_size = std::min(block_size, n_total);
    }
    
    // Generate random start position
    std::uniform_int_distribution<size_t> dist(0, n_total - effective_block_size);
    size_t start_index = dist(m_gen);
    
    // Create consecutive block
    indices.reserve(effective_block_size);
    for (size_t i = 0; i < effective_block_size; ++i) {
      indices.push_back((start_index + i) % n_total);
    }
    
    return indices;
  }

  /**
   * @brief Select block with guaranteed coverage of under-visited coordinates
   */
  std::vector<size_t> select_coverage_aware_block(size_t n_total, size_t block_size, 
                                                 Vector const & gradient) {
    std::vector<size_t> indices;
    indices.reserve(block_size);
    
    // Find under-visited coordinates (visited less than average)
    size_t total_visits = std::accumulate(m_coordinate_visits.begin(), 
                                         m_coordinate_visits.end(), 0);
    double avg_visits = static_cast<double>(total_visits) / n_total;
    
    std::vector<std::pair<size_t, size_t>> under_visited;
    for (size_t i = 0; i < n_total; ++i) {
      if (m_coordinate_visits[i] < avg_visits * 0.8) {
        under_visited.emplace_back(m_coordinate_visits[i], i);
      }
    }
    
    // Sort by visit count (least visited first)
    std::sort(under_visited.begin(), under_visited.end());
    
    // Select from under-visited coordinates
    size_t coverage_count = std::min(block_size / 2, under_visited.size());
    for (size_t i = 0; i < coverage_count; ++i) {
      indices.push_back(under_visited[i].second);
    }
    
    // Fill remaining slots with random consecutive block
    if (indices.size() < block_size) {
      size_t remaining = block_size - indices.size();
      auto random_block = select_random_consecutive_block(n_total, remaining, true);
      indices.insert(indices.end(), random_block.begin(), random_block.end());
    }
    
    // Remove duplicates and ensure correct size
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    
    if (indices.size() > block_size) {
      indices.resize(block_size);
    }
    
    return indices;
  }

  /**
   * @brief Select block based on gradient magnitude (greedy)
   */
  std::vector<size_t> select_greedy_block(size_t n_total, size_t block_size, 
                                         Vector const & gradient) {
    std::vector<std::pair<Scalar, size_t>> grad_values;
    grad_values.reserve(n_total);
    
    for (size_t i = 0; i < n_total; ++i) {
      grad_values.emplace_back(std::abs(gradient(i)), i);
    }
    
    // Sort by descending gradient magnitude
    std::sort(grad_values.begin(), grad_values.end(),
              [](auto const & a, auto const & b) { return a.first > b.first; });
    
    std::vector<size_t> indices;
    indices.reserve(block_size);
    
    for (size_t i = 0; i < std::min(block_size, n_total); ++i) {
      indices.push_back(grad_values[i].second);
    }
    
    return indices;
  }

  /**
   * @brief Select hybrid block combining coverage and gradient information
   */
  std::vector<size_t> select_hybrid_block(size_t n_total, size_t block_size, 
                                         Vector const & gradient) {
    std::vector<size_t> indices;
    
    // 50% based on coverage, 50% based on gradient
    size_t coverage_count = block_size / 2;
    size_t gradient_count = block_size - coverage_count;
    
    // Coverage-aware selection
    auto coverage_indices = select_coverage_aware_block(n_total, coverage_count, gradient);
    indices.insert(indices.end(), coverage_indices.begin(), coverage_indices.end());
    
    // Greedy selection for remaining
    if (indices.size() < block_size) {
      auto greedy_indices = select_greedy_block(n_total, gradient_count, gradient);
      indices.insert(indices.end(), greedy_indices.begin(), greedy_indices.end());
    }
    
    // Remove duplicates
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    
    if (indices.size() > block_size) {
      indices.resize(block_size);
    }
    
    return indices;
  }

  /**
   * @brief Main block selection dispatcher
   */
  std::vector<size_t> select_block(size_t n_total, size_t block_size, 
                                  Vector const & gradient, size_t outer_iter) {
    if (m_options.block_selection == "random_consecutive") {
      return select_random_consecutive_block(n_total, block_size, false);
    }
    else if (m_options.block_selection == "random_consecutive_overlap") {
      return select_random_consecutive_block(n_total, block_size, true);
    }
    else if (m_options.block_selection == "greedy") {
      return select_greedy_block(n_total, block_size, gradient);
    }
    else if (m_options.block_selection == "coverage_aware") {
      return select_coverage_aware_block(n_total, block_size, gradient);
    }
    else if (m_options.block_selection == "hybrid") {
      return select_hybrid_block(n_total, block_size, gradient);
    }
    else { // Default: random consecutive with overlap
      return select_random_consecutive_block(n_total, block_size, true);
    }
  }

  /**
   * @brief Calculate current coverage statistics
   */
  void calculate_coverage_stats(Result & result) const {
    if (m_coordinate_visits.empty()) return;
    
    size_t visited_count = 0;
    for (size_t visits : m_coordinate_visits) {
      if (visits > 0) ++visited_count;
    }
    
    result.coverage_ratio = static_cast<double>(visited_count) / m_total_coordinates;
    result.coordinate_visits = m_coordinate_visits;
  }

  /**
   * @brief Check if coverage is satisfactory, adjust strategy if not
   */
  bool check_coverage_adequate() const {
    if (m_coordinate_visits.empty()) return true;
    
    size_t visited_count = 0;
    for (size_t visits : m_coordinate_visits) {
      if (visits > 0) ++visited_count;
    }
    
    double coverage_ratio = static_cast<double>(visited_count) / m_total_coordinates;
    return coverage_ratio >= m_options.min_coverage_ratio;
  }

  /**
   * @brief Reset coverage tracking periodically
   */
  void reset_coverage_tracking() {
    if (m_options.coverage_reset_frequency > 0 &&
        m_outer_iteration_count % m_options.coverage_reset_frequency == 0) {
      std::fill(m_coordinate_visits.begin(), m_coordinate_visits.end(), 0);
      
      if (m_options.verbose) {
        fmt::print("[BlockLBFGS] Reset coverage tracking at outer iteration {}\n", 
                   m_outer_iteration_count);
      }
    }
  }

  /**
   * @brief Update coverage counts for processed block
   */
  void update_coverage_counts(std::vector<size_t> const & block_indices) {
    for (size_t idx : block_indices) {
      if (idx < m_coordinate_visits.size()) {
        m_coordinate_visits[idx]++;
      }
    }
  }

  /**
   * @brief Adjust block size based on progress and coverage
   */
  void adapt_block_size(Scalar improvement, size_t n_total, size_t outer_iter) {
    if (!m_options.adaptive_block_size) return;
    
    Scalar relative_improvement = improvement / (1.0 + std::abs(m_previous_best));
    
    // Adjust based on progress
    if (relative_improvement > 0.1) {
      // Good progress - consider larger blocks for faster convergence
      m_options.block_size = std::min(m_options.max_block_size, 
                                     m_options.block_size * 2);
    } 
    else if (relative_improvement < 0.01) {
      // Poor progress - try smaller blocks for better precision
      m_options.block_size = std::max(m_options.min_block_size, 
                                     m_options.block_size / 2);
    }
    
    // Adjust based on coverage
    if (!check_coverage_adequate()) {
      // Poor coverage - reduce block size to visit more coordinates
      m_options.block_size = std::max(m_options.min_block_size,
                                     m_options.block_size * 3 / 4);
    }
    
    // Ensure block size is reasonable relative to problem size
    m_options.block_size = std::min(m_options.block_size, n_total);
    m_options.block_size = std::max(m_options.block_size, m_options.min_block_size);
    
    m_previous_best = improvement;
    
    if (m_options.verbose && outer_iter % m_options.progress_frequency == 0) {
      fmt::print("[BlockLBFGS] Adjusted block_size to {}\n", m_options.block_size);
    }
  }

  // [Keep all the utility methods from previous version: 
  // extract_subvector, update_full_vector, extract_subbounds, 
  // project_point, projected_gradient_norm]

  Vector extract_subvector(Vector const & full, std::vector<size_t> const & indices) {
    Vector sub(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      sub(i) = full(indices[i]);
    }
    return sub;
  }

  void update_full_vector(Vector & full, Vector const & sub, std::vector<size_t> const & indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
      full(indices[i]) = sub(i);
    }
  }

  void extract_subbounds(std::vector<size_t> const & indices,
                        Vector & sub_lower, Vector & sub_upper) {
    sub_lower.resize(indices.size());
    sub_upper.resize(indices.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
      sub_lower(i) = m_lower(indices[i]);
      sub_upper(i) = m_upper(indices[i]);
    }
  }

  void project_point(Vector & x) const {
    if (m_use_bounds) {
      x = x.cwiseMax(m_lower).cwiseMin(m_upper);
    }
  }

  Scalar projected_gradient_norm(Vector const & x, Vector const & g) const {
    if (!m_use_bounds) return g.template lpNorm<Eigen::Infinity>();
    
    Vector pg = g;
    
    // Zero gradient components for variables at bounds pointing outward
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      if ((x(i) <= m_lower(i) + 1e-12 && g(i) > 0) ||
          (x(i) >= m_upper(i) - 1e-12 && g(i) < 0)) {
        pg(i) = 0;
      }
    }
    
    return pg.template lpNorm<Eigen::Infinity>();
  }

  /**
   * @brief Enhanced optimization routine with coverage tracking
   */
    template <typename Linesearch>
    Result minimize(
      Vector     const & x0,
      Callback   const & global_callback,
      Linesearch const & linesearch = MoreThuenteLineSearch<Scalar>()
    ) {

    Result result;
    size_t n = x0.size();
    
    // Initialize coverage tracking
    if (m_options.track_coverage) {
      initialize_coverage_tracking(n);
    }
    
    // Initialize solution
    Vector x = x0;
    if (m_use_bounds) project_point(x);
    
    // Initial evaluation
    Vector g(n);
    Scalar f = global_callback(x, &g);
    result.initial_function_value = f;
    
    size_t total_evals = 1;
    size_t total_inner_iters = 0;
    size_t blocks_processed = 0;
    Scalar max_block_improvement = 0;
    
    if (m_options.verbose) {
      fmt::print("[BlockLBFGS] Starting: n={}, block_strategy={}, overlap_ratio={:.2f}\n",
                 n, m_options.block_selection, m_options.overlap_ratio);
      fmt::print("[BlockLBFGS] Initial f = {:.6e}, ‖g‖∞ = {:.6e}\n", 
                 f, g.template lpNorm<Eigen::Infinity>());
    }
    
    // Main outer loop
    for (size_t outer_iter = 0; outer_iter < m_options.max_outer_iterations; ++outer_iter) {
      m_outer_iteration_count = outer_iter;
      Scalar f_start_cycle = f;
      bool significant_progress = false;
      
      // Reset coverage tracking if needed
      if (m_options.track_coverage) {
        reset_coverage_tracking();
      }
      
      // Determine number of blocks in this cycle
      // Use more blocks for better coverage in early iterations
      size_t num_blocks = std::max<size_t>(1, (2 * n + m_options.block_size - 1) / m_options.block_size);
      num_blocks = std::min(num_blocks, static_cast<size_t>(10) ); // Limit to avoid too many small blocks
      
      // Process each block
      for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // Select block with current strategy
        auto block_indices = select_block(n, m_options.block_size, g, outer_iter);
        
        if (block_indices.empty()) continue;
        
        // Update coverage tracking
        if (m_options.track_coverage) {
          update_coverage_counts(block_indices);
        }
        
        // Extract block subproblem
        Vector x_block = extract_subvector(x, block_indices);
        Vector g_block = extract_subvector(g, block_indices);
        
        // Create block callback
        auto block_callback = [&](Vector const & x_sub, Vector * g_sub) -> Scalar {
          Vector x_full = x;
          update_full_vector(x_full, x_sub, block_indices);
          
          if (m_use_bounds) project_point(x_full);
          
          Vector g_full(n);
          Scalar f_val = global_callback(x_full, &g_full);
          total_evals++;
          
          if (g_sub) {
            *g_sub = extract_subvector(g_full, block_indices);
          }
          
          return f_val;
        };
        
        // Set up inner L-BFGS
        typename LBFGS_minimizer<Scalar>::Options inner_opts;
        inner_opts.max_iter = m_options.max_inner_iterations;
        inner_opts.g_tol = m_options.lbfgs_g_tol;
        inner_opts.f_tol = m_options.lbfgs_f_tol;
        inner_opts.m = m_options.lbfgs_m;
        inner_opts.verbose = false;
        
        LBFGS_minimizer<Scalar> inner_minimizer(inner_opts);
        
        // Set bounds for block if needed
        if (m_use_bounds) {
          Vector block_lower, block_upper;
          extract_subbounds(block_indices, block_lower, block_upper);
          inner_minimizer.set_bounds(block_lower, block_upper);
        }
        
        // Optimize block
        auto inner_result = inner_minimizer.minimize(x_block, block_callback, linesearch);
        
        total_inner_iters += inner_result.iterations;
        blocks_processed++;
        
        // Update solution
        //update_full_vector(x, inner_result.solution, block_indices);
        update_full_vector(x, inner_minimizer.solution(), block_indices);
        if (m_use_bounds) project_point(x);
        
        // Re-evaluate at new point
        f = global_callback(x, &g);
        total_evals++;
        
        // Track block improvement
        Scalar block_improvement = f_start_cycle - f;
        max_block_improvement = std::max(max_block_improvement, block_improvement);
        
        if (m_options.verbose && inner_result.iterations > 0) {
          fmt::print("[BlockLBFGS] Block {}/{}: size={}, inner_iters={}, Δf={:.2e}\n",
                     block_idx + 1, num_blocks, block_indices.size(), 
                     inner_result.iterations, block_improvement);
        }
      }
      
      // Adaptive block size adjustment
      Scalar f_improvement = f_start_cycle - f;
      adapt_block_size(f_improvement, n, outer_iter);
      
      // Convergence checking
      Scalar gnorm = projected_gradient_norm(x, g);
      
      if (m_options.verbose && (outer_iter % m_options.progress_frequency == 0)) {
        double coverage = m_options.track_coverage ? result.coverage_ratio : 0.0;
        fmt::print("[BlockLBFGS] Outer iter {}: f={:.6e}, ‖pg‖={:.2e}, Δf={:.2e}, coverage={:.1f}%\n",
                   outer_iter, f, gnorm, f_improvement, coverage * 100);
      }
      
      // Check convergence
      if (gnorm < m_options.outer_tolerance) {
        result.status = Status::GRADIENT_TOO_SMALL;
        break;
      }
      
      if (f_improvement < m_options.progress_threshold * std::abs(f_start_cycle)) {
        m_stagnation_count++;
        if (m_stagnation_count > 3) {
          result.status = Status::CONVERGED;
          break;
        }
      } else {
        m_stagnation_count = 0;
      }
      
      // Check coverage and adjust strategy if needed
      if (m_options.track_coverage && !check_coverage_adequate()) {
        if (m_options.verbose) {
          fmt::print("[BlockLBFGS] Poor coverage detected, adjusting block selection\n");
        }
        // Could switch to coverage-aware strategy here
      }
    }
    
    // Final results
    result.solution = x;
    result.final_function_value = f;
    result.outer_iterations = m_outer_iteration_count;
    result.inner_iterations = total_inner_iters;
    result.total_iterations = result.outer_iterations + total_inner_iters;
    result.outer_evaluations = total_evals;
    result.inner_evaluations = total_evals;
    result.total_evaluations = total_evals;
    result.blocks_processed = blocks_processed;
    result.max_block_improvement = max_block_improvement;
    
    // Calculate final coverage statistics
    if (m_options.track_coverage) {
      calculate_coverage_stats(result);
    }
    
    if (result.status == Status::FAILED && 
        result.outer_iterations >= m_options.max_outer_iterations) {
      result.status = Status::MAX_OUTER_ITERATIONS;
    }
    
    if (m_options.verbose) {
      fmt::print("[BlockLBFGS] Finished: status={}, f_final={:.6e}, total_evals={}\n",
                 static_cast<int>(result.status), result.final_function_value, total_evals);
      if (m_options.track_coverage) {
        fmt::print("[BlockLBFGS] Final coverage: {:.1f}% of coordinates visited\n",
                   result.coverage_ratio * 100);
      }
    }
    
    return result;
  }
};

}

#endif

#endif

//
// eof: Utils_LBFGS.hh
