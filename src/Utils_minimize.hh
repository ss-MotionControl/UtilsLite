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

#ifndef UTILS_MINIMIZE_dot_HH
#define UTILS_MINIMIZE_dot_HH

#include <set>
#include <optional>
#include <limits>
#include "Utils_fmt.hh"
#include "Utils_nonlinear_linesearch.hh"

namespace Utils
{

  template <typename Scalar = double>
  class BoxConstraintHandler
  {
  public:
    using Vector  = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using integer = Eigen::Index;

  private:
    Vector m_lower, m_upper, m_tol_lower, m_tol_upper;
    bool   m_active{ false };
    Scalar m_epsi{ std::numeric_limits<Scalar>::epsilon() };

  public:
    void
    set_bounds( Vector const & lower, Vector const & upper )
    {
      UTILS_ASSERT(
        lower.size() == upper.size(),
        "Lower and upper bounds must have same size (#lower={},#upper={})",
        lower.size(),
        upper.size() );
      UTILS_ASSERT( ( lower.array() <= upper.array() ).all(), "Lower bounds must be <= upper bounds" );

      m_lower  = lower;
      m_upper  = upper;
      m_active = true;

      m_tol_lower = m_epsi * ( Vector::Ones( lower.size() ).array() + lower.array().abs() );
      m_tol_upper = m_epsi * ( Vector::Ones( upper.size() ).array() + upper.array().abs() );
    }

    void
    translate( Vector const & dir )
    {
      m_lower -= dir;
      m_upper -= dir;
      m_active = true;
    }

    Scalar
    lower( Eigen::Index i ) const
    {
      return m_lower( i );
    }
    Scalar
    upper( Eigen::Index i ) const
    {
      return m_upper( i );
    }

    Scalar &
    lower( Eigen::Index i )
    {
      return m_lower( i );
    }
    Scalar &
    upper( Eigen::Index i )
    {
      return m_upper( i );
    }

    Vector const &
    lower() const
    {
      return m_lower;
    }
    Vector const &
    upper() const
    {
      return m_upper;
    }

    bool
    is_active() const
    {
      return m_active;
    }

    void
    project( Vector & x ) const
    {
      if ( m_active ) x = x.cwiseMax( m_lower ).cwiseMin( m_upper );
    }

    void
    project_gradient( Vector const & x, Vector & g ) const
    {
      if ( !m_active ) return;

      const Scalar eps = Scalar( 10 ) * m_epsi;

      for ( int i = 0; i < x.size(); ++i )
      {
        bool on_lower = ( x[i] <= m_lower[i] + m_tol_lower[i] );
        bool on_upper = ( x[i] >= m_upper[i] - m_tol_upper[i] );

        if ( ( on_lower && g[i] < -eps ) || ( on_upper && g[i] > eps ) ) { g[i] = Scalar( 0 ); }
      }
    }

    void
    project_direction( Vector const & x, Vector & d ) const
    {
      if ( !m_active ) return;

      const Scalar eps = Scalar( 10 ) * m_epsi;

      integer sz = x.size();
      for ( integer i = 0; i < sz; ++i )
      {
        bool on_lower = ( x[i] <= m_lower[i] + m_tol_lower[i] );
        bool on_upper = ( x[i] >= m_upper[i] - m_tol_upper[i] );

        if ( ( on_lower && d[i] < -eps ) || ( on_upper && d[i] > eps ) ) { d[i] = Scalar( 0 ); }
      }
    }

    Scalar
    projected_gradient_norm( Vector const & x, Vector const & g ) const
    {
      Vector pg = g;
      project_gradient( x, pg );
      return pg.template lpNorm<Eigen::Infinity>();
    }

    // NEW: Get maximum feasible step along direction
    Scalar
    max_step_length( Vector const & x, Vector const & d ) const
    {
      if ( !m_active ) return std::numeric_limits<Scalar>::infinity();

      Scalar alpha_max = std::numeric_limits<Scalar>::infinity();

      integer sz = x.size();
      for ( integer i = 0; i < sz; ++i )
      {
        if ( d[i] > m_epsi ) { alpha_max = std::min( alpha_max, ( m_upper[i] - x[i] ) / d[i] ); }
        else if ( d[i] < -m_epsi ) { alpha_max = std::min( alpha_max, ( m_lower[i] - x[i] ) / d[i] ); }
      }

      return Scalar( 0.99 ) * alpha_max;  // Safety margin
    }

    void
    center( Scalar const rho, Vector & X, BoxConstraintHandler<Scalar> const & box )
    {
      m_lower.resize( X.size() );
      m_upper.resize( X.size() );
      m_tol_lower.resize( X.size() );
      m_tol_upper.resize( X.size() );
      integer N = X.size();
      for ( integer j = 0; j < N; ++j )
      {
        Scalar L  = box.m_lower( j );
        Scalar U  = box.m_upper( j );
        Scalar Xj = X( j );

        Scalar temp = U - L;
        Scalar wsl  = L - Xj;
        Scalar wsu  = U - Xj;

        if ( wsl >= -rho )
        {
          if ( wsl >= Scalar( 0 ) )
          {
            Xj  = L;
            wsl = Scalar( 0 );
            wsu = temp;
          }
          else
          {
            Xj  = L + rho;
            wsl = -rho;
            wsu = std::max( U - Xj, rho );
          }
        }
        else if ( wsu <= rho )
        {
          if ( wsu <= Scalar( 0 ) )
          {
            X( j ) = U;
            wsl    = -temp;
            wsu    = Scalar( 0 );
          }
          else
          {
            X( j ) = U - rho;
            wsl    = std::min( L - Xj, -rho );
            wsu    = rho;
          }
        }

        m_lower( j ) = wsl;
        m_upper( j ) = wsu;
      }

      m_tol_lower = m_epsi * ( Vector::Ones( m_lower.size() ).array() + m_lower.array().abs() );
      m_tol_upper = m_epsi * ( Vector::Ones( m_upper.size() ).array() + m_upper.array().abs() );
    }

    Scalar
    kkt_violation_norm2( Vector const & x, Vector const & v ) const
    {
      if ( !m_active ) return 0;

      auto xa = x.array();
      auto va = v.array();

      auto atLower = ( xa <= m_lower.array() + m_tol_lower.array() );
      auto atUpper = ( xa >= m_upper.array() - m_tol_upper.array() );
      auto freeVar = !( atLower || atUpper );

      return freeVar.select( va, Scalar( 0 ) ).cwiseAbs2().sum() +
             atLower.select( va.min( Scalar( 0 ) ), Scalar( 0 ) ).cwiseAbs2().sum() +
             atUpper.select( va.max( Scalar( 0 ) ), Scalar( 0 ) ).cwiseAbs2().sum();
    }

    void
    clear()
    {
      m_active = false;
      m_lower.resize( 0 );
      m_upper.resize( 0 );
      m_tol_lower.resize( 0 );
      m_tol_upper.resize( 0 );
    }
  };

  //============================================================================
  // Interpolation Model - Manages quadratic model
  //============================================================================

  /**
   * @class InterpolationModel
   * @brief Manages the local quadratic interpolation model for the BOBYQA algorithm.
   *
   * The model approximates the objective function \f$ f(x) \f$ in the neighborhood
   * of the current base point \f$ x_{base} \f$ using a quadratic function \f$ Q(\delta) \f$,
   * where \f$ \delta \f$ is the offset from \f$ x_{base} \f$.
   *
   * The quadratic model is defined as:
   * \f[
   * Q(\delta) = f(x_{base} + x_{opt}) + g_{opt}^T (\delta - x_{opt}) + \frac{1}{2} (\delta - x_{opt})^T H (\delta -
   * x_{opt})
   * \f]
   *
   * In BOBYQA, the Hessian matrix \f$ H \f$ is not stored explicitly. Instead, it is
   * composed of an explicit part \f$ H_Q \f$ and an implicit part derived from the
   * interpolation points:
   * \f[
   * H \cdot v = H_Q \cdot v + \sum_{i=1}^{npt} pq_i (x_i^T v) x_i
   * \f]
   * where \f$ x_i \f$ are the columns of \f$ m\_xpt \f$ and \f$ pq_i \f$ are the
   * coefficients in \f$ m\_pq \f$.
   *
   * @tparam Scalar The floating-point type (e.g., double, float).
   */
  template <typename Scalar>
  class InterpolationModel
  {
  public:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Index  = Eigen::Index;

  private:
    Index m_nv{ 0 };    ///< Number of variables (dimension of the problem)
    Index m_npt{ 0 };   ///< Number of interpolation points (usually 2n+1)
    Index m_nptm{ 0 };  ///< Equal to npt - nv - 1

    Matrix m_xpt;   ///< Matrix (nv x npt) containing interpolation points (offsets from base)
    Vector m_fval;  ///< Function values \f$ f(x_i) \f$ at the interpolation points
    Matrix m_HQ;    ///< Upper triangular part of the explicit Hessian \f$ H_Q \f$
    Vector m_pq;    ///< Coefficients for the implicit part of the Hessian
    Matrix m_B;     ///< Matrix used for updating Lagrange polynomials
    Matrix m_Z;     ///< Null space matrix for Frobenius norm minimization of the Hessian

    Index m_kopt{ 0 };  ///< Index of the current best point (trust region center) in the set

  public:
    InterpolationModel() = default;

    /**
     * @brief Initializes the data structures for the model.
     * * Resizes all internal matrices and vectors according to the problem size
     * and the number of interpolation points.
     * * @param nv Number of variables.
     * @param npt Number of interpolation points.
     */
    void
    initialize( Index nv, Index npt )
    {
      m_nv   = nv;
      m_npt  = npt;
      m_nptm = npt - nv - 1;

      m_xpt.resize( nv, npt );
      m_fval.resize( npt );
      m_HQ.resize( nv, nv );
      m_pq.resize( npt );
      m_B.resize( nv, npt + nv );
      m_Z.resize( npt, m_nptm );
      this->setZero();
      m_kopt = 0;
    }

    void
    setZero()
    {
      m_xpt.setZero();
      m_fval.setZero();
      m_HQ.setZero();
      m_pq.setZero();
      m_B.setZero();
      m_Z.setZero();
    }


    /**
     * @brief Computes the model gradient \f$ \nabla Q \f$ at a given point.
     * * The gradient is calculated as:
     * \f[
     * \nabla Q(\delta) = g_{opt} + H (\delta - x_{opt})
     * \f]
     * where \f$ H \f$ is the composite Hessian.
     * * @param g_opt The gradient of the model at the current optimal point \f$ x_{opt} \f$.
     * @param x_opt The offset of the current best point from the base.
     * @param delta The point (offset) where the gradient is to be evaluated.
     * @param[out] grad Output vector for the computed gradient.
     */
    void
    gradient( Vector const & g_opt, Vector const & x_opt, Vector const & delta, Vector & grad ) const
    {
      Vector diff = delta - x_opt;

      // Part 1: g_opt + H_Q * (delta - x_opt)
      grad.noalias() = g_opt + m_HQ.template selfadjointView<Eigen::Upper>() * diff;

      // Part 2: Add implicit terms sum( pq_i * (xpt_i^T * diff) * xpt_i )
      // Optimized via Eigen arrays to avoid explicit summation loops.
      grad.noalias() += m_xpt * ( m_pq.array() * ( m_xpt.transpose() * diff ).array() ).matrix();
    }

    /**
     * @brief Computes the Hessian-vector product \f$ hs = H \cdot s \f$.
     * * This operation is performed in \f$ O(n \cdot npt) \f$ time by exploiting
     * the implicit structure of the Hessian in BOBYQA.
     * * @param s The vector to be multiplied.
     * @param[out] hs Output vector for the product \f$ H \cdot s \f$.
     */
    void
    add_hessian_product( Vector const & s, Vector & hs ) const
    {
      // Explicit part
      hs.noalias() += m_HQ.template selfadjointView<Eigen::Upper>() * s;

      // Implicit part: sum_i [ pq_i * (xpt_i . s) * xpt_i ]
      hs.noalias() += m_xpt * ( m_pq.array() * ( m_xpt.transpose() * s ).array() ).matrix();
    }

    /**
     * @brief Evaluates the quadratic model \f$ Q(\delta) \f$ at a specific offset.
     * * Uses the Taylor expansion around \f$ x_{opt} \f$:
     * \f[
     * Q(\delta) = f_{opt} + g_{opt}^T (\delta - x_{opt}) + \frac{1}{2} (\delta - x_{opt})^T H (\delta - x_{opt})
     * \f]
     * * @param g_opt Model gradient at the optimum.
     * @param x_opt Offset of the optimum point.
     * @param delta Point where the model is evaluated.
     * @return The scalar value of the quadratic model.
     */
    Scalar
    evaluate( Vector const & g_opt, Vector const & x_opt, Vector const & delta ) const
    {
      Vector diff = delta - x_opt;
      Vector h_diff( m_nv );
      hessian_product( diff, h_diff );
      return m_fval( m_kopt ) + diff.dot( g_opt ) + Scalar( 0.5 ) * diff.dot( h_diff );
    }

    /**
     * @brief Updates the data for a specific interpolation point.
     * * @param knew The index of the point to be replaced [0, npt-1].
     * @param xnew The new offset vector for the point.
     * @param fnew The function value at the new point.
     */
    void
    update_point( Index knew, Vector const & xnew, Scalar fnew )
    {
      m_xpt.col( knew ) = xnew;
      m_fval( knew )    = fnew;

      // Check if the new point is better than the current optimum
      if ( fnew < m_fval( m_kopt ) ) { m_kopt = knew; }
    }

    /**
     * @brief Performs a base point shift to maintain numerical stability.
     * * When the trust region center \f$ x_{opt} \f$ becomes the new base point,
     * the coordinates of all interpolation points and the Hessian coefficients
     * must be updated. This ensures the model remains mathematically equivalent
     * in the new coordinate system.
     * * @param xopt The translation vector (offset of the new base from the old base).
     */
    void
    shift_base( Vector const & xopt )
    {
      // Translate all points: xpt_new = xpt_old - xopt
      m_xpt.colwise() -= xopt;

      // Update the explicit Hessian HQ to reflect the coordinate change.
      // This follows Powell's specific algebraic update rules for BOBYQA.
      Scalar sumpq = m_pq.sum();
      Vector temp  = ( m_xpt * m_pq ).noalias() - ( Scalar( 0.5 ) * sumpq ) * xopt;

      for ( Index j = 0; j < m_nv; ++j )
      {
        for ( Index i = 0; i <= j; ++i ) { m_HQ( i, j ) += temp( i ) * xopt( j ) + xopt( i ) * temp( j ); }
      }
    }

    // --- Basic Getters ---
    Index
    nv() const
    {
      return m_nv;
    }
    Index
    npt() const
    {
      return m_npt;
    }
    Index
    kopt() const
    {
      return m_kopt;
    }
    Scalar
    fopt() const
    {
      return m_fval( m_kopt );
    }

    /** @return The offset of the best point currently in the model. */
    auto
    xopt_vec() const
    {
      return m_xpt.col( m_kopt );
    }

    // --- Matrix/Vector Accessors (Required for B and Z updates) ---
    Matrix &
    xpt()
    {
      return m_xpt;
    }
    Vector &
    fval()
    {
      return m_fval;
    }
    Matrix &
    HQ()
    {
      return m_HQ;
    }
    Vector &
    pq()
    {
      return m_pq;
    }
    Matrix &
    B()
    {
      return m_B;
    }
    Matrix &
    Z()
    {
      return m_Z;
    }
    Index &
    kopt()
    {
      return m_kopt;
    }

    Matrix const &
    xpt() const
    {
      return m_xpt;
    }
    Vector const &
    fval() const
    {
      return m_fval;
    }
    Matrix const &
    HQ() const
    {
      return m_HQ;
    }
    Vector const &
    pq() const
    {
      return m_pq;
    }
    Matrix const &
    B() const
    {
      return m_B;
    }
    Matrix const &
    Z() const
    {
      return m_Z;
    }
  };

  //============================================================================
  // Trust Region Solver - Solves trust region subproblem
  //============================================================================

  template <typename Scalar>
  class TrustRegionSolver
  {
  public:
    using Vector  = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using VectorI = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
    using Index   = Eigen::Index;

    struct Result
    {
      Vector step;
      Scalar reduction;
      Scalar crvmin;
      bool   converged;
    };

  private:
    Index                                m_nv;
    BoxConstraintHandler<Scalar> const * m_bounds;

    // Working arrays
    VectorI m_xbdi;
    Vector  m_s, m_hs, m_hred, m_gnew;

  public:
    TrustRegionSolver() = default;

    void
    initialize( Index nv, BoxConstraintHandler<Scalar> const * bounds )
    {
      m_nv     = nv;
      m_bounds = bounds;

      m_xbdi.resize( nv );
      m_s.resize( nv );
      m_hs.resize( nv );
      m_hred.resize( nv );
      m_gnew.resize( nv );
    }

    Result
    solve( Vector const & xopt, Vector const & gopt, InterpolationModel<Scalar> const & model, Scalar delta )
    {
      Result res;
      res.step.resize( m_nv );
      res.step.setZero();
      res.reduction = 0;
      res.crvmin    = -1;
      res.converged = false;

      // Identify active constraints
      m_xbdi.setZero();
      Index nact = 0;

      for ( Index i = 0; i < m_nv; ++i )
      {
        if ( xopt( i ) <= m_bounds->lower( i ) && gopt( i ) >= 0 )
        {
          m_xbdi( i ) = -1;
          ++nact;
        }
        else if ( xopt( i ) >= m_bounds->upper( i ) && gopt( i ) <= 0 )
        {
          m_xbdi( i ) = 1;
          ++nact;
        }
      }

      // Truncated conjugate gradient
      m_gnew        = gopt;
      Scalar delsq  = delta * delta;
      Scalar gredsq = 0;

      for ( Index i = 0; i < m_nv; ++i )
      {
        if ( m_xbdi( i ) == 0 ) { gredsq += m_gnew( i ) * m_gnew( i ); }
      }

      if ( gredsq == 0 )
      {
        res.converged = true;
        return res;
      }

      Index  itermax = m_nv - nact;
      Scalar beta    = 0;

      for ( Index iter = 0; iter < itermax; ++iter )
      {
        // Compute search direction
        m_s = beta * m_s - m_gnew;

        for ( Index i = 0; i < m_nv; ++i )
        {
          if ( m_xbdi( i ) != 0 ) m_s( i ) = 0;
        }

        Scalar stepsq = m_s.squaredNorm();
        if ( stepsq == 0 ) break;

        // Compute Hessian-vector product
        model.hessian_product( m_s, m_hs );

        // Line search in trust region
        Scalar ds    = res.step.dot( m_s );
        Scalar shs   = m_s.dot( m_hs );
        Scalar resid = delsq - res.step.squaredNorm();

        if ( resid <= 0 ) break;

        Scalar temp = std::sqrt( stepsq * resid + ds * ds );
        Scalar blen = ( ds < 0 ) ? ( temp - ds ) / stepsq : resid / ( temp + ds );

        Scalar stplen = ( shs > 0 ) ? std::min( blen, gredsq / shs ) : blen;

        // Check bound constraints
        for ( Index i = 0; i < m_nv; ++i )
        {
          if ( m_s( i ) != 0 )
          {
            Scalar xsum  = xopt( i ) + res.step( i );
            Scalar bound = ( m_s( i ) > 0 ) ? ( m_bounds->upper( i ) - xsum ) / m_s( i )
                                            : ( m_bounds->lower( i ) - xsum ) / m_s( i );

            stplen = std::min( stplen, bound );
          }
        }

        // Update step
        if ( stplen > 0 )
        {
          Scalar ggsav = gredsq;
          gredsq       = 0;

          m_gnew += stplen * m_hs;
          res.step += stplen * m_s;

          for ( Index i = 0; i < m_nv; ++i )
          {
            if ( m_xbdi( i ) == 0 ) { gredsq += m_gnew( i ) * m_gnew( i ); }
          }

          Scalar sdec = std::max( stplen * ( ggsav - Scalar( 0.5 ) * stplen * shs ), Scalar( 0 ) );
          res.reduction += sdec;

          if ( shs > 0 && res.crvmin < 0 ) { res.crvmin = shs / stepsq; }

          beta = gredsq / ggsav;

          // Check convergence
          if ( gredsq * delsq <= Scalar( 1e-4 ) * res.reduction * res.reduction )
          {
            res.converged = true;
            break;
          }
        }
        else
        {
          break;
        }
      }

      return res;
    }
  };
}  // namespace Utils

#endif
