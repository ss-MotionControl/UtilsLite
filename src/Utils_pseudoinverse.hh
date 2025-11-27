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
// file: Utils_pseudoinverse.hh
//

#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef UTILS_PSEUDOINVERSE_dot_HH
#define UTILS_PSEUDOINVERSE_dot_HH

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

#include <random>

namespace Utils {

  using std::abs;
  using std::min;
  using std::max;
  using std::sqrt;
  using std::pow;
  
  template <typename Scalar>
  class TikhonovPseudoInverse {
  private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SVD    = Eigen::JacobiSVD<Matrix,Eigen::ComputeThinU|Eigen::ComputeThinV>;
  
    mutable SVD    m_svd;
    mutable Vector m_inv_singular;

  public:
    // Costruttore con parametro di regolarizzazione (default 0.01)
    explicit
    TikhonovPseudoInverse(
      Matrix const & A,
      Scalar const   lambda,
      Scalar const   epsilon = 1e-12
    ) {
      compute( A, lambda, epsilon );
    }

    // Metodo per calcolare la pseudo-inversa regolarizzata
    void
    compute(
      Matrix const & A,
      Scalar const   lambda,
      Scalar const   epsilon = 1e-12
    ) const {
      new (&m_svd) SVD( A );

      auto const & singular_values = m_svd.singularValues();

      Eigen::Index ns{ singular_values.size() };
      m_inv_singular.resize( ns );
      for ( Eigen::Index i{0}; i < ns; ++i ) {
        Scalar s = singular_values.coeff(i);
        m_inv_singular.coeffRef(i) = s / (s * s + lambda * lambda + epsilon);
      }
    }

    // Applica la pseudo-inversa a un vettore o matrice
    template<typename Derived>
    Derived apply( Eigen::MatrixBase<Derived> const & b ) const {
      auto const & U = m_svd.matrixU();
      auto const & V = m_svd.matrixV();

      // Applica: x = V * Σ⁺ * Uᵀ * b
      return V * m_inv_singular.asDiagonal() * U.transpose() * b;
    }

    // Applica la trasposta della pseudo-inversa a un vettore o matrice
    template<typename Derived>
    Derived apply_transpose( Eigen::MatrixBase<Derived> const & b ) const {
      auto const & U = m_svd.matrixU();
      auto const & V = m_svd.matrixV();

      // Applica la trasposta: x = U * Σ⁺ * Vᵀ * b
      return U * m_inv_singular.asDiagonal() * V.transpose() * b;
    }
  };

  template <typename Scalar>
  class TikhonovSolver {
  private:
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SVD    = Eigen::JacobiSVD<Matrix,Eigen::ComputeThinU|Eigen::ComputeThinV>;
  
    Matrix       m_A;
    Eigen::Index m_m, m_n;
    Eigen::ColPivHouseholderQR<Matrix> m_qr;
    bool         m_lambda_gt_zero{true};
    
    Vector
    qr_solve_transpose( Vector const & c ) const {

      // estrai R (triangolare superiore n×n)
      auto R = m_qr.matrixR().topRows(m_n);

      // permutazione
      auto const & P = m_qr.colsPermutation();

      // rhs = Pᵀ c
      Vector rhs = P.transpose() * c;

      // risolvi Rᵀ y = rhs  (triangolare inferiore)
      Vector y = R.transpose().template triangularView<Eigen::Lower>().solve(rhs);

      // risolvi R z = y     (triangolare superiore)
      Vector z = R.template triangularView<Eigen::Upper>().solve(y);

      // x = P z
      return P * z;
    }

  public:

    TikhonovSolver( Matrix const & A, Scalar const lambda )
    : m_A(A), m_m(A.rows()), m_n(A.cols())
    {
      m_lambda_gt_zero = lambda > 0;
      if ( m_lambda_gt_zero ) {
        // Costruiamo matrice aumentata [A; lambda*I]
        Matrix A_aug(m_m + m_n, m_n);
        A_aug.topRows(m_m)    = m_A;
        A_aug.bottomRows(m_n) = lambda * Matrix::Identity(m_n, m_n);
        // QR decomposition della matrice aumentata
        m_qr.compute(A_aug);
      } else {
        m_qr.compute(m_A);
      }
    }

    // Risolve x = argmin ||A x - b||^2 + lambda^2 ||x||^2
    Vector
    solve( Vector const & b ) const {
      if ( m_lambda_gt_zero ) {
        Vector b_aug(m_m + m_n);
        b_aug.head(m_m) = b;
        b_aug.tail(m_n).setZero();
        return m_qr.solve(b_aug);
      } else {
        return m_qr.solve(b);
      }
    }

    // Risolve y = A * (A^T A + lambda^2 I)^-1 * c
    Vector
    solve_transpose( Vector const & c ) const {
      // Risolvi (A^T A + lambda^2 I) x = c usando il sistema aumentato
      return m_A * qr_solve_transpose(c);
    }
  };

}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

#endif

//
// eof: Utils_pseudoinverse.hh
