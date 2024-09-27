/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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
 |      Universita` degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: Utils_Algo748.hh
//

#pragma once

#ifndef UTILS_ALGO748_dot_HH
#define UTILS_ALGO748_dot_HH

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>

#include "Utils.hh"

namespace Utils {

  using std::pow;
  using std::abs;

  /*
  //      _    _           _____ _  _    ___
  //     / \  | | __ _  __|___  | || |  ( _ )
  //    / _ \ | |/ _` |/ _ \ / /| || |_ / _ \
  //   / ___ \| | (_| | (_) / / |__   _| (_) |
  //  /_/   \_\_|\__, |\___/_/     |_|  \___/
  //             |___/
  */

  //!
  //! \class Algo748_base_fun
  //! \brief Abstract base class for defining mathematical functions used in the zero search algorithm.
  //!
  //! This class serves as a base interface for user-defined functions that can be evaluated.
  //! It allows for the implementation the numerical method to
  //! find the solution of the one dimensional equation \f$ f(x) = 0 \f$.
  //! Users must inherit from this class and implement the virtual method to define their specific functions.
  //!
  //! **Template Parameter:**
  //! - `Real`: A numeric type representing the data type of the function's input and output,
  //!   such as `float`, `double`, etc.
  //!
  //! **Usage Example:**
  //! To create a custom function, derive from this class and implement the required methods.
  //! Here is an example for the function \f$ f(x) = x^2 - 2 \f$:
  //!
  //! \code{cpp}
  //! class Fun1 : public Trichotomy_base_fun<double> {
  //! public:
  //!     double eval(double x) const override { return x*x - 2; }
  //! };
  //! \endcode
  //!
  template <typename Real>
  class Algo748_base_fun {
  public:
    //!
    //! Evaluate the function \f$ f(x) \f$
    //!
    //! \param[in] x the point to evaluate \f$ f(x) \f$
    //! \return the value of \f$ f(x) \f$
    //!
    virtual Real eval( Real x ) const = 0;
    //!
    //! Evaluate the function \f$ f(x) \f$
    //!
    //! \param[in] x the point to evaluate \f$ f(x) \f$
    //! \return the value of \f$ f(x) \f$
    //!
    Real operator () ( Real x ) const { return this->eval(x); }
  };

  #ifndef DOXYGEN_SHOULD_SKIP_THIS
  template <typename Real, typename PFUN>
  class Algo748_fun : public Algo748_base_fun<Real> {
    PFUN m_fun;
  public:
    explicit Algo748_fun( PFUN pfun ) : m_fun(pfun) {}
    Real eval( Real x ) const override { return m_fun(x); };
  };
  #endif

  //!
  //! \class Algo748
  //! \brief Class for solving \f$ f(x)=0 \f$ without the usew of derivative
  //!
  //! \note The used algorithm is described in:
  //!       **G. E. Alefeld, Florian A Potra, Yixun Shi**,
  //!       *Algorithm 748: enclosing zeros of continuous functions*,
  //!       ACM Transactions on Mathematical Software, vol 21, N.3, 1995
  //!
  //! **Usage Example**
  //!
  //! To use this class, first wrap your function in a derived class. For instance, for the function \f$ f(x) = x^2 - 2 \f$, you can define:
  //!
  //! \code{cpp}
  //! class Fun1 : public Algo748_base_fun<double> {
  //! public:
  //!   double eval(double x) const override { return x*x - 2; }
  //! };
  //! \endcode
  //!
  //! Next, instantiate the function and the solver. Then, call the desired method to find the root:
  //!
  //! \code{cpp}
  //! Algo748<real_type> solver;
  //! Fun1 f;
  //! real_type a=-1,b=2;
  //! real_type x_solution = solver.eval2(a,b,f);
  //! \endcode
  //!
  //! If the method converges, `x_solution` will contain the computed solution.
  //!
  template <typename Real>
  class Algo748 {

    using Integer = int;

    Real m_mu{Real(0.5)};
    Real m_tolerance{pow(machine_eps<Real>(),Real(2./3.))};
    Real m_interval_shink{Real(0.025)};

    bool m_converged{false};

    Real m_a{0}, m_fa{0};
    Real m_b{0}, m_fb{0};
    Real m_c{0}, m_fc{0};
    Real m_d{0}, m_fd{0};
    Real m_e{0}, m_fe{0};

    Algo748_base_fun<Real> * m_function{nullptr};

    Integer m_max_fun_evaluation{1000}; // max number of function evaluations
    Integer m_max_iteration{200};       // max number of iterations

    mutable Integer m_iteration_count{0};    // explore iteration counter
    mutable Integer m_fun_evaluation_count{0};

    bool bracketing();
    void set_tolerance( Real tol );
    Real pzero();
    Real newton_quadratic( Integer niter );
    Real evaluate( Real x ) { ++m_fun_evaluation_count; return m_function->eval(x); };
    bool all_different( Real a, Real b, Real c, Real d ) const;

    Real eval();
    Real eval( Real a, Real b );
    Real eval( Real a, Real b, Real amin, Real bmax );

  public:

    Algo748() = default;
    ~Algo748() = default;

    //!
    //! Find the solution for a function wrapped in the class `Algo748_base_fun<Real>`
    //! starting from guess interval `[a,b]`
    //!
    //! \param a    lower bound search interval
    //! \param b    upper bound search interval
    //! \param fun  the pointer to base class `Algo748_base_fun<Real>` wrapping the user function
    //!
    Real
    eval( Real a, Real b, Algo748_base_fun<Real> * fun ) {
      m_function = fun;
      return this->eval( a, b );
    }

    //!
    //! Find the solution for a function wrapped in the class `Algo748_base_fun<Real>`
    //! starting from guess interval `[a,b]`
    //!
    //! \param a    guess interval lower bound
    //! \param b    guess interval upper bound
    //! \param amin lower bound search interval
    //! \param bmax upper bound search interval
    //! \param fun  the pointer to base class `Algo748_base_fun<Real>` wrapping the user function
    //!
    Real
    eval( Real a, Real b, Real amin, Real bmax, Algo748_base_fun<Real> * fun ) {
      m_function = fun;
      return this->eval( a, b, amin, bmax );
    }

    //!
    //! Find the solution for a function stored in `pfun`
    //! starting from guess interval `[a,b]`
    //!
    //! \param a    lower bound search interval
    //! \param b    upper bound search interval
    //! \param pfun object storing the function
    //!
    template <typename PFUN>
    Real
    eval2( Real a, Real b, PFUN pfun ) {
      Algo748_fun<Real,PFUN> fun( pfun );
      m_function = &fun;
      return this->eval( a, b );
    }

    //!
    //! Find the solution for a function stored in `pfun`
    //! starting from guess interval `[a,b]`
    //!
    //! \param a    guess interval lower bound
    //! \param b    guess interval upper bound
    //! \param amin lower bound search interval
    //! \param bmax upper bound search interval
    //! \param pfun object storing the function
    //!
    template <typename PFUN>
    Real
    eval2( Real a, Real b, Real amin, Real bmax, PFUN pfun ) {
      Algo748_fun<Real,PFUN> fun( pfun );
      m_function = &fun;
      return this->eval( a, b, amin, bmax );
    }

    //!
    //! Find the solution for a function wrapped into `pfun`
    //! starting from guess interval `[a,b]`
    //!
    //! \param a    lower bound search interval
    //! \param b    upper bound search interval
    //! \param fa   the value \f$ f(a) \f$
    //! \param fb   the value \f$ f(b) \f$
    //! \param pfun object storing the function
    //!
    template <typename PFUN>
    Real
    eval3( Real a, Real b, Real fa, Real fb, PFUN pfun ) {
      Algo748_fun<Real,PFUN> fun( pfun );
      m_function             = &fun;
      m_iteration_count      = 0;
      m_fun_evaluation_count = 0;
      m_a = a; m_fa = fa;
      m_b = b; m_fb = fb;
      return eval();
    }

    //!
    //! Fix the maximum number of iteration.
    //!
    //! \param mit the maximum number of iteration
    //!
    void set_max_iterations( Integer mit );

    //!
    //! Fix the maximum number of evaluation.
    //!
    //! \param mfev the maximum number of evaluation of \f$ f(x) \f$
    //!
    void set_max_fun_evaluation( Integer mfev );

    //!
    //! \return the number of iterations used in the last computation
    //!
    Integer used_iter() const { return m_iteration_count; }

    //!
    //! \return the number of evaluation used in the last computation
    //!
    Integer num_fun_eval() const { return m_fun_evaluation_count; }

    //!
    //! \return the tolerance set for computation
    //!
    Real tolerance() const { return m_tolerance; }

    //!
    //! \return true if the last computation was successfull
    //!
    bool converged() const { return m_converged; }

  };

  #ifndef UTILS_OS_WINDOWS
  extern template class Algo748<float>;
  extern template class Algo748<double>;
  #endif

}

#endif

//
// EOF: Utils_Algo748.hh
//
