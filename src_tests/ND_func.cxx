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

/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  MADS test suite for gradient-free optimization problems                 |
 |                                                                          |
 |  Updated adaptation with verbose output and final x saving               |
\*--------------------------------------------------------------------------*/

using Scalar = double;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using std::string;

// -------------------------------------------------------------------
// 1. Rosenbrock 2D (Fixed dimension problem)
// -------------------------------------------------------------------

/**
 * @class Rosenbrock2D
 * @brief 2D Rosenbrock function, a classic benchmark for optimization
 * algorithms
 *
 * The Rosenbrock function is also known as the "banana function" due to its
 * curved valley. It has a global minimum at (1,1) with value 0.
 * The function is defined as: f(x,y) = (1-x)² + 100(y-x²)²
 */
template <typename T>
class Rosenbrock2D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-∞ for both variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 2, -std::numeric_limits<T>::infinity() );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (+∞ for both variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 2, std::numeric_limits<T>::infinity() );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (-1.2, 1.0)
   */
  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << -1.2, 1.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1.0, 1.0)
   */
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }

  /**
   * @brief Evaluate Rosenbrock function
   * @param x Input vector of size 2
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T a = 1.0, b = 100.0, t1 = a - x[0], t2 = x[1] - x[0] * x[0];
    return t1 * t1 + b * t2 * t2;
  }
};

// -------------------------------------------------------------------
// ---------------- Nesterov-Chebyshev-Rosenbrock --------------------
// -------------------------------------------------------------------

/**
 * @class NesterovChebyshevRosenbrock
 * @brief Nesterov-Chebyshev-Rosenbrock function, a non-smooth variant
 *
 * This function combines elements of Rosenbrock with Chebyshev polynomials
 * and absolute value functions, creating a challenging non-smooth optimization
 * problem. The global minimum is at (1,1,...,1) with value 0.
 */
template <typename T, int N>
class NesterovChebyshevRosenbrock
{
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /**
   * @brief Constructor enforcing dimension constraint
   */
  NesterovChebyshevRosenbrock() { static_assert( N >= 2, "NesterovChebyshevRosenbrock requires N >= 2" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (-1, -1, ..., -1)
   */
  Vector
  init() const
  {
    Vector x0 = Vector::Constant( N, 0.0 );
    x0[0]     = -1;
    for ( int i = 1; i < N; ++i ) x0[i] = -1.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1.0, 1.0, ..., 1.0)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Evaluate Nesterov-Chebyshev-Rosenbrock function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0.25 * std::abs( x[0] - 1.0 );
    for ( int i = 0; i < N - 1; ++i )
    {
      T term = x[i + 1] - 2.0 * std::abs( x[i] ) + 1.0;
      f += std::abs( term );
    }
    return f;
  }
};

// -------------------------------------------------------------------
// 2. Rosenbrock N-Dimensionale (Scalabile)
// -------------------------------------------------------------------

/**
 * @class RosenbrockN
 * @brief N-dimensional Rosenbrock function
 *
 * Scalable version of the Rosenbrock function for N dimensions.
 * The global minimum is at (1,1,...,1) with value 0.
 * The function is defined as: f(x) = Σ[i=0 to N-2] [(1-x_i)² +
 * 100(x_{i+1}-x_i²)²]
 */
template <typename T, int N>
class RosenbrockN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  RosenbrockN() { static_assert( N >= 2, "RosenbrockN requires N >= 2" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (-1, -1, ..., 1)
   */
  Vector
  init() const
  {
    Vector x0 = Vector::Constant( N, -1.0 );
    x0[N - 1] = 1.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1.0, 1.0, ..., 1.0)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Evaluate N-dimensional Rosenbrock function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N - 1; ++i )
    {
      T t1 = 1.0 - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      f += t1 * t1 + 100.0 * t2 * t2;
    }
    return f;
  }
};

// -------------------------------------------------------------------
// 3. Extended Powell Singular (Scalabile)
// -------------------------------------------------------------------

/**
 * @class PowellSingularN
 * @brief Extended Powell Singular function
 *
 * A scalable function composed of quartets of variables with singular behavior.
 * The global minimum is at the origin with value 0.
 * Requires N to be a multiple of 4.
 */
template <typename T, int N>
class PowellSingularN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  PowellSingularN() { static_assert( N % 4 == 0, "PowellSingularN requires N to be a multiple of 4" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-4.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -4.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (4.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 4.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point with pattern (3, -1, 0, 1) repeated
   */
  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N / 4; ++i )
    {
      x0[4 * i]     = 3.0;
      x0[4 * i + 1] = -1.0;
      x0[4 * i + 2] = 0.0;
      x0[4 * i + 3] = 1.0;
    }
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at the origin (0,0,...,0)
   */
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /**
   * @brief Evaluate Powell Singular function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;
      T   t1 = x[k1] + 10.0 * x[k2];
      T   t2 = x[k3] - x[k4];
      T   t3 = x[k2] - 2.0 * x[k3];
      T   t4 = x[k1] - x[k4];
      f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
    }
    return f;
  }
};

/**
 * @class ExtendedWoodN
 * @brief Extended Wood function
 *
 * Scalable version of the Wood function, extended to N dimensions.
 * Composed of quartets of variables with the Wood function structure.
 * The global minimum is at (1,1,...,1) with value 0.
 */
template <typename T, int N>
class ExtendedWoodN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  ExtendedWoodN() { static_assert( N % 4 == 0, "ExtendedWoodN requires N to be a multiple of 4" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-3.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -3.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (3.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 3.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point with alternating -3 and -1 values
   */
  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) { x0[i] = ( i % 2 == 0 ) ? -3.0 : -1.0; }
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1.0, 1.0, ..., 1.0)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Evaluate Extended Wood function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;
      T   t1 = x[k1] * x[k1] - x[k2];
      T   t2 = x[k1] - 1.0;
      T   t3 = x[k3] * x[k3] - x[k4];
      T   t4 = x[k3] - 1.0;
      T   t5 = x[k2] + x[k4] - 2.0;
      T   t6 = x[k2] - x[k4];
      f += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 + 10.0 * t5 * t5 + 0.1 * t6 * t6;
    }
    return f;
  }
};

// -------------------- Beale (2D) --------------------

/**
 * @class Beale2D
 * @brief Beale function, a 2D optimization test function
 *
 * The Beale function has a global minimum at (3,0.5) with value 0.
 * It features steep valleys and is multimodal.
 */
template <typename T>
class Beale2D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-4.5 for both variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 2, -4.5 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (4.5 for both variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 2, 4.5 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (1.0, 1.0)
   */
  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 1.0, 1.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (3.0, 0.5)
   */
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 3.0, 0.5;
    return x;
  }

  /**
   * @brief Evaluate Beale function
   * @param x Input vector of size 2
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );
    return t1 * t1 + t2 * t2 + t3 * t3;
  }
};

// -------------------- Himmelblau (2D) --------------------

/**
 * @class Himmelblau2D
 * @brief Himmelblau's function, a classic 2D optimization test function
 *
 * This function has four identical local minima with value 0, making it
 * useful for testing global optimization algorithms.
 * Minima at: (3.0,2.0), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)
 */
template <typename T>
class Himmelblau2D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-6.0 for both variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 2, -6.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (6.0 for both variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 2, 6.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (-3.0, -3.0)
   */
  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << -3.0, -3.0;
    return x0;
  }

  /**
   * @brief Get one of the exact solutions
   * @return One global minimum at (3.0, 2.0)
   */
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 3.0, 2.0;
    return x;
  }

  /**
   * @brief Evaluate Himmelblau's function
   * @param x Input vector of size 2
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;
    return f1 * f1 + f2 * f2;
  }
};

// -------------------- Freudenstein-Roth (2D) --------------------

/**
 * @class FreudensteinRoth2D
 * @brief Freudenstein and Roth function, a 2D nonlinear test function
 *
 * This function has a global minimum at (5,4) with value 0 and
 * a local minimum at (11.41..., -0.8968...) with value 48.9842...
 */
template <typename T>
class FreudensteinRoth2D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for both variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for both variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (0.5, -2.0)
   */
  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 0.5, -2.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (5.0, 4.0)
   */
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 5.0, 4.0;
    return x;
  }

  /**
   * @brief Evaluate Freudenstein-Roth function
   * @param x Input vector of size 2
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = -13.0 + x1 + ( ( 5.0 - x2 ) * x2 - 2.0 ) * x2;
    T f2 = -29.0 + x1 + ( ( x2 + 1.0 ) * x2 - 14.0 ) * x2;
    return f1 * f1 + f2 * f2;
  }
};

// -------------------- Helical Valley (3D) --------------------

/**
 * @class HelicalValley3D
 * @brief Helical Valley function, a 3D optimization test function
 *
 * This function has a global minimum at (1,0,0) with value 0.
 * The function features a helical valley that winds around the z-axis.
 */
template <typename T>
class HelicalValley3D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 3, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 3, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (-1.0, 0.0, 0.0)
   */
  Vector
  init() const
  {
    Vector x0( 3 );
    x0 << -1.0, 0.0, 0.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1.0, 0.0, 0.0)
   */
  Vector
  exact() const
  {
    Vector x( 3 );
    x << 1.0, 0.0, 0.0;
    return x;
  }

  /**
   * @brief Evaluate Helical Valley function
   * @param x Input vector of size 3
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T theta = 0.0;
    if ( x1 > 0 ) { theta = std::atan( x2 / x1 ) / ( 2.0 * M_PI ); }
    else if ( x1 < 0 ) { theta = std::atan( x2 / x1 ) / ( 2.0 * M_PI ) + 0.5; }
    else
    {
      theta = ( x2 >= 0 ) ? 0.25 : -0.25;
    }

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( std::sqrt( x1 * x1 + x2 * x2 ) - 1.0 );
    T f3 = x3;
    return f1 * f1 + f2 * f2 + f3 * f3;
  }
};

// -------------------- Powell Badly Scaled (2D) --------------------

/**
 * @class PowellBadlyScaled2D
 * @brief Powell's badly scaled function, a challenging 2D test function
 *
 * This function is badly scaled and has a global minimum at approximately
 * (1.098...e-5, 9.106...) with value 0.
 */
template <typename T>
class PowellBadlyScaled2D
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for both variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for both variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (0.0, 1.0)
   */
  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 0.0, 1.0;
    return x0;
  }

  /**
   * @brief Get exact solution (approximate due to numerical precision)
   * @return Global minimum approximately at (1.098e-5, 9.106)
   */
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 1.098159329699759e-05, 9.106146739867318;
    return x;
  }

  /**
   * @brief Evaluate Powell's badly scaled function
   * @param x Input vector of size 2
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;
    return f1 * f1 + f2 * f2;
  }
};

// -------------------- Brown Almost Linear (n=10) --------------------

/**
 * @class BrownAlmostLinearN
 * @brief Brown's almost linear function, a scalable test function
 *
 * This function has a global minimum when the product of all variables
 * equals 1. The function is designed to test algorithms on problems with nearly
 * linear constraints.
 */
template <typename T, int N>
class BrownAlmostLinearN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  BrownAlmostLinearN() { static_assert( N >= 2, "BrownAlmostLinearN requires N>=2" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-5.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -5.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (5.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 5.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (0.5, 0.5, ..., 0.5)
   */
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1,1,...,1)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Evaluate Brown's almost linear function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0.0;
    for ( int i = 0; i < N - 1; ++i )
    {
      T t = x[i] + x[i + 1] * x[i + 1] * x[i + 1] - 3.0;
      f += t * t;
    }
    for ( int i = 0; i < N; ++i ) f += 1e-3 * x[i] * x[i];
    return f;
  }
};

// -------------------- Broyden Tridiagonal (n-dim) --------------------

/**
 * @class BroydenTridiagonalN
 * @brief Broyden tridiagonal function, a scalable test function
 *
 * This function represents a tridiagonal system and is useful for testing
 * algorithms on problems with banded structure. The global minimum is at
 * (1,1,...,1).
 */
template <typename T, int N>
class BroydenTridiagonalN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  BroydenTridiagonalN() { static_assert( N >= 2, "BroydenTridiagonalN requires N>=2" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (0.5, 0.5, ..., 0.5)
   */
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (1,1,...,1)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Evaluate Broyden tridiagonal function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0.0;
    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? 0.0 : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;
      f += t * t;
    }
    return f;
  }
};

// -------------------- Ill-conditioned Quadratic (n-dim) --------------------

/**
 * @class IllConditionedQuadraticN
 * @brief Ill-conditioned quadratic function
 *
 * This function features exponentially increasing eigenvalues, creating
 * a very ill-conditioned Hessian matrix. Useful for testing algorithm
 * robustness to poor conditioning. The global minimum is at the origin.
 */
template <typename T, int N>
class IllConditionedQuadraticN
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-10.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (10.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point with alternating 1 and -1 values
   */
  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? 1.0 : -1.0;
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at the origin (0,0,...,0)
   */
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /**
   * @brief Evaluate ill-conditioned quadratic function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      T lambda = std::pow( 1e6, T( i ) / T( N - 1 ) );
      f += lambda * x[i] * x[i];
    }
    return f;
  }
};

// -------------------- Trigonometric Sum (n-dim) --------------------

/**
 * @class TrigonometricSumN
 * @brief Trigonometric sum function
 *
 * This function combines trigonometric terms and is useful for testing
 * algorithms on oscillatory functions. The global minimum is at the origin.
 */
template <typename T, int N>
class TrigonometricSumN
{
public:
  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-π for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -M_PI );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (π for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, M_PI );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (0.5, 0.5, ..., 0.5)
   */
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at the origin (0,0,...,0)
   */
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /**
   * @brief Evaluate trigonometric sum function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N; ++i )
    {
      T idx = T( i + 1 );
      T t   = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      f += t * t;
    }
    return f;
  }
};

// -------------------- Schwefel Function --------------------

/**
 * @class SchwefelN
 * @brief Schwefel function, a multimodal test function
 *
 * The Schwefel function is characterized by its many local minima
 * and a global minimum at (420.9687, 420.9687, ...). The function
 * is deceptive as the local minima are far from the global minimum.
 */
template <typename T, int N>
class SchwefelN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  SchwefelN() { static_assert( N >= 1, "SchwefelN requires N >= 1" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-500.0 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -500.0 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (500.0 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 500.0 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point with alternating 100 and -100 values
   */
  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) { x0[i] = ( i % 2 == 0 ) ? 100.0 : -100.0; }
    return x0;
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at (420.9687, 420.9687, ...)
   */
  Vector
  exact() const
  {
    return Vector::Constant( N, 420.9687 );
  }

  /**
   * @brief Evaluate Schwefel function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 418.9829 * static_cast<T>( N );
    for ( int i = 0; i < N; ++i ) { f -= x[i] * std::sin( std::sqrt( std::abs( x[i] ) ) ); }
    return f;
  }
};

/**
 * @class AckleyN
 * @brief Ackley function, a multimodal test function
 *
 * The Ackley function is widely used for testing optimization algorithms.
 * It has many local minima and a global minimum at the origin with value 0.
 * The function features an exponential term wrapped by a cosine wave.
 */
template <typename T, int N>
class AckleyN
{
public:
  /**
   * @brief Constructor enforcing dimension constraint
   */
  AckleyN() { static_assert( N >= 1, "AckleyN requires N >= 1" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-32.768 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -32.768 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (32.768 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 32.768 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (1.0, 1.0, ..., 1.0)
   */
  Vector
  init() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at the origin (0,0,...,0)
   */
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /**
   * @brief Evaluate Ackley function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;

    T sum1 = 0.0;
    T sum2 = 0.0;

    for ( int i = 0; i < N; ++i )
    {
      sum1 += x[i] * x[i];
      sum2 += std::cos( c * x[i] );
    }

    sum1 /= static_cast<T>( N );
    sum2 /= static_cast<T>( N );

    return -a * std::exp( -b * std::sqrt( sum1 ) ) - std::exp( sum2 ) + a + std::exp( 1.0 );
  }
};

/**
 * @class RastriginN
 * @brief Rastrigin function, a highly multimodal test function
 *
 * The Rastrigin function is based on the sphere function with the addition
 * of cosine modulation to create many local minima. The global minimum
 * is at the origin with value 0. This function is particularly challenging
 * due to its large number of local minima.
 */
template <typename T, int N>
class RastriginN
{
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  /**
   * @brief Constructor enforcing dimension constraint
   */
  RastriginN() { static_assert( N >= 1, "RastriginN requires N >= 1" ); }

  /**
   * @brief Get lower bounds for variables
   * @return Vector of lower bounds (-5.12 for all variables)
   */
  Vector
  lower() const
  {
    return Vector::Constant( N, -5.12 );
  }

  /**
   * @brief Get upper bounds for variables
   * @return Vector of upper bounds (5.12 for all variables)
   */
  Vector
  upper() const
  {
    return Vector::Constant( N, 5.12 );
  }

  /**
   * @brief Get initial guess
   * @return Initial point (2.0, 2.0, ..., 2.0)
   */
  Vector
  init() const
  {
    return Vector::Constant( N, 2.0 );
  }

  /**
   * @brief Get exact solution
   * @return Global minimum at the origin (0,0,...,0)
   */
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /**
   * @brief Evaluate Rastrigin function
   * @param x Input vector
   * @return Function value at x
   */
  T
  operator()( Vector const & x ) const
  {
    T f = 10.0 * static_cast<T>( N );
    for ( int i = 0; i < N; ++i ) { f += x[i] * x[i] - 10.0 * std::cos( 2.0 * M_PI * x[i] ); }
    return f;
  }
};
