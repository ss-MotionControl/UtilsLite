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

using std::string;

// -------------------------------------------------------------------
// 1. Ackley Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "AckleyN requires N >= 1" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -32.768 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 32.768 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 1.0 );
  }
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /// Objective function
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

    sum1 /= T( N );
    sum2 /= T( N );

    return -a * std::exp( -b * std::sqrt( sum1 ) ) - std::exp( sum2 ) + a + std::exp( 1.0 );
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;
  
    T sum1 = x.squaredNorm();
    T avg_sum1 = sum1 / T(N);
    T sqrt_avg_sum1 = std::sqrt(avg_sum1);
  
    T sum2 = 0.0;
    for ( int i = 0; i < N; ++i ) sum2 += std::cos( c * x[i] );
    T avg_sum2 = sum2 / T(N);
  
    T exp1 = std::exp( -b * sqrt_avg_sum1 );
    T exp2 = std::exp( avg_sum2 );
  
    Vector g( N );
    // Evitiamo divisione per zero nell'origine
    T common_prefix = (sqrt_avg_sum1 > 1e-14) ? (a * b * exp1) / (T(N) * sqrt_avg_sum1) :   0.0;
  
    for ( int i = 0; i < N; ++i )
    {
      T term1 = common_prefix * x[i];
      T term2 = (exp2 * c / T(N)) * std::sin( c * x[i] );
      g[i] = term1 + term2;
    }
    return g;
  }

  SparseMatrix
  hessian(Vector const & x) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve(N * N); // Ackley è densa, riserviamo spazio per tutti gli elementi
  
    const T a = 20.0;
    const T b = 0.2;
    const T c = 2.0 * M_PI;
  
    const T sum1 = x.squaredNorm();
    const T avg_sum1 = sum1 / T(N);
    const T sqrt_avg_sum1 = std::sqrt(avg_sum1);
  
    // Somma dei coseni per il secondo termine
    T sum2 = 0.0;
    for (int i = 0; i < N; ++i) sum2 += std::cos(c * x[i]);
    const T avg_sum2 = sum2 / T(N);
  
    const T exp1 = std::exp(-b * sqrt_avg_sum1);
    const T exp2 = std::exp(avg_sum2);
  
    // Gestione singolarità nell'origine (la radice non è derivabile in 0)
    if (sqrt_avg_sum1 < 1e-12)
    {
      SparseMatrix H(N, N);
      return H; // Ritorna matrice vuota o identità scalata a seconda della necessità
    }
  
    // Costanti pre-calcolate per velocizzare i cicli
    const T common_T1 = (a * b * exp1) / (T(N) * sqrt_avg_sum1);
    const T common_T2 = (c * c * exp2) / T(N);
    const T inv_N = 1.0 / T(N);

    for (int i = 0; i < N; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        // --- Termine 1 (Esponenziale della radice) ---
        // Derivata seconda di: -a * exp(-b * sqrt(1/N * sum(x^2)))
        T Hij_T1 = common_T1 * ( ((i == j) ? 1.0 : 0.0)
                   - (x[i] * x[j]) / (T(N) * avg_sum1)
                   - (b * x[i] * x[j]) / (T(N) * sqrt_avg_sum1) );

        // --- Termine 2 (Coseni) ---
        // Derivata seconda di: -exp(1/N * sum(cos(c * x)))
        T Hij_T2 = common_T2 * ( ((i == j) ? std::cos(c * x[i]) : 0.0)
                   + (std::sin(c * x[i]) * std::sin(c * x[j])) * inv_N );

        T Hij = Hij_T1 + Hij_T2;
    
        // Filtro opzionale: aggiungi alla matrice solo se il valore è significativo
        if (std::abs(Hij) > 1e-18)
        {
          triplets.emplace_back(i, j, Hij);
        }
      }
    }

    SparseMatrix H(N, N);
    H.setFromTriplets(triplets.begin(), triplets.end());
    return H;
  }
};

// -------------------------------------------------------------------
// 2. Beale Function (2D)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -4.5 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 4.5 );
  }

  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 1.0, 1.0;
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 2 );
    x << 3.0, 0.5;
    return x;
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );
    return t1 * t1 + t2 * t2 + t3 * t3;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];

    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );

    Vector g( 2 );
    g[0] = -2.0 * t1 * ( 1.0 - x2 ) - 2.0 * t2 * ( 1.0 - x2 * x2 ) - 2.0 * t3 * ( 1.0 - x2 * x2 * x2 );
    g[1] = 2.0 * t1 * x1 + 4.0 * t2 * x1 * x2 + 6.0 * t3 * x1 * x2 * x2;

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];

    T t1 = 1.5 - x1 * ( 1.0 - x2 );
    T t2 = 2.25 - x1 * ( 1.0 - x2 * x2 );
    T t3 = 2.625 - x1 * ( 1.0 - x2 * x2 * x2 );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 2.0 * ( 1.0 - x2 ) * ( 1.0 - x2 ) + 2.0 * ( 1.0 - x2 * x2 ) * ( 1.0 - x2 * x2 ) +
                       2.0 * ( 1.0 - x2 * x2 * x2 ) * ( 1.0 - x2 * x2 * x2 );
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 2.0 * t1 + 4.0 * t2 * x2 + 6.0 * t3 * x2 * x2;
    H.insert( 1, 1 ) = 2.0 * x1 * x1 + 4.0 * x1 * x1 * x2 * x2 + 4.0 * t2 * x1 + 12.0 * t3 * x1 * x2 +
                       12.0 * x1 * x1 * x2 * x2;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 3. Booth Function (2D)
// -------------------------------------------------------------------

/**
 * @class Booth2D
 * @brief Booth function, a 2D optimization test function
 *
 * The Booth function has a global minimum at (1,3) with value 0.
 * It is a simple quadratic function used for testing optimization algorithms.
 */
template <typename T>
class Booth2D
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }
  Vector
  init() const
  {
    Vector x( 2 );
    x << 1.0, 3.0;
    return x;
  }
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 1.0, 3.0;
    return x;
  }

  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    return std::pow( x1 + 2 * x2 - 7, 2 ) + std::pow( 2 * x1 + x2 - 5, 2 );
  }

  Vector
  gradient( Vector const & x ) const
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    g[0] = 2 * ( x1 + 2 * x2 - 7 ) + 4 * ( 2 * x1 + x2 - 5 );
    g[1] = 4 * ( x1 + 2 * x2 - 7 ) + 2 * ( 2 * x1 + x2 - 5 );
    return g;
  }

  SparseMatrix
  hessian( Vector const & ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    // elementi della Hessiana densa
    triplets.emplace_back( 0, 0, 10.0 );
    triplets.emplace_back( 0, 1, 8.0 );
    triplets.emplace_back( 1, 0, 8.0 );
    triplets.emplace_back( 1, 1, 8.0 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 4. Broyden Tridiagonal Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "BroydenTridiagonalN requires N >= 2" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /// Objective function
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

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g = Vector::Zero( N );
    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? 0.0 : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;

      T dt_dx = 3.0 - 4.0 * x[i];
      g[i] += 2.0 * t * dt_dx;

      if ( i > 0 ) g[i - 1] += 2.0 * t * ( -2.0 );
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.setZero();

    for ( int i = 0; i < N; ++i )
    {
      T xim1 = ( i == 0 ) ? T( 0 ) : x[i - 1];
      T t    = ( 3.0 - 2.0 * x[i] ) * x[i] - 2.0 * xim1 + 1.0;

      T dt_dx   = 3.0 - 4.0 * x[i];
      T d2t_dx2 = -4.0;

      // diagonale i,i
      H.coeffRef( i, i ) += 2.0 * ( dt_dx * dt_dx + t * d2t_dx2 );

      if ( i > 0 )
      {
        // termini fuori diagonale
        T v = 2.0 * dt_dx * ( -2.0 );
        H.coeffRef( i, i - 1 ) += v;
        H.coeffRef( i - 1, i ) += v;

        // diagonale (i-1,i-1)
        H.coeffRef( i - 1, i - 1 ) += 2.0 * 4.0;
      }
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 5. Brown Almost Linear Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "BrownAlmostLinearN requires N >= 2" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -5.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 5.0 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /// Objective function
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

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g = Vector::Zero( N );
    for ( int i = 0; i < N - 1; ++i )
    {
      T t = x[i] + x[i + 1] * x[i + 1] * x[i + 1] - 3.0;
      g[i] += 2.0 * t;
      g[i + 1] += 2.0 * t * 3.0 * x[i + 1] * x[i + 1];
    }
    for ( int i = 0; i < N; ++i ) g[i] += 2e-3 * x[i];
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.setZero();

    for ( int i = 0; i < N - 1; ++i )
    {
      T xi1 = x[i + 1];
      T t   = x[i] + xi1 * xi1 * xi1 - 3.0;

      // d²/dx_i² (t^2)
      H.coeffRef( i, i ) += 2.0;

      // d²/dx_{i+1}² (t^2)
      // = 2 * [ (3 x^2)^2 + t * (6 x) ]
      H.coeffRef( i + 1, i + 1 ) += 2.0 * ( 9.0 * xi1 * xi1 * xi1 * xi1 + 6.0 * t * xi1 );

      // termini misti
      T v = 2.0 * 3.0 * xi1 * xi1;
      H.coeffRef( i, i + 1 ) += v;
      H.coeffRef( i + 1, i ) += v;
    }

    // regolarizzazione
    for ( int i = 0; i < N; ++i ) H.coeffRef( i, i ) += 2e-3;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 6. Extended Wood Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  ExtendedWoodN() { static_assert( N % 4 == 0, "ExtendedWoodN requires N to be a multiple of 4" ); }

  Vector
  lower() const
  {
    return Vector::Constant( N, -3.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 3.0 );
  }

  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? -3.0 : -1.0;
    return x0;
  }

  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /// Objective function
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

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] * x[k1] - x[k2];
      T t2 = x[k1] - 1.0;
      T t3 = x[k3] * x[k3] - x[k4];
      T t4 = x[k3] - 1.0;
      T t5 = x[k2] + x[k4] - 2.0;
      T t6 = x[k2] - x[k4];

      g[k1] += 400.0 * t1 * x[k1] + 2.0 * t2;
      g[k2] += -200.0 * t1 + 20.0 * t5 + 0.2 * t6;
      g[k3] += 180.0 * t3 * x[k3] + 2.0 * t4;
      g[k4] += -90.0 * t3 + 20.0 * t5 - 0.2 * t6;
    }

    return g;
  }

  /// Hessian ∇²f(x) sparse 4x4 blocks
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.reserve( Eigen::VectorXi::Constant( N, 4 ) );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] * x[k1] - x[k2];
      T t3 = x[k3] * x[k3] - x[k4];

      H.coeffRef( k1, k1 ) += 400.0 * t1 + 400.0 * x[k1] * x[k1] + 2.0;
      H.coeffRef( k1, k2 ) += -200.0;
      H.coeffRef( k2, k1 ) += -200.0;
      H.coeffRef( k2, k2 ) += 20.0 + 0.2;
      H.coeffRef( k2, k4 ) += 20.0 - 0.2;
      H.coeffRef( k3, k3 ) += 180.0 * t3 + 180.0 * x[k3] * x[k3] + 2.0;
      H.coeffRef( k3, k4 ) += -90.0;
      H.coeffRef( k4, k3 ) += -90.0;
      H.coeffRef( k4, k4 ) += 20.0 + 0.2;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 7. Freudenstein-Roth Function (2D)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }

  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 0.5, -2.0;
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 2 );
    x << 5.0, 4.0;
    return x;
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = -13.0 + x1 + ( ( 5.0 - x2 ) * x2 - 2.0 ) * x2;
    T f2 = -29.0 + x1 + ( ( x2 + 1.0 ) * x2 - 14.0 ) * x2;
    return f1 * f1 + f2 * f2;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];

    T f1 = -13.0 + x1 + ( ( 5.0 - x2 ) * x2 - 2.0 ) * x2;   // = -13 + x1 + 5x2^2 - x2^3 -2 x2
    T f2 = -29.0 + x1 + ( ( x2 + 1.0 ) * x2 - 14.0 ) * x2;  // = -29 + x1 + x2^3 + x2^2 -14 x2

    Vector g( 2 );
    g[0] = 2.0 * f1 + 2.0 * f2;
    g[1] = 2.0 * f1 * ( 10.0 * x2 - 3.0 * x2 * x2 - 2.0 ) + 2.0 * f2 * ( 3.0 * x2 * x2 + 2.0 * x2 - 14.0 );

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];

    T f1 = -13.0 + x1 + ( ( 5.0 - x2 ) * x2 - 2.0 ) * x2;
    T f2 = -29.0 + x1 + ( ( x2 + 1.0 ) * x2 - 14.0 ) * x2;

    T d2f1_dx2x2 = 10.0 - 6.0 * x2;
    T d2f2_dx2x2 = 6.0 * x2 + 2.0;

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 4.0;
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 2.0 *
                                          ( ( 10.0 * x2 - 3.0 * x2 * x2 - 2.0 ) + ( 3.0 * x2 * x2 + 2.0 * x2 - 14.0 ) );

    T df1_dx2        = 10.0 * x2 - 3.0 * x2 * x2 - 2.0;
    T df2_dx2        = 3.0 * x2 * x2 + 2.0 * x2 - 14.0;
    T h11            = 2.0 * ( df1_dx2 * df1_dx2 + df2_dx2 * df2_dx2 ) + 2.0 * ( f1 * d2f1_dx2x2 + f2 * d2f2_dx2x2 );
    H.insert( 1, 1 ) = h11;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 8. Griewank Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class GriewankN
 * @brief Griewank function, a multimodal test function
 *
 * The Griewank function has many widespread local minima regularly distributed.
 * The global minimum is at the origin with value 0.
 */
template <typename T, int N>
class GriewankN
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( N, -600.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 600.0 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 0.0 );
  }
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  T
  operator()( Vector const & x ) const
  {
    T sum = 0, prod = 1;
    for ( int i = 0; i < N; ++i )
    {
      sum += x[i] * x[i] / 4000;
      prod *= std::cos( x[i] / std::sqrt( T( i + 1 ) ) );
    }
    return sum - prod + 1;
  }

  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    T      prod_all = 1;
    for ( int i = 0; i < N; ++i ) prod_all *= std::cos( x[i] / std::sqrt( T( i + 1 ) ) );

    for ( int i = 0; i < N; ++i )
    {
      T term   = 2.0 * x[i] / 4000.0;
      T prod_i = prod_all / std::cos( x[i] / std::sqrt( T( i + 1 ) ) ) * std::sin( x[i] / std::sqrt( T( i + 1 ) ) ) /
                 std::sqrt( T( i + 1 ) );
      g[i] = term + prod_i;
    }
    return g;
  }

  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale non-zero

    // prodotto complessivo
    T prod_all = 1.0;
    for ( int i = 0; i < N; ++i ) prod_all *= std::cos( x[i] / std::sqrt( T( i + 1 ) ) );

    for ( int i = 0; i < N; ++i )
    {
      T xi    = x[i];
      T si    = std::sin( xi / std::sqrt( T( i + 1 ) ) );
      T ci    = std::cos( xi / std::sqrt( T( i + 1 ) ) );
      T sqrti = std::sqrt( T( i + 1 ) );

      // elemento diagonale
      T Hii = 1.0 / 2000.0 + prod_all * ( si * si / ( sqrti * sqrti ) - ci / ( sqrti * sqrti ) );
      triplets.emplace_back( i, i, Hii );

      // off-diagonal nulla, quindi non serve inserirla (sparse)
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 9. Helical Valley Function (3D)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  // Pi greco costante
  const T PI = std::acos( T( -1.0 ) );

  Vector
  lower() const
  {
    return Vector::Constant( 3, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 3, 10.0 );
  }

  Vector
  init() const
  {
    Vector x0( 3 );
    x0 << T( -1.0 ), T( 0.0 ), T( 0.0 );
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 3 );
    x << T( 1.0 ), T( 0.0 ), T( 0.0 );
    return x;
  }

  /// Objective function f(x)
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T theta;

    if ( x1 > 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
    else if ( x1 < 0 )
      theta = ( std::atan( x2 / x1 ) / ( 2.0 * PI ) ) + 0.5;
    else
      theta = ( x2 >= 0 ) ? 0.25 : -0.25;

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( std::sqrt( x1 * x1 + x2 * x2 ) - 1.0 );
    T f3 = x3;

    return f1 * f1 + f2 * f2 + f3 * f3;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T r2 = x1 * x1 + x2 * x2;
    T r  = std::sqrt( r2 );

    T theta, dtheta_dx1, dtheta_dx2;
    if ( r2 > 1e-12 )
    {  // Evitiamo divisione per zero all'origine
      if ( x1 > 0 )
        theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
      else if ( x1 < 0 )
        theta = std::atan( x2 / x1 ) / ( 2.0 * PI ) + 0.5;
      else
        theta = ( x2 >= 0 ) ? 0.25 : -0.25;

      dtheta_dx1 = -x2 / ( 2.0 * PI * r2 );
      dtheta_dx2 = x1 / ( 2.0 * PI * r2 );
    }
    else
    {
      theta      = 0;
      dtheta_dx1 = 0;
      dtheta_dx2 = 0;
    }

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( r - 1.0 );

    // Derivate componenti:
    // df1/dx1 = 10 * (-10 * dtheta/dx1) = -100 * dtheta/dx1
    // df2/dx1 = 10 * (x1/r)

    Vector g( 3 );
    g[0] = 2.0 * f1 * ( -100.0 * dtheta_dx1 ) + 2.0 * f2 * ( 10.0 * x1 / r );
    g[1] = 2.0 * f1 * ( -100.0 * dtheta_dx2 ) + 2.0 * f2 * ( 10.0 * x2 / r );
    g[2] = 2.0 * f1 * 10.0 + 2.0 * x3;

    return g;
  }

  /// Hessian ∇²f(x)
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1], x3 = x[2];
    T r2 = x1 * x1 + x2 * x2;
    T r  = std::sqrt( r2 );

    T dtheta_dx1 = -x2 / ( 2.0 * PI * r2 );
    T dtheta_dx2 = x1 / ( 2.0 * PI * r2 );
    T theta;
    if ( x1 > 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI );
    else if ( x1 < 0 )
      theta = std::atan( x2 / x1 ) / ( 2.0 * PI ) + 0.5;
    else
      theta = ( x2 >= 0 ) ? 0.25 : -0.25;

    T f1 = 10.0 * ( x3 - 10.0 * theta );
    T f2 = 10.0 * ( r - 1.0 );

    // Derivate seconde di theta
    T d2theta_dx1dx1 = ( 2.0 * x1 * x2 ) / ( 2.0 * PI * r2 * r2 );
    T d2theta_dx2dx2 = ( -2.0 * x1 * x2 ) / ( 2.0 * PI * r2 * r2 );
    T d2theta_dx1dx2 = ( x1 * x1 - x2 * x2 ) / ( 2.0 * PI * r2 * r2 );

    SparseMatrix H( 3, 3 );
    H.reserve( Eigen::VectorXi::Constant( 3, 3 ) );

    // f1 = 10(x3 - 10*theta) -> df1/dx1 = -100 * dtheta_dx1
    T df1_dx1 = -100.0 * dtheta_dx1;
    T df1_dx2 = -100.0 * dtheta_dx2;
    T df1_dx3 = 10.0;

    // f2 = 10(r - 1) -> df2/dx1 = 10 * x1/r
    T df2_dx1 = 10.0 * x1 / r;
    T df2_dx2 = 10.0 * x2 / r;

    // H(0,0)
    H.insert( 0, 0 ) = 2.0 * ( df1_dx1 * df1_dx1 + f1 * ( -100.0 * d2theta_dx1dx1 ) + df2_dx1 * df2_dx1 +
                               f2 * ( 10.0 * ( r2 - x1 * x1 ) / ( r * r2 ) ) );

    // H(1,1)
    H.insert( 1, 1 ) = 2.0 * ( df1_dx2 * df1_dx2 + f1 * ( -100.0 * d2theta_dx2dx2 ) + df2_dx2 * df2_dx2 +
                               f2 * ( 10.0 * ( r2 - x2 * x2 ) / ( r * r2 ) ) );

    // H(2,2)
    H.insert( 2, 2 ) = 2.0 * ( df1_dx3 * df1_dx3 ) + 2.0;  // (2 * 10^2) + 2 = 202

    // H(0,1) Simmetrica
    T h01            = 2.0 * ( df1_dx1 * df1_dx2 + f1 * ( -100.0 * d2theta_dx1dx2 ) + df2_dx1 * df2_dx2 +
                    f2 * ( -10.0 * x1 * x2 / ( r * r2 ) ) );
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = h01;

    // H(0,2) e H(1,2)
    H.insert( 0, 2 ) = H.insert( 2, 0 ) = 2.0 * df1_dx1 * 10.0;
    H.insert( 1, 2 ) = H.insert( 2, 1 ) = 2.0 * df1_dx2 * 10.0;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 10. Himmelblau Function (2D)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -6.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 6.0 );
  }

  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << -3.0, -3.0;
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 2 );
    x << 3.0, 2.0;
    return x;
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;
    return f1 * f1 + f2 * f2;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;

    Vector g( 2 );
    g[0] = 4.0 * x1 * f1 + 2.0 * f2;
    g[1] = 2.0 * f1 + 4.0 * x2 * f2;

    return g;
  }

  /// Hessian ∇²f(x) sparse 2x2
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = x1 * x1 + x2 - 11.0;
    T f2 = x1 + x2 * x2 - 7.0;

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 4.0 * f1 + 8.0 * x1 * x1 + 2.0;
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 4.0 * x1 + 4.0 * x2;
    H.insert( 1, 1 )                    = 2.0 + 4.0 * f2 + 8.0 * x2 * x2;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 11. Ill-conditioned Quadratic Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 2, "IllConditionedQuadraticN requires N >= 2" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? 1.0 : -1.0;
    return x0;
  }

  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /// Objective function
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

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T lambda = std::pow( 1e6, T( i ) / T( N - 1 ) );
      g[i]     = 2.0 * lambda * x[i];
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN (diagonal)
  SparseMatrix
  hessian( Vector const & ) const
  {
    SparseMatrix H( N, N );
    H.reserve( N );

    for ( int i = 0; i < N; ++i )
    {
      T lambda         = std::pow( 1e6, T( i ) / T( N - 1 ) );
      H.insert( i, i ) = 2.0 * lambda;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 12. Levy Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class LevyN
 * @brief Levy function, a multimodal test function
 *
 * The Levy function is characterized by its many local minima.
 * The global minimum is at (1,1,...,1) with value 0.
 */
template <typename T, int N>
class LevyN
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 1.0 );
  }
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  T
  operator()( Vector const & x ) const
  {
    auto w     = [&]( int i ) { return 1 + ( x[i] - 1 ) / 4; };
    T    term1 = std::pow( std::sin( M_PI * w( 0 ) ), 2 );
    T    sum   = 0;
    for ( int i = 0; i < N - 1; ++i )
      sum += std::pow( w( i ) - 1, 2 ) * ( 1 + 10 * std::pow( std::sin( M_PI * w( i + 1 ) ), 2 ) );
    T termN = std::pow( w( N - 1 ) - 1, 2 ) * ( 1 + std::pow( std::sin( 2 * M_PI * w( N - 1 ) ), 2 ) );
    return term1 + sum + termN;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g  = Vector::Zero( N );
    auto   w  = [&]( int i ) { return 1 + ( x[i] - 1 ) / 4; };
    auto   dw = []( int ) { return 0.25; };

    for ( int i = 0; i < N; ++i )
    {
      T wi  = w( i );
      T dwi = dw( i );
      if ( i == 0 ) { g[i] += 2 * std::sin( M_PI * wi ) * std::cos( M_PI * wi ) * M_PI * dwi; }
      if ( i < N - 1 )
      {
        T wip1  = w( i + 1 );
        T dwip1 = dw( i + 1 );
        g[i] += 2 * ( wi - 1 ) * dwi * ( 1 + 10 * std::pow( std::sin( M_PI * wip1 ), 2 ) );
        g[i + 1] += std::pow( wi - 1, 2 ) * 10 * 2 * std::sin( M_PI * wip1 ) * std::cos( M_PI * wip1 ) * M_PI * dwip1;
      }
      if ( i == N - 1 )
      {
        g[i] += 2 * ( wi - 1 ) * dwi * ( 1 + std::pow( std::sin( 2 * M_PI * wi ), 2 ) ) +
                std::pow( wi - 1, 2 ) * 2 * std::sin( 2 * M_PI * wi ) * std::cos( 2 * M_PI * wi ) * 2 * M_PI * dwi;
      }
    }
    return g;
  }

  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    auto w  = [&]( int i ) { return 1 + ( x[i] - 1 ) / 4; };
    auto dw = []( int ) { return 0.25; };

    for ( int i = 0; i < N; ++i )
    {
      T wi  = w( i );
      T dwi = dw( i );

      // diagonale i
      T Hii = 0.0;
      if ( i == 0 )
      {
        Hii += 2 * M_PI * dwi * M_PI * dwi *
               ( std::cos( M_PI * wi ) * std::cos( M_PI * wi ) - std::sin( M_PI * wi ) * std::sin( M_PI * wi ) );
      }
      if ( i < N - 1 )
      {
        T wip1  = w( i + 1 );
        T dwip1 = dw( i + 1 );
        Hii += 2 * dwi * dwi * ( 1 + 10 * std::pow( std::sin( M_PI * wip1 ), 2 ) );
        // off-diagonal (i,i+1)
        T Hij = 2 * ( wi - 1 ) * dwi * 10 * 2 * std::sin( M_PI * wip1 ) * std::cos( M_PI * wip1 ) * M_PI * dwip1;
        triplets.emplace_back( i, i + 1, Hij );
        triplets.emplace_back( i + 1, i, Hij );  // simmetrico
        // contributo diagonale i+1
        triplets.emplace_back(
          i + 1,
          i + 1,
          std::pow( wi - 1, 2 ) * 10 * 2 * M_PI * dwip1 * M_PI * dwip1 *
            ( std::cos( M_PI * wip1 ) * std::cos( M_PI * wip1 ) - std::sin( M_PI * wip1 ) * std::sin( M_PI * wip1 ) ) );
      }
      if ( i == N - 1 )
      {
        Hii += 2 * dwi * dwi * ( 1 + std::pow( std::sin( 2 * M_PI * wi ), 2 ) ) +
               4 * ( wi - 1 ) * dwi * dwi * 2 * std::sin( 2 * M_PI * wi ) * std::cos( 2 * M_PI * wi ) * 2 * M_PI +
               std::pow( wi - 1, 2 ) * 2 * ( 2 * M_PI * dwi ) * ( 2 * M_PI * dwi ) *
                 ( std::cos( 2 * M_PI * wi ) * std::cos( 2 * M_PI * wi ) -
                   std::sin( 2 * M_PI * wi ) * std::sin( 2 * M_PI * wi ) );
      }

      triplets.emplace_back( i, i, Hii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 13. Matyas Function (2D)
// -------------------------------------------------------------------

/**
 * @class Matyas2D
 * @brief Matyas function, a simple 2D test function
 *
 * The Matyas function has a global minimum at (0,0) with value 0.
 * It is a simple quadratic function used for testing optimization algorithms.
 */
template <typename T>
class Matyas2D
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }
  Vector
  init() const
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    return 0.26 * ( x1 * x1 + x2 * x2 ) - 0.48 * x1 * x2;
  }

  Vector
  gradient( Vector const & x ) const
  {
    Vector g( 2 );
    g[0] = 0.52 * x[0] - 0.48 * x[1];
    g[1] = 0.52 * x[1] - 0.48 * x[0];
    return g;
  }

  SparseMatrix
  hessian( Vector const & ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    // elementi della Hessiana 2x2
    triplets.emplace_back( 0, 0, 0.52 );
    triplets.emplace_back( 0, 1, -0.48 );
    triplets.emplace_back( 1, 0, -0.48 );
    triplets.emplace_back( 1, 1, 0.52 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 14. McCormick Function (2D)
// -------------------------------------------------------------------

/**
 * @class McCormick2D
 * @brief McCormick function, a 2D optimization test function
 *
 * The McCormick function has a global minimum at (-0.54719,-1.54719) with value
 * -1.9133. It features both trigonometric and quadratic terms.
 */
template <typename T>
class McCormick2D
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    Vector x( 2 );
    x << -1.5, -3.0;
    return x;
  }
  Vector
  upper() const
  {
    Vector x( 2 );
    x << 4.0, 4.0;
    return x;
  }
  Vector
  init() const
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }
  Vector
  exact() const
  {
    Vector x( 2 );
    x << -0.54719, -1.54719;
    return x;
  }

  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    return std::sin( x1 + x2 ) + ( x1 - x2 ) * ( x1 - x2 ) - 1.5 * x1 + 2.5 * x2 + 1;
  }

  Vector
  gradient( Vector const & x ) const
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    g[0] = std::cos( x1 + x2 ) + 2 * ( x1 - x2 ) - 1.5;
    g[1] = std::cos( x1 + x2 ) - 2 * ( x1 - x2 ) + 2.5;
    return g;
  }

  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    T s = -std::sin( x[0] + x[1] );

    // diagonali
    triplets.emplace_back( 0, 0, s + 2.0 );
    triplets.emplace_back( 1, 1, s + 2.0 );

    // off-diagonale
    triplets.emplace_back( 0, 1, s - 2.0 );
    triplets.emplace_back( 1, 0, s - 2.0 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 15. Michalewicz Function (N-dimensional)
// -------------------------------------------------------------------

/**
 * @class MichalewiczN
 * @brief Michalewicz function, a multimodal test function
 *
 * The Michalewicz function has many local minima and a global minimum
 * that is difficult to find. The function is characterized by steep
 * valleys and ridges.
 */
template <typename T, int N>
class MichalewiczN
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  int m = 10;  // common parameter
  Vector
  lower() const
  {
    return Vector::Zero( N );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, M_PI );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, M_PI / 2 );
  }
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }  // approximate value

  T
  operator()( Vector const & x ) const
  {
    T f = 0;
    for ( int i = 0; i < N; ++i ) f -= std::sin( x[i] ) * std::pow( std::sin( ( i + 1 ) * x[i] * x[i] / M_PI ), 2 * m );
    return f;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T xi    = x[i];
      T a     = ( i + 1 ) * xi * xi / M_PI;
      T sina  = std::sin( a );
      T cosa  = std::cos( a );
      T term1 = std::cos( xi ) * std::pow( sina, 2 * m );
      T term2 = std::sin( xi ) * 2 * m * std::pow( sina, 2 * m - 1 ) * cosa * ( 2 * ( i + 1 ) * xi / M_PI );
      g[i]    = -term1 - term2;
    }
    return g;
  }

  /// Hessian ∇²f(x) diagonal approximation
  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale non-zero

    for ( int i = 0; i < N; ++i )
    {
      T xi   = x[i];
      T a    = ( i + 1 ) * xi * xi / M_PI;
      T sina = std::sin( a );
      T cosa = std::cos( a );

      T d2 = -std::sin( xi ) * std::pow( sina, 2 * m ) -
             2 * std::cos( xi ) * 2 * m * std::pow( sina, 2 * m - 1 ) * cosa * ( 2 * ( i + 1 ) * xi / M_PI ) -
             std::sin( xi ) * 2 * m *
               ( ( 2 * m - 1 ) * std::pow( sina, 2 * m - 2 ) * cosa * cosa * ( 2 * ( i + 1 ) * xi / M_PI ) *
                   ( 2 * ( i + 1 ) * xi / M_PI ) +
                 std::pow( sina, 2 * m - 1 ) * ( -sina ) * ( 2 * ( i + 1 ) * xi / M_PI ) *
                   ( 2 * ( i + 1 ) * xi / M_PI ) +
                 std::pow( sina, 2 * m - 1 ) * cosa * ( 2 * ( i + 1 ) / M_PI ) );

      triplets.emplace_back( i, i, -d2 );  // solo diagonale
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 16. Nesterov-Chebyshev-Rosenbrock Function (N-dimensional)
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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  NesterovChebyshevRosenbrock() { static_assert( N >= 2, "NesterovChebyshevRosenbrock requires N >= 2" ); }

  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, -1.0 );
  }
  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T f = T( 0.25 ) * std::abs( x[0] - T( 1.0 ) );
    for ( int i = 0; i < N - 1; ++i )
    {
      T term = x[i + 1] - T( 2.0 ) * std::abs( x[i] ) + T( 1.0 );
      f += std::abs( term );
    }
    return f;
  }

  /// Subgradient ∂f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g    = Vector::Zero( N );
    auto   sign = []( T v ) -> T { return ( v > T( 0 ) ) - ( v < T( 0 ) ); };

    g[0] += T( 0.25 ) * sign( x[0] - T( 1.0 ) );
    for ( int i = 0; i < N - 1; ++i )
    {
      T term = x[i + 1] - T( 2.0 ) * std::abs( x[i] ) + T( 1.0 );
      T s    = sign( term );
      g[i + 1] += s;
      g[i] += -T( 2.0 ) * sign( x[i] ) * s;
    }
    return g;
  }

  /// Hessian (zero almost everywhere)
  SparseMatrix
  hessian( Vector const & ) const
  {
    SparseMatrix H( N, N );
    H.setZero();
    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 17. Powell Badly Scaled Function (2D)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 10.0 );
  }

  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << 0.0, 1.0;
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 2 );
    x << 1.098159329699759e-5, 9.106146739867318;
    return x;
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;
    return f1 * f1 + f2 * f2;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;

    Vector g( 2 );
    g[0] = 2.0 * f1 * ( 1e4 * x2 ) + 2.0 * f2 * ( -std::exp( -x1 ) );
    g[1] = 2.0 * f1 * ( 1e4 * x1 ) + 2.0 * f2 * ( -std::exp( -x2 ) );
    return g;
  }

  /// Hessian ∇²f(x) sparse 2×2
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp( -x1 ) + std::exp( -x2 ) - 1.0001;

    T df1_dx1 = 1e4 * x2;
    T df1_dx2 = 1e4 * x1;
    T df2_dx1 = -std::exp( -x1 );
    T df2_dx2 = -std::exp( -x2 );

    T d2f1_dx1dx1 = 0.0;
    T d2f1_dx1dx2 = 1e4;
    T d2f1_dx2dx2 = 0.0;

    T d2f2_dx1dx1 = std::exp( -x1 );
    T d2f2_dx1dx2 = 0.0;
    T d2f2_dx2dx2 = std::exp( -x2 );

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    H.insert( 0, 0 ) = 2.0 * ( df1_dx1 * df1_dx1 + f1 * d2f1_dx1dx1 + df2_dx1 * df2_dx1 + f2 * d2f2_dx1dx1 );
    H.insert( 0, 1 ) = H.insert( 1, 0 ) = 2.0 * ( df1_dx1 * df1_dx2 + f1 * d2f1_dx1dx2 + df2_dx1 * df2_dx2 +
                                                  f2 * d2f2_dx1dx2 );
    H.insert( 1, 1 ) = 2.0 * ( df1_dx2 * df1_dx2 + f1 * d2f1_dx2dx2 + df2_dx2 * df2_dx2 + f2 * d2f2_dx2dx2 );

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 18. Powell Singular Function (N-dimensional)
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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  PowellSingularN() { static_assert( N % 4 == 0, "PowellSingularN requires N to be a multiple of 4" ); }

  Vector
  lower() const
  {
    return Vector::Constant( N, -4.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 4.0 );
  }

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

  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T f = T( 0 );
    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      g[k1] += 2.0 * t1 + 40.0 * t4 * t4 * t4;
      g[k2] += 20.0 * t1 + 4.0 * t3 * t3 * t3;
      g[k3] += 10.0 * t2 - 8.0 * t3 * t3 * t3;
      g[k4] += -10.0 * t2 - 40.0 * t4 * t4 * t4;
    }

    return g;
  }

  /// Hessian ∇²f(x) (sparse, 4x4 blocks)
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.reserve( Eigen::VectorXi::Constant( N, 4 ) );

    for ( int i = 0; i < N / 4; ++i )
    {
      int k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      H.coeffRef( k1, k1 ) += 2.0 + 120.0 * t4 * t4;
      H.coeffRef( k1, k2 ) += 20.0;
      H.coeffRef( k1, k4 ) += -120.0 * t4 * t4;

      H.coeffRef( k2, k1 ) += 20.0;
      H.coeffRef( k2, k2 ) += 200.0 + 12.0 * t3 * t3;
      H.coeffRef( k2, k3 ) += -24.0 * t3 * t3;

      H.coeffRef( k3, k2 ) += -24.0 * t3 * t3;
      H.coeffRef( k3, k3 ) += 10.0 + 48.0 * t3 * t3;
      H.coeffRef( k3, k4 ) += -10.0;

      H.coeffRef( k4, k1 ) += -120.0 * t4 * t4;
      H.coeffRef( k4, k3 ) += -10.0;
      H.coeffRef( k4, k4 ) += 10.0 + 120.0 * t4 * t4;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 19. Rastrigin Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "RastriginN requires N >= 1" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -5.12 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 5.12 );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 2.0 );
  }
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T f = 10.0 * T( N );
    for ( int i = 0; i < N; ++i ) f += x[i] * x[i] - 10.0 * std::cos( 2.0 * M_PI * x[i] );
    return f;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i ) g[i] = 2.0 * x[i] + 20.0 * M_PI * std::sin( 2.0 * M_PI * x[i] );
    return g;
  }

  /// Hessian ∇²f(x) (diagonal NxN)
  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale non-zero

    for ( int i = 0; i < N; ++i )
    {
      T Hii = 2.0 + 40.0 * M_PI * M_PI * std::cos( 2.0 * M_PI * x[i] );
      triplets.emplace_back( i, i, Hii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 20. Rosenbrock Function (2D)
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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -std::numeric_limits<T>::infinity() );
  }

  Vector
  upper() const
  {
    return Vector::Constant( 2, std::numeric_limits<T>::infinity() );
  }

  Vector
  init() const
  {
    Vector x0( 2 );
    x0 << -1.2, 1.0;
    return x0;
  }

  Vector
  exact() const
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T a = 1.0, b = 100.0;
    T t1 = a - x[0];
    T t2 = x[1] - x[0] * x[0];
    return t1 * t1 + b * t2 * t2;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    T a = 1.0, b = 100.0;

    Vector g( 2 );
    g[0] = -2.0 * ( a - x[0] ) - 4.0 * b * x[0] * ( x[1] - x[0] * x[0] );
    g[1] = 2.0 * b * ( x[1] - x[0] * x[0] );

    return g;
  }

  /// Hessian ∇²f(x) sparse
  SparseMatrix
  hessian( Vector const & x ) const
  {
    T b = 100.0;

    SparseMatrix H( 2, 2 );
    H.reserve( Eigen::VectorXi::Constant( 2, 2 ) );

    T h00 = 2.0 - 4.0 * b * ( x[1] - 3.0 * x[0] * x[0] );
    T h01 = -4.0 * b * x[0];
    T h11 = 2.0 * b;

    H.insert( 0, 0 ) = h00;
    H.insert( 0, 1 ) = h01;
    H.insert( 1, 0 ) = h01;
    H.insert( 1, 1 ) = h11;

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 21. Rosenbrock Function (N-dimensional)
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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  RosenbrockN() { static_assert( N >= 2, "RosenbrockN requires N >= 2" ); }

  Vector
  lower() const
  {
    return Vector::Constant( N, -10.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 10.0 );
  }

  Vector
  init() const
  {
    Vector x0 = Vector::Constant( N, -1.0 );
    x0[N - 1] = 1.0;
    return x0;
  }

  Vector
  exact() const
  {
    return Vector::Constant( N, 1.0 );
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T f = T( 0 );
    for ( int i = 0; i < N - 1; ++i )
    {
      T t1 = T( 1.0 ) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      f += t1 * t1 + T( 100.0 ) * t2 * t2;
    }
    return f;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g = Vector::Zero( N );

    for ( int i = 0; i < N - 1; ++i )
    {
      T t1 = T( 1.0 ) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];

      g[i] += -2.0 * t1 - 400.0 * x[i] * t2;
      g[i + 1] += 200.0 * t2;
    }

    return g;
  }

  /// Hessian ∇²f(x) sparse
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.reserve( Eigen::VectorXi::Constant( N, 3 ) );

    for ( int i = 0; i < N - 1; ++i )
    {
      T t2 = x[i + 1] - x[i] * x[i];

      T h_ii    = 2.0 - 400.0 * ( t2 - 2.0 * x[i] * x[i] );
      T h_iip1  = -400.0 * x[i];
      T h_ip1p1 = 200.0;

      H.coeffRef( i, i ) += h_ii;
      H.coeffRef( i, i + 1 ) += h_iip1;
      H.coeffRef( i + 1, i ) += h_iip1;
      H.coeffRef( i + 1, i + 1 ) += h_ip1p1;
    }

    H.makeCompressed();
    return H;
  }
};

// -------------------------------------------------------------------
// 22. Schaffer Function (2D)
// -------------------------------------------------------------------

/**
 * @class Schaffer2D
 * @brief Schaffer function, a 2D optimization test function
 *
 * The Schaffer function has a global minimum at (0,0) with value 0.
 * It features many local minima and is used for testing global optimization.
 */
template <typename T>
class Schaffer2D
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -100.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 100.0 );
  }
  Vector
  init() const
  {
    Vector x( 2 );
    x << 1.0, 1.0;
    return x;
  }
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T u   = x1 * x1 - x2 * x2;
    T v   = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );
    T num = std::pow( std::sin( u ), 2 ) - 0.5;
    T den = v * v;
    return 0.5 + num / den;
  }

  Vector
  gradient( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T u = x1 * x1 - x2 * x2;
    T v = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );

    T sin_u = std::sin( u );
    T cos_u = std::cos( u );

    T num = sin_u * sin_u - 0.5;
    T den = v * v;

    // Derivate del numeratore e denominatore
    T dnum_dx1 = 2.0 * sin_u * cos_u * ( 2.0 * x1 );
    T dnum_dx2 = 2.0 * sin_u * cos_u * ( -2.0 * x2 );
    T dden_dx1 = 2.0 * v * ( 0.002 * x1 );
    T dden_dx2 = 2.0 * v * ( 0.002 * x2 );

    Vector g( 2 );
    g[0] = ( dnum_dx1 * den - num * dden_dx1 ) / ( den * den );
    g[1] = ( dnum_dx2 * den - num * dden_dx2 ) / ( den * den );
    return g;
  }

  SparseMatrix
  hessian( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    T u = x1 * x1 - x2 * x2;
    T v = 1.0 + 0.001 * ( x1 * x1 + x2 * x2 );

    T sin_u  = std::sin( u );
    T cos_u  = std::cos( u );
    T sin_2u = 2.0 * sin_u * cos_u;
    T cos_2u = std::cos( 2.0 * u );

    T num = sin_u * sin_u - 0.5;
    T den = v * v;

    // Derivate prime componenti
    T du_dx1 = 2.0 * x1;
    T du_dx2 = -2.0 * x2;
    T dv_dx1 = 0.002 * x1;
    T dv_dx2 = 0.002 * x2;

    // Derivate prime num e den
    T dnum_dx1 = sin_2u * du_dx1;
    T dnum_dx2 = sin_2u * du_dx2;
    T dden_dx1 = 2.0 * v * dv_dx1;
    T dden_dx2 = 2.0 * v * dv_dx2;

    // Derivate seconde numeratore
    // d/dx(sin_2u * du_dx) = 2*cos_2u*du_dx^2 + sin_2u*d2u_dx2
    T d2num_dx1dx1 = 2.0 * cos_2u * du_dx1 * du_dx1 + sin_2u * 2.0;
    T d2num_dx2dx2 = 2.0 * cos_2u * du_dx2 * du_dx2 + sin_2u * ( -2.0 );
    T d2num_dx1dx2 = 2.0 * cos_2u * du_dx1 * du_dx2;  // d2u/dx1dx2 è 0

    // Derivate seconde denominatore
    // d/dx(2*v*dv_dx) = 2*(dv_dx^2 + v*d2v_dx2)
    T d2den_dx1dx1 = 2.0 * ( dv_dx1 * dv_dx1 + v * 0.002 );
    T d2den_dx2dx2 = 2.0 * ( dv_dx2 * dv_dx2 + v * 0.002 );
    T d2den_dx1dx2 = 2.0 * ( dv_dx1 * dv_dx2 );  // d2v/dx1dx2 è 0

    // Applichiamo la regola del quoziente per l'Hessiano
    auto compute_hessian_element = [&]( T f, T f_i, T f_j, T f_ij, T g, T g_i, T g_j, T g_ij )
    {
      T den2 = g * g;
      T den3 = den2 * g;
      return ( f_ij * g - f * g_ij - f_i * g_j - f_j * g_i ) / den2 + ( 2.0 * f * g_i * g_j ) / den3;
    };

    T H00 = compute_hessian_element( num, dnum_dx1, dnum_dx1, d2num_dx1dx1, den, dden_dx1, dden_dx1, d2den_dx1dx1 );
    T H11 = compute_hessian_element( num, dnum_dx2, dnum_dx2, d2num_dx2dx2, den, dden_dx2, dden_dx2, d2den_dx2dx2 );
    T H01 = compute_hessian_element( num, dnum_dx1, dnum_dx2, d2num_dx1dx2, den, dden_dx1, dden_dx2, d2den_dx1dx2 );

    std::vector<Eigen::Triplet<T>> triplets;
    triplets.emplace_back( 0, 0, H00 );
    triplets.emplace_back( 1, 1, H11 );
    triplets.emplace_back( 0, 1, H01 );
    triplets.emplace_back( 1, 0, H01 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};
// -------------------------------------------------------------------
// 23. Schwefel Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "SchwefelN requires N >= 1" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -500.0 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, 500.0 );
  }

  Vector
  init() const
  {
    Vector x0( N );
    for ( int i = 0; i < N; ++i ) x0[i] = ( i % 2 == 0 ) ? 100.0 : -100.0;
    return x0;
  }

  Vector
  exact() const
  {
    return Vector::Constant( N, 420.9687 );
  }

  /// Objective function
  T
  operator()( Vector const & x ) const
  {
    T f = 418.9829 * static_cast<T>( N );
    for ( int i = 0; i < N; ++i ) f -= x[i] * std::sin( std::sqrt( std::abs( x[i] ) ) );
    return f;
  }

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T xi       = x[i];
      T sgn      = ( xi >= 0 ) ? 1.0 : -1.0;
      T sqrt_abs = std::sqrt( std::abs( xi ) );
      if ( xi != 0 )
        g[i] = -std::sin( sqrt_abs ) - 0.5 * sgn * std::cos( sqrt_abs ) / sqrt_abs * xi;
      else
        g[i] = 0.0;
    }
    return g;
  }

  /// Hessian ∇²f(x) (diagonal, NxN)
  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;
    triplets.reserve( N );  // solo diagonale

    for ( int i = 0; i < N; ++i )
    {
      T xi  = x[i];
      T Hii = 0.0;

      if ( xi != 0 )
      {
        T abs_x  = std::abs( xi );
        T sqrt_x = std::sqrt( abs_x );
        T sign_x = ( xi > 0 ) ? 1.0 : -1.0;
        T term1  = -0.5 * sign_x * ( std::cos( sqrt_x ) / sqrt_x );
        T term2  = 0.25 * xi / abs_x * std::sin( sqrt_x ) / sqrt_x;
        Hii      = term1 + term2;
      }

      triplets.emplace_back( i, i, Hii );
    }

    SparseMatrix H( N, N );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 24. Three-Hump Camel Function (2D)
// -------------------------------------------------------------------

/**
 * @class ThreeHumpCamel2D
 * @brief Three-hump camel function, a 2D test function
 *
 * The three-hump camel function has three local minima and a global minimum
 * at (0,0) with value 0.
 */
template <typename T>
class ThreeHumpCamel2D
{
public:
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  Vector
  lower() const
  {
    return Vector::Constant( 2, -3 );
  }
  Vector
  upper() const
  {
    return Vector::Constant( 2, 3 );
  }
  Vector
  init() const
  {
    Vector x( 2 );
    x << -1, 2;
    return x;
  }
  Vector
  exact() const
  {
    Vector x( 2 );
    x << 0.0, 0.0;
    return x;
  }

  T
  operator()( Vector const & x ) const
  {
    T x1 = x[0], x2 = x[1];
    return 2 * x1 * x1 - 1.05 * std::pow( x1, 4 ) + std::pow( x1, 6 ) / 6 + x1 * x2 + x2 * x2;
  }

  Vector
  gradient( Vector const & x ) const
  {
    T      x1 = x[0], x2 = x[1];
    Vector g( 2 );
    g[0] = 4 * x1 - 4.2 * std::pow( x1, 3 ) + std::pow( x1, 5 ) + x2;
    g[1] = x1 + 2 * x2;
    return g;
  }

  SparseMatrix
  hessian( Vector const & x ) const
  {
    using Triplet = Eigen::Triplet<T>;
    std::vector<Triplet> triplets;

    T x1 = x[0];

    // diagonale e off-diagonale
    triplets.emplace_back( 0, 0, 4 - 12.6 * x1 * x1 + 5 * x1 * x1 * x1 * x1 );
    triplets.emplace_back( 0, 1, 1 );
    triplets.emplace_back( 1, 0, 1 );
    triplets.emplace_back( 1, 1, 2 );

    SparseMatrix H( 2, 2 );
    H.setFromTriplets( triplets.begin(), triplets.end() );
    return H;
  }
};

// -------------------------------------------------------------------
// 25. Trigonometric Sum Function (N-dimensional)
// -------------------------------------------------------------------

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
  using Vector       = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using SparseMatrix = Eigen::SparseMatrix<T>;

  static_assert( N >= 1, "TrigonometricSumN requires N >= 1" );

  Vector
  lower() const
  {
    return Vector::Constant( N, -M_PI );
  }
  Vector
  upper() const
  {
    return Vector::Constant( N, M_PI );
  }
  Vector
  init() const
  {
    return Vector::Constant( N, 0.5 );
  }
  Vector
  exact() const
  {
    return Vector::Zero( N );
  }

  /// Objective function
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

  /// Gradient ∇f(x)
  Vector
  gradient( Vector const & x ) const
  {
    Vector g( N );
    for ( int i = 0; i < N; ++i )
    {
      T idx   = T( i + 1 );
      T t     = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      T dt_dx = std::cos( x[i] ) + idx * std::sin( x[i] );
      g[i]    = 2.0 * t * dt_dx;
    }
    return g;
  }

  /// Hessian ∇²f(x) sparse NxN (diagonal)
  SparseMatrix
  hessian( Vector const & x ) const
  {
    SparseMatrix H( N, N );
    H.reserve( N );

    for ( int i = 0; i < N; ++i )
    {
      T idx            = T( i + 1 );
      T t              = std::sin( x[i] ) + idx * ( 1.0 - std::cos( x[i] ) );
      T dt_dx          = std::cos( x[i] ) + idx * std::sin( x[i] );
      T d2t_dx2        = -std::sin( x[i] ) + idx * std::cos( x[i] );
      H.insert( i, i ) = 2.0 * ( dt_dx * dt_dx + t * d2t_dx2 );
    }

    H.makeCompressed();
    return H;
  }
};
