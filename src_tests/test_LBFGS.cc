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

#include "Utils_LBFGS.hh"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <map>

using Scalar = double;
using Vector = Utils::LBFGS_minimizer<Scalar>::Vector;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  std::string                                   problem_name;
  std::string                                   linesearch_name;
  Utils::LBFGS_minimizer<Scalar>::IterationData iteration_data;
  Utils::LBFGS_minimizer<Scalar>::Status        status;
  Scalar                                        final_value;
  Vector                                        final_solution;
  size_t                                        dimension;
};

// Collettore globale dei risultati
std::vector<TestResult> global_test_results;

/**
 * Classe base astratta per i problemi di ottimizzazione.
 * Definisce l'interfaccia comune richiesta dal minimizzatore.
 */
template <typename T, size_t N>
class OptimizationProblem {
protected:
  using Vector = Utils::LBFGS_minimizer<Scalar>::Vector;
public:
  OptimizationProblem() {}
  virtual ~OptimizationProblem() = default;

  // Metodo per la valutazione della funzione e del gradiente (Callback)
  virtual T operator()( Vector const & x, Vector * grad ) const = 0;

  // Metodi per i vincoli Box
  virtual Vector lower() const = 0;
  virtual Vector upper() const = 0;

  // Metodo per il valore iniziale
  virtual Vector init() const = 0;

  size_t size() const { return N; }
};

// -------------------------------------------------------------------
// 1. Rosenbrock 2D (Problema a dimensione fissa)
// -------------------------------------------------------------------

template <typename T>
class Rosenbrock2D : public OptimizationProblem<T,2> {
  using Base   = OptimizationProblem<T,2>;
  using Vector = typename Base::Vector;
public:
  Rosenbrock2D() : Base() {}

  T operator()( Vector const& x, Vector * grad ) const override {
    T a  = 1.0;
    T b  = 100.0;
    T t1 = a - x[0];
    T t2 = x[1] - x[0] * x[0];
    T f  = t1 * t1 + b * t2 * t2;

    if (grad) {
      grad->resize(2);
      (*grad)[0] = -2.0 * t1 - 4.0 * b * x[0] * t2;
      (*grad)[1] = 2.0 * b * t2;
    }
    return f;
  }

  // Minimo non vincolato: f(1, 1) = 0.0
  Vector lower() const override { return Vector::Constant(2, -std::numeric_limits<T>::infinity()); }
  Vector upper() const override { return Vector::Constant(2,  std::numeric_limits<T>::infinity()); }

  // Valore iniziale tipico e difficile
  Vector init() const override {
    Vector x0(2);
    x0 << -1.2, 1.0;
    return x0;
  }
};

// -------------------------------------------------------------------
// 2. Rosenbrock N-Dimensionale (Scalabile)
// -------------------------------------------------------------------

template <typename T, size_t N>
class RosenbrockN : public OptimizationProblem<T,N> {
  using Base   = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;

public:
  // Costruttore che imposta la dimensione N
  RosenbrockN() : Base() {
    if constexpr ( N < 2 ) throw std::invalid_argument("RosenbrockN requires N >= 2");
  }

  T operator()( Vector const & x, Vector * grad ) const override {
    T f{0};

    for ( size_t i{0}; i < N-1; ++i) {
      T t1 = 1.0 - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      f += t1 * t1 + 100.0 * t2 * t2;
    }

    if (grad) {
      grad->resize(N);
      grad->setZero();

      for (size_t i{0}; i < N-1; ++i) {
        T t1 = 1.0 - x[i];
        T t2 = x[i + 1] - x[i] * x[i];

        (*grad)[i]   += -2.0 * t1 - 400.0 * x[i] * t2;
        (*grad)[i+1] += 200.0 * t2;
      }
    }
    return f;
  }

  // Minimo non vincolato: f(1, 1, ..., 1) = 0.0
  Vector lower() const override { return Vector::Constant(N, -10.0); }
  Vector upper() const override { return Vector::Constant(N,  10.0); }

  // Valore iniziale comune per N-dim
  Vector init() const override {
    Vector x0 = Vector::Constant(N,-1.0);
    x0[N-1] = 1.0; // Variazione per la Rosenbrock
    return x0;
  }
};

// -------------------------------------------------------------------
// 3. Extended Powell Singular (Scalabile)
// -------------------------------------------------------------------

template <typename T, size_t N>
class PowellSingularN : public OptimizationProblem<T,N> {
  using Base   = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;

public:
  PowellSingularN() : Base() {
    if constexpr (N % 4 != 0) throw std::invalid_argument("PowellSingularN requires N to be a multiple of 4");
  }

  T operator()( Vector const & x, Vector * grad ) const override {
    T f{0};

    if (grad) {
      grad->resize(N);
      grad->setZero();
    }

    for ( size_t i{0}; i < N / 4; ++i ) {
      size_t k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];

      f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;

      if (grad) {
        (*grad)[k1] += 2.0 * t1 + 40.0 * t4 * t4 * t4;
        (*grad)[k2] += 20.0 * t1 + 4.0 * t3 * t3 * t3;
        (*grad)[k3] += 10.0 * t2 - 8.0 * t3 * t3 * t3;
        (*grad)[k4] += -10.0 * t2 - 40.0 * t4 * t4 * t4;
      }
    }
    return f;
  }

  // Vincoli box consigliati
  Vector lower() const override { return Vector::Constant(N, -4.0); }
  Vector upper() const override { return Vector::Constant(N,  4.0); }

  // Valore iniziale tipico per Powell Singular
  Vector init() const override {
    Vector x0(N);
    for ( size_t i{0}; i < N/4; ++i) {
      x0[4 * i    ] = 3.0;
      x0[4 * i + 1] = -1.0;
      x0[4 * i + 2] = 0.0;
      x0[4 * i + 3] = 1.0;
    }
    return x0;
  }
};

// -------------------------------------------------------------------
// 4. Extended Wood (Scalabile)
// -------------------------------------------------------------------

template <typename T, size_t N>
class ExtendedWoodN : public OptimizationProblem<T,N> {
  using Base   = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;

public:
  ExtendedWoodN() : Base() {
    if constexpr (N % 4 != 0) throw std::invalid_argument("ExtendedWoodN requires N to be a multiple of 4");
  }

  T operator()( Vector const & x, Vector * grad ) const override {
    T f{0};

    if (grad) {
      grad->resize(N);
      grad->setZero();
    }

    for ( size_t i{0}; i < N / 4; ++i) {
      size_t k1 = 4 * i, k2 = 4 * i + 1, k3 = 4 * i + 2, k4 = 4 * i + 3;

      T t1 = x[k1] * x[k1] - x[k2];
      T t2 = x[k1] - 1.0;
      T t3 = x[k3] * x[k3] - x[k4];
      T t4 = x[k3] - 1.0;
      T t5 = x[k2] + x[k4] - 2.0;
      T t6 = x[k2] - x[k4];

      f += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 + 10.0 * t5 * t5 + 0.1 * t6 * t6;

      if (grad) {
        (*grad)[k1] += 400.0 * x[k1] * t1 + 2.0 * t2;
        (*grad)[k2] += -200.0 * t1 + 20.0 * t5 + 0.2 * t6;
        (*grad)[k3] += 360.0 * x[k3] * t3 + 2.0 * t4;
        (*grad)[k4] += -180.0 * t3 + 20.0 * t5 - 0.2 * t6;
      }
    }
    return f;
  }

  // Vincoli box consigliati
  Vector lower() const override { return Vector::Constant(N, -3.0); }
  Vector upper() const override { return Vector::Constant(N,  3.0); }

  // Valore iniziale tipico per Wood
  Vector init() const override {
    Vector x0(N);
    for ( size_t i{0}; i < N; ++i ) {
      if (i % 2 == 0) x0[i] = -3.0;
      else            x0[i] = -1.0;
    }
    return x0;
  }
};

// -------------------- Beale (2D) --------------------
template <typename T>
class Beale2D : public OptimizationProblem<T,2> {
  using Base = OptimizationProblem<T,2>;
  using Vector = typename Base::Vector;
public:
  Beale2D() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T x1 = x[0], x2 = x[1];
    T t1 = 1.5 - x1 * (1.0 - x2);
    T t2 = 2.25 - x1 * (1.0 - x2*x2);
    T t3 = 2.625 - x1 * (1.0 - x2*x2*x2);
    
    T f = t1 * t1 + t2 * t2 + t3 * t3;
    
    if (grad) {
      grad->resize(2);
      // Derivata rispetto a x1
      (*grad)[0] = -2.0 * t1 * (1.0 - x2)
                   - 2.0 * t2 * (1.0 - x2*x2)
                   - 2.0 * t3 * (1.0 - x2*x2*x2);
      
      // Derivata rispetto a x2
      (*grad)[1] = 2.0 * t1 * x1
                   + 2.0 * t2 * (2.0 * x1 * x2)
                   + 2.0 * t3 * (3.0 * x1 * x2*x2);
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(2,-4.5); }
  Vector upper() const override { return Vector::Constant(2,4.5); }
  Vector init()  const override { Vector x0(2); x0<<1.0,1.0; return x0; }
};

// -------------------- Himmelblau (2D) --------------------
template <typename T>
class Himmelblau2D : public OptimizationProblem<T,2> {
  using Base = OptimizationProblem<T,2>;
  using Vector = typename Base::Vector;
public:
  Himmelblau2D() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T x1 = x[0], x2 = x[1];
    T f1 = x1*x1 + x2 - 11.0;
    T f2 = x1 + x2*x2 - 7.0;
    T f = f1*f1 + f2*f2;
    
    if (grad) {
      grad->resize(2);
      (*grad)[0] = 4.0 * x1 * f1 + 2.0 * f2;
      (*grad)[1] = 2.0 * f1 + 4.0 * x2 * f2;
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(2,-6.0); }
  Vector upper() const override { return Vector::Constant(2,6.0); }
  Vector init() const override { Vector x0(2); x0 << -3.0, -3.0; return x0; }
};

// -------------------- Freudenstein-Roth (2D) --------------------
template <typename T>
class FreudensteinRoth2D : public OptimizationProblem<T,2> {
  using Base = OptimizationProblem<T,2>;
  using Vector = typename Base::Vector;
public:
  FreudensteinRoth2D() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T x1 = x[0], x2 = x[1];
    T f1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2;
    T f2 = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2;
    T f = f1*f1 + f2*f2;
    
    if (grad) {
      grad->resize(2);
      // Derivata rispetto a x1
      (*grad)[0] = 2.0 * f1 + 2.0 * f2;
      
      // Derivata rispetto a x2
      T df1_dx2 = (5.0 - 2.0*x2) * x2 + (5.0 - x2) * x2 - 2.0; // Derivata di ((5-x2)*x2 - 2)*x2
      T df2_dx2 = ((x2 + 1.0) + x2) * x2 + ((x2 + 1.0) * x2 - 14.0); // Derivata di ((x2+1)*x2 - 14)*x2
      (*grad)[1] = 2.0 * f1 * df1_dx2 + 2.0 * f2 * df2_dx2;
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(2,-10.0); }
  Vector upper() const override { return Vector::Constant(2,10.0); }
  Vector init() const override { Vector x0(2); x0 << 0.5, -2.0; return x0; }
};

// -------------------- Helical Valley (3D) --------------------
template <typename T>
class HelicalValley3D : public OptimizationProblem<T,3> {
  using Base = OptimizationProblem<T,3>;
  using Vector = typename Base::Vector;
public:
  HelicalValley3D() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T x1=x[0], x2=x[1], x3=x[2];
    T theta = 0.0;
    if (x1 > 0) {
      theta = std::atan(x2/x1) / (2.0*M_PI);
    } else if (x1 < 0) {
      theta = std::atan(x2/x1) / (2.0*M_PI) + 0.5;
    } else {
      theta = (x2 >= 0) ? 0.25 : -0.25;
    }
    
    T f1 = 10.0 * (x3 - 10.0 * theta);
    T f2 = 10.0 * (std::sqrt(x1*x1 + x2*x2) - 1.0);
    T f3 = x3;
    T f = f1*f1 + f2*f2 + f3*f3;
    
    if (grad) {
      grad->resize(3);
      T r2 = x1*x1 + x2*x2;
      T r = std::sqrt(r2);
      
      // Derivata di theta
      T dtheta_dx1, dtheta_dx2;
      if (x1 != 0) {
        dtheta_dx1 = -x2 / (2.0*M_PI * r2);
        dtheta_dx2 = x1 / (2.0*M_PI * r2);
      } else {
        dtheta_dx1 = 0.0;
        dtheta_dx2 = 0.0;
      }
      
      // Gradiente
      (*grad)[0] = 2.0 * f1 * (-100.0 * dtheta_dx1) +
                   2.0 * f2 * (10.0 * x1 / r);
      (*grad)[1] = 2.0 * f1 * (-100.0 * dtheta_dx2) +
                   2.0 * f2 * (10.0 * x2 / r);
      (*grad)[2] = 2.0 * f1 * 10.0 + 2.0 * f3;
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(3,-10.0); }
  Vector upper() const override { return Vector::Constant(3,10.0); }
  Vector init() const override { Vector x0(3); x0 << -1.0, 0.0, 0.0; return x0; }
};

// -------------------- Powell Badly Scaled (2D) --------------------
template <typename T>
class PowellBadlyScaled2D : public OptimizationProblem<T,2> {
  using Base = OptimizationProblem<T,2>;
  using Vector = typename Base::Vector;
public:
  PowellBadlyScaled2D() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp(-x1) + std::exp(-x2) - 1.0001;
    T f = f1*f1 + f2*f2;
    
    if (grad) {
      grad->resize(2);
      (*grad)[0] = 2.0 * f1 * (1e4 * x2) + 2.0 * f2 * (-std::exp(-x1));
      (*grad)[1] = 2.0 * f1 * (1e4 * x1) + 2.0 * f2 * (-std::exp(-x2));
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(2,-10.0); }
  Vector upper() const override { return Vector::Constant(2,10.0); }
  Vector init() const override { Vector x0(2); x0 << 0.0, 1.0; return x0; }
};

// -------------------- Brown Almost Linear (n=10) --------------------
template <typename T, size_t N>
class BrownAlmostLinearN : public OptimizationProblem<T,N> {
  using Base = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;
public:
  BrownAlmostLinearN() : Base() {
    static_assert(N >= 2, "BrownAlmostLinearN requires N>=2");
  }
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T f = 0.0;
    for (size_t i=0; i<N-1; ++i) {
      T t = x[i] + x[i+1]*x[i+1]*x[i+1] - 3.0; // x_{i+1}^3
      f += t*t;
    }
    // Piccolo regolarizzatore quadratico
    for (size_t i=0; i<N; ++i) f += 1e-3 * x[i]*x[i];
    
    if (grad) {
      grad->resize(N);
      grad->setZero();
      
      for (size_t i=0; i<N-1; ++i) {
        T t = x[i] + x[i+1]*x[i+1]*x[i+1] - 3.0;
        
        (*grad)[i]   += 2.0 * t;
        (*grad)[i+1] += 2.0 * t * (3.0 * x[i+1]*x[i+1]);
      }
      
      // Derivata del regolarizzatore
      for (size_t i=0; i<N; ++i) {
        (*grad)[i] += 2e-3 * x[i];
      }
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(N,-5.0); }
  Vector upper() const override { return Vector::Constant(N,5.0); }
  Vector init()  const override { return Vector::Constant(N, 0.5); }
};

// -------------------- Broyden Tridiagonal (n-dim) --------------------
template <typename T, size_t N>
class BroydenTridiagonalN : public OptimizationProblem<T,N> {
  using Base = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;
public:
  BroydenTridiagonalN() : Base() { static_assert(N>=2,"BroydenTridiagonalN requires N>=2"); }
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T f = 0.0;
    for (size_t i=0; i<N; ++i) {
      T xim1 = (i==0) ? 0.0 : x[i-1];
      T t = (3.0 - 2.0*x[i]) * x[i] - 2.0 * xim1 + 1.0;
      f += t*t;
    }
    
    if (grad) {
      grad->resize(N);
      grad->setZero();
      
      for (size_t i=0; i<N; ++i) {
        T xim1 = (i==0) ? 0.0 : x[i-1];
        T xip1 = (i==N-1) ? 0.0 : x[i+1];
        T t = (3.0 - 2.0*x[i]) * x[i] - 2.0 * xim1 + 1.0;
        
        // Derivata rispetto a x[i]
        T dt_dxi = (3.0 - 4.0*x[i]);
        (*grad)[i] += 2.0 * t * dt_dxi;
        
        // Contributo dal termine -2*x[i] quando appare come x_{i-1} nel termine successivo
        if (i < N-1) {
          T t_next = (3.0 - 2.0*x[i+1]) * x[i+1] - 2.0 * x[i] + 1.0;
          (*grad)[i] += 2.0 * t_next * (-2.0);
        }
      }
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(N,-10.0); }
  Vector upper() const override { return Vector::Constant(N,10.0); }
  Vector init()  const override { return Vector::Constant(N, 0.5); }
};

// -------------------- Ill-conditioned Quadratic (n-dim) --------------------
template <typename T, size_t N>
class IllConditionedQuadraticN : public OptimizationProblem<T,N> {
  using Base = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;
public:
  IllConditionedQuadraticN() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T f = 0;
    if (grad) grad->resize(N);
    
    for (size_t i=0; i<N; ++i) {
      T lambda = std::pow(1e6, T(i) / T(N-1)); // da 1 a 1e6
      f += lambda * x[i] * x[i];
      
      if (grad) (*grad)[i] = 2.0 * lambda * x[i];
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(N,-10.0); }
  Vector upper() const override { return Vector::Constant(N,10.0); }
  Vector init() const override {
    Vector x0(N);
    for (size_t i=0; i<N; ++i) x0[i] = (i%2==0 ? 1.0 : -1.0);
    return x0;
  }
};


template <typename T, size_t N>
class IllConditionedQuadRot : public OptimizationProblem<T,N> {
  using Base = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;
  using Matrix = Eigen::Matrix<T,N,N>;
    
  Matrix Q_;
    
public:
  IllConditionedQuadRot() : Base() {
    // Step 1: Construct diagonal eigenvalues 1..1e6
    Eigen::Matrix<T,N,1> lambdas;
    for(size_t i=0; i<N; ++i)
      lambdas(i) = std::pow(1e6, T(i)/T(N-1));
        
    // Step 2: Random orthogonal matrix via QR decomposition
    Eigen::Matrix<T,N,N> A;
    std::mt19937 rng(42);
    std::normal_distribution<T> dist(0.0,1.0);
    for(size_t i=0;i<N;++i)
      for(size_t j=0;j<N;++j)
        A(i,j) = dist(rng);
    Eigen::HouseholderQR<Matrix> qr(A);
    Matrix U = qr.householderQ();
        
    // Step 3: Construct Q = U^T * Lambda * U
    Q_ = U.transpose() * lambdas.asDiagonal() * U;
  }
    
  T operator()(Vector const& x, Vector* grad) const override {
    if(grad) grad->resize(N);
    T f = x.dot(Q_ * x);
    if(grad) *grad = 2.0 * Q_ * x;
    return f;
  }
    
  Vector lower() const override { return Vector::Constant(N,-10.0); }
  Vector upper() const override { return Vector::Constant(N,10.0); }
    
  Vector init() const override {
    Vector x0(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<T> dist(-1.0,1.0);
    for(size_t i=0;i<N;++i) x0[i] = dist(rng);
    return x0;
  }
};

// -------------------- Trigonometric Sum (n-dim) --------------------
template <typename T, size_t N>
class TrigonometricSumN : public OptimizationProblem<T,N> {
  using Base = OptimizationProblem<T,N>;
  using Vector = typename Base::Vector;
public:
  TrigonometricSumN() : Base() {}
  
  T operator()( Vector const & x, Vector * grad ) const override {
    T f = 0;
    if (grad) grad->resize(N);
    
    for (size_t i=0; i<N; ++i) {
      T idx = T(i+1);
      T t = std::sin(x[i]) + idx * (1.0 - std::cos(x[i]));
      f += t*t;
      
      if (grad) (*grad)[i] = 2.0 * t * (std::cos(x[i]) + idx * std::sin(x[i]));
    }
    return f;
  }
  
  Vector lower() const override { return Vector::Constant(N,-M_PI); }
  Vector upper() const override { return Vector::Constant(N,M_PI); }
  Vector init()  const override { return Vector::Constant(N, 0.5); }
};

// -------------------------------------------------------------------
// Funzione per stampare la tabella riassuntiva
// -------------------------------------------------------------------
void
print_summary_table() {
  fmt::print(
    "\n\n{:=^80}\n"
    "{:<28} {:<12} {:<8} {:<12} {:<15} {:<10}\n"
    "{:-<80}\n",
    " SUMMARY TEST RESULTS ",
    "Problem", "LineSearch", "Dimension", "Iterations", "final f(x)", "Status",
    ""
  );
    
  for ( auto const & result : global_test_results ) {
    std::string status_str;
    bool converged = false;
        
    switch (result.status) {
    case Utils::LBFGS_minimizer<Scalar>::Status::CONVERGED:
      status_str = "CONVERGED";
      converged = true;
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::MAX_ITERATIONS:
      status_str = "MAX_ITER";
      converged = false;
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::FAILED:
      status_str = "FAILED";
      converged = false;
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::GRADIENT_TOO_SMALL:
      status_str = "GRAD_SMALL";
      converged = true;
      break;
    default:
      status_str = "UNKNOWN";
      converged = false;
    }
        
    // Usa colori: verde per convergenza, rosso altrimenti
    auto const & GREEN { fmt::fg(fmt::color::green) };
    auto const & RED   { fmt::fg(fmt::color::red)   };
        
    fmt::print(
      "{:<28} {:<12} {:<8} {:<12} {:<15.6e} ",
      result.problem_name,
      result.linesearch_name,
      result.dimension,
      result.iteration_data.iterations,
      result.final_value
    );

    if ( converged ) fmt::print( GREEN, "{}\n", status_str );
    else             fmt::print( RED,   "{}\n", status_str );
  }

  fmt::print("{:=^80}\n", "");

  // Statistiche finali
  size_t total_tests     = global_test_results.size();
  size_t converged_tests = std::count_if(global_test_results.begin(), global_test_results.end(),
    [](const TestResult& r) {
       return r.status == Utils::LBFGS_minimizer<Scalar>::Status::CONVERGED ||
              r.status == Utils::LBFGS_minimizer<Scalar>::Status::GRADIENT_TOO_SMALL;
  });
    
  fmt::print("\nFinal Stats: {} / {} test converged ({:.1f}%)\n",
             converged_tests, total_tests, (100.0 * converged_tests / total_tests));
}

// -------------------------------------------------------------------
// test runner modificato per raccogliere risultati
// -------------------------------------------------------------------
template <typename T, size_t N>
static
void
test( OptimizationProblem<T,N> const * tp, const std::string& problem_name ){

  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  Utils::LBFGS_minimizer<Scalar>::Options opts;
  opts.max_iter       = 20000;
  opts.m              = 20;
  opts.verbose        = true; // Disabilita verbose per output più pulito
  opts.use_projection = true;

  Vector x0 = tp->init();

  auto cb = [&tp]( Vector const & x, Vector * g ) -> Scalar { return (*tp)( x, g ); };

  // Lista di line search da testare - usa std::function per gestire i diversi tipi
  std::vector<std::pair<std::string, std::function<std::optional<Scalar>(
    Scalar, Scalar, Vector const&, Vector const&, 
    std::function<Scalar(Vector const&, Vector*)>, Scalar)>>> line_searches;

  // Inizializza le line search
  Utils::WolfeLineSearch<Scalar>       wolfe;
  Utils::StrongWolfeLineSearch<Scalar> wolfe_strong;
  Utils::ArmijoLineSearch<Scalar>      armijo;
  Utils::GoldsteinLineSearch<Scalar>   gold;
  Utils::HagerZhangLineSearch<Scalar>  HZ;
  Utils::MoreThuenteLineSearch<Scalar> More;

  // Aggiungi le line search al vettore usando lambda per incapsulare l'oggetto
  line_searches.emplace_back("Armijo", [&armijo](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return armijo(f0, Df0, x, d, callback, alpha0);
  });

  line_searches.emplace_back("Wolfe", [&wolfe](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return wolfe(f0, Df0, x, d, callback, alpha0);
  });

  line_searches.emplace_back("WolfeStrong", [&wolfe_strong](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return wolfe_strong(f0, Df0, x, d, callback, alpha0);
  });

  line_searches.emplace_back("Goldstein", [&gold](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return gold(f0, Df0, x, d, callback, alpha0);
  });

  line_searches.emplace_back("HagerZhang", [&HZ](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return HZ(f0, Df0, x, d, callback, alpha0);
  });

  line_searches.emplace_back("MoreThuente", [&More](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return More(f0, Df0, x, d, callback, alpha0);
  });

  for (const auto& [ls_name, line_search] : line_searches) {
    Utils::LBFGS_minimizer<Scalar> minimizer(opts);
    minimizer.set_bounds( tp->lower(), tp->upper() );
    
    auto [status, xf, ff, iter_data] = minimizer.minimize( x0, cb, line_search );
    
    // Salva il risultato
    TestResult result;
    result.problem_name = problem_name;
    result.linesearch_name = ls_name;
    result.iteration_data = iter_data;
    result.status = status;
    result.final_value = ff;
    result.final_solution = xf;
    result.dimension = N;
    
    global_test_results.push_back(result);
    
    // Converti status in stringa
    std::string status_str;
    switch (status) {
    case Utils::LBFGS_minimizer<Scalar>::Status::CONVERGED:
      status_str = "CONVERGED";
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::MAX_ITERATIONS:
      status_str = "MAX_ITERATIONS";
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::LINE_SEARCH_FAILED:
      status_str = "LINE_SEARCH_FAILED";
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::GRADIENT_TOO_SMALL:
      status_str = "GRADIENT_TOO_SMALL";
      break;
    case Utils::LBFGS_minimizer<Scalar>::Status::FAILED:
      status_str = "FAILED";
      break;
    default:
      status_str = "UNKNOWN";
    }
      
    fmt::print("{} - {}: {} after {} iterations, f = {:.6e}\n",
                problem_name, ls_name, status_str, iter_data.iterations, ff);
  }
  fmt::print("\n");
}

// -------------------------------------------------------------------
// main: run original 4 tests plus the 10 new ones
// -------------------------------------------------------------------
int
main(){

  fmt::print( "Esecuzione test di ottimizzazione L-BFGS...\n" );

  // Test originali
  Rosenbrock2D<Scalar> rosen;
  test( &rosen, "Rosenbrock2D" );

  RosenbrockN<Scalar,10> rosenN;
  test( &rosenN, "Rosenbrock10D" );

  PowellSingularN<Scalar,16> powerllN;
  test( &powerllN, "PowellSingular16D" );

  ExtendedWoodN<Scalar,16> woodN;
  test( &woodN, "ExtendedWood16D" );

  // Nuovi problemi
  Beale2D<Scalar> beale;
  test( &beale, "Beale2D" );

  Himmelblau2D<Scalar> himm;
  test( &himm, "Himmelblau2D" );

  FreudensteinRoth2D<Scalar> fr;
  test( &fr, "FreudensteinRoth2D" );

  HelicalValley3D<Scalar> heli;
  test( &heli, "HelicalValley3D" );

  PowellBadlyScaled2D<Scalar> pbs;
  test( &pbs, "PowellBadlyScaled2D" );

  BrownAlmostLinearN<Scalar,10> brown;
  test( &brown, "BrownAlmostLinear10D" );

  BroydenTridiagonalN<Scalar,12> broy;
  test( &broy, "BroydenTridiagonal12D" );

  IllConditionedQuadraticN<Scalar,20> illq;
  test( &illq, "IllConditionedQuadratic20D" );

  IllConditionedQuadRot<Scalar,20> illq2;
  test( &illq2, "IllConditionedQuadRot20D" );

  TrigonometricSumN<Scalar,15> trig;
  test( &trig, "TrigonometricSum15D" );

  // Stampa tabella riassuntiva
  print_summary_table();

  return 0;
}
