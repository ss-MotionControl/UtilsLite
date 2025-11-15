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

using Scalar = double;

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
    T a = 1.0;
    T b = 100.0;
    T t1 = a - x[0];
    T t2 = x[1] - x[0] * x[0];
    T f = t1 * t1 + b * t2 * t2;

    if (grad) {
      grad->resize(2);
      (*grad)[0] = -2.0 * t1 - 4.0 * b * x[0] * t2;
      (*grad)[1] = 2.0 * b * t2;
    }
    return f;
  }

  // Minimo non vincolato: f(1, 1) = 0.0
  Vector lower() const override { return Vector::Constant(2, -std::numeric_limits<T>::infinity()); }
  Vector upper() const override { return Vector::Constant(2, std::numeric_limits<T>::infinity()); }

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
  Vector upper() const override { return Vector::Constant(N, 10.0); }

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

template <typename T, size_t N>
static
void
test( OptimizationProblem<T,N> const * tp ){

  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  Utils::LBFGS_minimizer<Scalar>::Options opts;
  opts.max_iter       = 100;
  opts.m              = 20;
  opts.verbose        = true;
  opts.use_projection = true;

  Utils::WolfeLineSearch<Scalar>       wolfe;
  Utils::StrongWolfeLineSearch<Scalar> wolfe_strong;
  Utils::ArmijoLineSearch<Scalar>      armijo;
  Utils::GoldsteinLineSearch<Scalar>   gold;
  Utils::HagerZhangLineSearch<Scalar>  HZ;
  Utils::MoreThuenteLineSearch<Scalar> More;

  Utils::LBFGS_minimizer<Scalar> minimizer(opts);

  Vector x0 = tp->init();

  auto cb = [&tp]( Vector const & x, Vector * g ) -> Scalar { return (*tp)( x, g ); };
  minimizer.set_bounds( tp->lower(), tp->upper() );
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, armijo );
    fmt::print( "ARMIJO status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, wolfe );
    fmt::print( "WOLFE status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, wolfe_strong );
    fmt::print( "WOLFE STRONG status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, gold );
    fmt::print( "GOLD status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, HZ );
    fmt::print( "HZ status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
  {
    auto [status, xf, ff] = minimizer.minimize( x0, cb, More );
    fmt::print( "More-Thuente status={}\nf={:.4}\nx0={}\nx={}\n\n\n", status, ff, x0.transpose(), xf.transpose() );
  }
}

int
main(){
  
  fmt::print( "\n\n\nRosenbrock2D\n\n\n" );
  Rosenbrock2D<Scalar> rosen;
  test( &rosen );

  fmt::print( "\n\n\nRosenbrockN\n\n\n" );
  RosenbrockN<Scalar,10> rosenN;
  test( &rosenN );

  fmt::print( "\n\n\nPowellSingularN\n\n\n" );
  PowellSingularN<Scalar,16> powerllN;
  test( &powerllN );

  fmt::print( "\n\n\nExtendedWoodN\n\n\n" );
  ExtendedWoodN<Scalar,16> woodN;
  test( &woodN );

  return 0;
}
