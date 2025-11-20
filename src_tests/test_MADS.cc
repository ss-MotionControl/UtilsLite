/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  MADS test suite per problemi di ottimizzazione senza gradienti          |
 |                                                                          |
 |  Adattamento aggiornato con verbose e salvataggio x_final                |
\*--------------------------------------------------------------------------*/

#include "Utils_MADS.hh"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <map>

using Utils::MADS_minimizer;
using Scalar = double;
using Vector = MADS_minimizer<double>::Vector;
using std::string;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  string test_name;
  double initial_f;
  double final_f;
  double error;
  size_t iterations;
  size_t f_eval_count;
  size_t dimension;
  bool converged;
  string message;
  Vector final_solution;  // Aggiunto per salvare la soluzione finale
};

// Statistiche line search
struct LineSearchStats {
  std::string name;
  size_t total_tests{0};
  size_t successful_tests{0};
  size_t total_iterations{0};
  size_t total_function_evals{0};
};

// Collettore globale
std::vector<TestResult> global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

// -------------------------------------------------------------------
// Aggiorna statistiche
// -------------------------------------------------------------------
void update_line_search_statistics(const TestResult& result) {
  auto& stats = line_search_statistics["MADS"]; // Fisso per MADS
  stats.name = "MADS";
  stats.total_tests++;
  if (result.converged) {
    stats.successful_tests++;
    stats.total_iterations += result.iterations;
  }
}

// -------------------------------------------------------------------
// Stampa statistiche
// -------------------------------------------------------------------
void print_line_search_statistics() {
  fmt::print("\n\n{:=^80}\n", " MADS STATISTICS ");
  fmt::print("{:<15} {:<8} {:<8} {:<12} {:<10}\n",
             "Optimizer", "Tests", "Success", "Success%", "AvgIter");
  fmt::print("{:-<80}\n", "");

  for (const auto& [name, stats] : line_search_statistics) {
    double success_rate = (stats.total_tests > 0) ?
                          100.0 * stats.successful_tests / stats.total_tests : 0.0;
    double avg_iterations = (stats.successful_tests > 0) ?
                            static_cast<double>(stats.total_iterations) / stats.successful_tests : 0.0;
    
    fmt::print("{:<15} {:<8} {:<8} {:<12.1f} {:<10.1f}\n", 
               stats.name, stats.total_tests, stats.successful_tests, 
               success_rate, avg_iterations);
  }
  fmt::print("{:=^80}\n", "");
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva
// -------------------------------------------------------------------
void print_summary_table() {
  fmt::print("\n\n{:=^80}\n", " SUMMARY TEST RESULTS ");
  fmt::print("{:<25} {:<8} {:<12} {:<15} {:<10}\n",
             "Problem", "Dimension", "Iterations", "final f(x)", "Status");
  fmt::print("{:-<80}\n", "");

  for (auto const & result : global_test_results) {
    std::string status_str = result.converged ? "CONVERGED" : "MAX_ITER";
    
    fmt::print("{:<25} {:<8} {:<12} {:<15.6e} {:<10}\n",
               result.test_name,
               result.dimension,
               result.iterations,
               result.final_f,
               status_str);
  }

  fmt::print("{:=^80}\n", "");
}

// -------------------------------------------------------------------
// 1. Rosenbrock 2D (Problema a dimensione fissa)
// -------------------------------------------------------------------

template <typename T>
class Rosenbrock2D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(2, -std::numeric_limits<T>::infinity()); }
  Vector upper() const { return Vector::Constant(2,  std::numeric_limits<T>::infinity()); }
  Vector init() const { Vector x0(2); x0 << -1.2, 1.0; return x0; }
  T operator()(Vector const& x) const {
    T a = 1.0, b = 100.0, t1 = a - x[0], t2 = x[1] - x[0] * x[0];
    return t1 * t1 + b * t2 * t2;
  }
};

// -------------------------------------------------------------------
// ---------------- Nesterov-Chebyshev-Rosenbrock --------------------
// -------------------------------------------------------------------

template <typename T, size_t N>
class NesterovChebyshevRosenbrock {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  NesterovChebyshevRosenbrock() {
    static_assert(N >= 2, "NesterovChebyshevRosenbrock requires N >= 2");
  }

  Vector lower() const { return Vector::Constant(N, -10.0); }
  Vector upper() const { return Vector::Constant(N, 10.0); }
  
  Vector init() const {
    Vector x0 = Vector::Constant(N, 0.0);
    x0[0] = -1;
    for (Eigen::Index i = 1; i < N; ++i) x0[i] = -1.0;
    return x0;
  }

  T operator()(Vector const & x) const {
    T f = 0.25 * std::abs(x[0] - 1.0);
    for (Eigen::Index i = 0; i < N-1; ++i) {
      T term = x[i+1] - 2.0 * std::abs(x[i]) + 1.0;
      f += std::abs(term);
    }
    return f;
  }
};

// -------------------------------------------------------------------
// 2. Rosenbrock N-Dimensionale (Scalabile)
// -------------------------------------------------------------------

template <typename T, size_t N>
class RosenbrockN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  RosenbrockN() {
    static_assert(N >= 2, "RosenbrockN requires N >= 2");
  }

  Vector lower() const { return Vector::Constant(N, -10.0); }
  Vector upper() const { return Vector::Constant(N, 10.0); }
  
  Vector init() const {
    Vector x0 = Vector::Constant(N, -1.0);
    x0[N-1] = 1.0;
    return x0;
  }

  T operator()(Vector const & x) const {
    T f = 0;
    for (Eigen::Index i = 0; i < N-1; ++i) {
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

template <typename T, size_t N>
class PowellSingularN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  PowellSingularN() {
    static_assert(N % 4 == 0, "PowellSingularN requires N to be a multiple of 4");
  }

  Vector lower() const { return Vector::Constant(N, -4.0); }
  Vector upper() const { return Vector::Constant(N, 4.0); }
  
  Vector init() const {
    Vector x0(N);
    for (size_t i = 0; i < N/4; ++i) {
      x0[4*i] = 3.0;
      x0[4*i + 1] = -1.0;
      x0[4*i + 2] = 0.0;
      x0[4*i + 3] = 1.0;
    }
    return x0;
  }

  T operator()(Vector const & x) const {
    T f = 0;
    for (size_t i = 0; i < N/4; ++i) {
      size_t k1 = 4*i, k2 = 4*i + 1, k3 = 4*i + 2, k4 = 4*i + 3;
      T t1 = x[k1] + 10.0 * x[k2];
      T t2 = x[k3] - x[k4];
      T t3 = x[k2] - 2.0 * x[k3];
      T t4 = x[k1] - x[k4];
      f += t1 * t1 + 5.0 * t2 * t2 + t3 * t3 * t3 * t3 + 10.0 * t4 * t4 * t4 * t4;
    }
    return f;
  }
};

// -------------------------------------------------------------------
// 4. Extended Wood (Scalabile)
// -------------------------------------------------------------------

template <typename T, size_t N>
class ExtendedWoodN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  ExtendedWoodN() {
    static_assert(N % 4 == 0, "ExtendedWoodN requires N to be a multiple of 4");
  }

  Vector lower() const { return Vector::Constant(N, -3.0); }
  Vector upper() const { return Vector::Constant(N, 3.0); }
  
  Vector init() const {
    Vector x0(N);
    for (Eigen::Index i = 0; i < N; ++i) {
      x0[i] = (i % 2 == 0) ? -3.0 : -1.0;
    }
    return x0;
  }

  T operator()(Vector const & x) const {
    T f = 0;
    for (size_t i = 0; i < N/4; ++i) {
      size_t k1 = 4*i, k2 = 4*i + 1, k3 = 4*i + 2, k4 = 4*i + 3;
      T t1 = x[k1] * x[k1] - x[k2];
      T t2 = x[k1] - 1.0;
      T t3 = x[k3] * x[k3] - x[k4];
      T t4 = x[k3] - 1.0;
      T t5 = x[k2] + x[k4] - 2.0;
      T t6 = x[k2] - x[k4];
      f += 100.0 * t1 * t1 + t2 * t2 + 90.0 * t3 * t3 + t4 * t4 + 10.0 * t5 * t5 + 0.1 * t6 * t6;
    }
    return f;
  }
};

// -------------------- Beale (2D) --------------------
template <typename T>
class Beale2D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(2, -4.5); }
  Vector upper() const { return Vector::Constant(2, 4.5); }
  Vector init() const { Vector x0(2); x0 << 1.0, 1.0; return x0; }
  
  T operator()(Vector const & x) const {
    T x1 = x[0], x2 = x[1];
    T t1 = 1.5 - x1 * (1.0 - x2);
    T t2 = 2.25 - x1 * (1.0 - x2*x2);
    T t3 = 2.625 - x1 * (1.0 - x2*x2*x2);
    return t1 * t1 + t2 * t2 + t3 * t3;
  }
};

// -------------------- Himmelblau (2D) --------------------
template <typename T>
class Himmelblau2D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(2, -6.0); }
  Vector upper() const { return Vector::Constant(2, 6.0); }
  Vector init() const { Vector x0(2); x0 << -3.0, -3.0; return x0; }
  
  T operator()(Vector const & x) const {
    T x1 = x[0], x2 = x[1];
    T f1 = x1*x1 + x2 - 11.0;
    T f2 = x1 + x2*x2 - 7.0;
    return f1*f1 + f2*f2;
  }
};

// -------------------- Freudenstein-Roth (2D) --------------------
template <typename T>
class FreudensteinRoth2D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(2, -10.0); }
  Vector upper() const { return Vector::Constant(2, 10.0); }
  Vector init() const { Vector x0(2); x0 << 0.5, -2.0; return x0; }
  
  T operator()(Vector const & x) const {
    T x1 = x[0], x2 = x[1];
    T f1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2;
    T f2 = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2;
    return f1*f1 + f2*f2;
  }
};

// -------------------- Helical Valley (3D) --------------------
template <typename T>
class HelicalValley3D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(3, -10.0); }
  Vector upper() const { return Vector::Constant(3, 10.0); }
  Vector init() const { Vector x0(3); x0 << -1.0, 0.0, 0.0; return x0; }
  
  T operator()(Vector const & x) const {
    T x1 = x[0], x2 = x[1], x3 = x[2];
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
    return f1*f1 + f2*f2 + f3*f3;
  }
};

// -------------------- Powell Badly Scaled (2D) --------------------
template <typename T>
class PowellBadlyScaled2D {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  Vector lower() const { return Vector::Constant(2, -10.0); }
  Vector upper() const { return Vector::Constant(2, 10.0); }
  Vector init() const { Vector x0(2); x0 << 0.0, 1.0; return x0; }
  
  T operator()(Vector const & x) const {
    T x1 = x[0], x2 = x[1];
    T f1 = 1e4 * x1 * x2 - 1.0;
    T f2 = std::exp(-x1) + std::exp(-x2) - 1.0001;
    return f1*f1 + f2*f2;
  }
};

// -------------------- Brown Almost Linear (n=10) --------------------
template <typename T, size_t N>
class BrownAlmostLinearN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  BrownAlmostLinearN() {
    static_assert(N >= 2, "BrownAlmostLinearN requires N>=2");
  }

  Vector lower() const { return Vector::Constant(N, -5.0); }
  Vector upper() const { return Vector::Constant(N, 5.0); }
  Vector init() const { return Vector::Constant(N, 0.5); }
  
  T operator()(Vector const & x) const {
    T f = 0.0;
    for (Eigen::Index i = 0; i < N-1; ++i) {
      T t = x[i] + x[i+1]*x[i+1]*x[i+1] - 3.0;
      f += t*t;
    }
    for (Eigen::Index i = 0; i < N; ++i) f += 1e-3 * x[i]*x[i];
    return f;
  }
};

// -------------------- Broyden Tridiagonal (n-dim) --------------------
template <typename T, size_t N>
class BroydenTridiagonalN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  
  BroydenTridiagonalN() {
    static_assert(N >= 2, "BroydenTridiagonalN requires N>=2");
  }

  Vector lower() const { return Vector::Constant(N, -10.0); }
  Vector upper() const { return Vector::Constant(N, 10.0); }
  Vector init() const { return Vector::Constant(N, 0.5); }
  
  T operator()(Vector const & x) const {
    T f = 0.0;
    for (Eigen::Index i = 0; i < N; ++i) {
      T xim1 = (i == 0) ? 0.0 : x[i-1];
      T t = (3.0 - 2.0*x[i]) * x[i] - 2.0 * xim1 + 1.0;
      f += t*t;
    }
    return f;
  }
};

// -------------------- Ill-conditioned Quadratic (n-dim) --------------------
template <typename T, size_t N>
class IllConditionedQuadraticN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Vector lower() const { return Vector::Constant(N, -10.0); }
  Vector upper() const { return Vector::Constant(N, 10.0); }
  
  Vector init() const {
    Vector x0(N);
    for (Eigen::Index i = 0; i < N; ++i) x0[i] = (i%2 == 0) ? 1.0 : -1.0;
    return x0;
  }

  T operator()(Vector const & x) const {
    T f = 0;
    for (Eigen::Index i = 0; i < N; ++i) {
      T lambda = std::pow(1e6, T(i) / T(N-1));
      f += lambda * x[i] * x[i];
    }
    return f;
  }
};

// -------------------- Trigonometric Sum (n-dim) --------------------
template <typename T, size_t N>
class TrigonometricSumN {
public:
  using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  Vector lower() const { return Vector::Constant(N, -M_PI); }
  Vector upper() const { return Vector::Constant(N, M_PI); }
  Vector init() const { return Vector::Constant(N, 0.5); }
  
  T operator()(Vector const & x) const {
    T f = 0;
    for (Eigen::Index i = 0; i < N; ++i) {
      T idx = T(i+1);
      T t = std::sin(x[i]) + idx * (1.0 - std::cos(x[i]));
      f += t*t;
    }
    return f;
  }
};

// -------------------------------------------------------------------
// Funzione di test MADS verbosa
// -------------------------------------------------------------------
template <typename Problem>
void test(Problem & prob, std::string const & problem_name) {
  
  fmt::print("\n\nSTART: {}\n", problem_name);

  // Parametri MADS
  Utils::MADS_minimizer<Scalar>::Options opts;
  opts.max_iter = 500;
  opts.verbose = true;
  opts.print_every = 50; // Stampa ogni 50 iterazioni per meno output

  Utils::MADS_minimizer<Scalar> optimizer(opts);
  optimizer.set_bounds(prob.lower(), prob.upper());

  Vector x0 = prob.init();
  Scalar initial_f = prob(x0); // Valore iniziale

  auto result_data = optimizer.minimize(x0, prob);

  TestResult result;
  result.test_name = problem_name;
  result.initial_f = initial_f;
  result.final_f = result_data.final_f;
  result.error = 0.0; // Non abbiamo il vero minimo
  result.iterations = result_data.iterations;
  result.f_eval_count = result_data.f_eval_count;
  result.dimension = x0.size();
  result.converged = result_data.converged;
  result.message = result_data.message;
  result.final_solution = result_data.final_x;

  global_test_results.push_back(result);
  update_line_search_statistics(result);

  fmt::print(
    "{}: initial f = {:.6e}, final f = {:.6e}, iterations = {}, evaluations = {}\n",
    problem_name, initial_f, result_data.final_f,
    result_data.iterations, result_data.f_eval_count
  );
  
  if (x0.size() <= 10) { // Stampa soluzione solo per dimensioni piccole
    fmt::print("Final solution: {}\n", result_data.final_x.transpose());
  }
  
  fmt::print("Status: {}\n\n", result_data.message);
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main() {
  fmt::print("Esecuzione test MADS_minimizer...\n");

  // Test originali
  Rosenbrock2D<Scalar> rosen;
  test(rosen, "Rosenbrock2D");

  NesterovChebyshevRosenbrock<Scalar, 8> nesterov; // Ridotto per test più veloce
  test(nesterov, "NesterovChebyshevRosenbrock");

  RosenbrockN<Scalar, 6> rosenN; // Ridotto per test più veloce
  test(rosenN, "Rosenbrock6D");

  PowellSingularN<Scalar, 8> powellN; // Ridotto per test più veloce
  test(powellN, "PowellSingular8D");

  ExtendedWoodN<Scalar, 8> woodN; // Ridotto per test più veloce
  test(woodN, "ExtendedWood8D");

  // Nuovi problemi
  Beale2D<Scalar> beale;
  test(beale, "Beale2D");

  Himmelblau2D<Scalar> himm;
  test(himm, "Himmelblau2D");

  FreudensteinRoth2D<Scalar> fr;
  test(fr, "FreudensteinRoth2D");

  HelicalValley3D<Scalar> heli;
  test(heli, "HelicalValley3D");

  PowellBadlyScaled2D<Scalar> pbs;
  test(pbs, "PowellBadlyScaled2D");

  BrownAlmostLinearN<Scalar, 6> brown; // Ridotto per test più veloce
  test(brown, "BrownAlmostLinear6D");

  BroydenTridiagonalN<Scalar, 6> broy; // Ridotto per test più veloce
  test(broy, "BroydenTridiagonal6D");

  IllConditionedQuadraticN<Scalar, 8> illq; // Ridotto per test più veloce
  test(illq, "IllConditionedQuadratic8D");

  TrigonometricSumN<Scalar, 6> trig; // Ridotto per test più veloce
  test(trig, "TrigonometricSum6D");

  print_summary_table();
  print_line_search_statistics();

  return 0;
}
