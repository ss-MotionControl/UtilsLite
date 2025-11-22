/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  NelderMead_minimizer per problemi di ottimizzazione senza gradienti     |
 |                                                                          |
 |  Test harness aggiornato: verbose, salvataggio x_final, statistiche       |
 \*--------------------------------------------------------------------------*/

#include "Utils_NelderMead.hh"
#include "Utils_fmt.hh"
#include "ND_func.cxx" // problem definitions (assumed to provide lower(), upper(), init(), operator())

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <map>
#include <string>


using Scalar = double;
using NM     = Utils::NelderMead_Hybrid<Scalar>;
//using NM     = Utils::NelderMead_classic<Scalar>;
using Vector = NM::Vector;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  std::string problem_name;
  std::string optimizer_name;
  Scalar final_value{0};
  Vector final_solution;
  size_t dimension{0};
  size_t iterations{0};
  size_t function_evaluations{0};
  bool   converged{false};
};

// Statistiche optimizer
struct OptimizerStats {
  std::string name;
  size_t total_tests{0};
  size_t successful_tests{0};
  size_t total_iterations{0};
  size_t total_function_evals{0};
};

// Collettori globali
static std::vector<TestResult> global_test_results;
static std::map<std::string, OptimizerStats> optimizer_statistics;

// Helper: convert Eigen vector to formatted string
static std::string vec_to_string(Vector const & v) {
  std::string s;
  s.reserve(v.size() * 16);
  s.push_back('[');
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    if (i) s += ", ";
    s += fmt::format("{:.6e}", static_cast<double>(v(i)));
  }
  s.push_back(']');
  return s;
}

// -------------------------------------------------------------------
// Aggiorna statistiche
// -------------------------------------------------------------------
void update_optimizer_statistics(const TestResult& result) {
  auto & stats = optimizer_statistics[result.optimizer_name];
  stats.name = result.optimizer_name;
  stats.total_tests++;
  stats.total_iterations     += result.iterations;
  stats.total_function_evals += result.function_evaluations;
  if ( result.converged ) stats.successful_tests++;
}

// -------------------------------------------------------------------
// Stampa statistiche
// -------------------------------------------------------------------
void print_optimizer_statistics() {
  fmt::print("\n\n{:=^100}\n", " Nelder Mead STATISTICS ");
  fmt::print("{:<24} {:>8} {:>10} {:>12} {:>12}\n",
             "Optimizer", "Tests", "Success", "AvgIter", "AvgFunEvals");
  fmt::print("{:-<100}\n", "");

  for (const auto & [name, stats] : optimizer_statistics) {
    double success_rate = (stats.total_tests > 0) ?
      100.0 * static_cast<double>(stats.successful_tests) / static_cast<double>(stats.total_tests) : 0.0;
    double avg_iter = (stats.total_tests > 0) ?
      static_cast<double>(stats.total_iterations) / static_cast<double>(stats.total_tests) : 0.0;
    double avg_fevals = (stats.total_tests > 0) ?
      static_cast<double>(stats.total_function_evals) / static_cast<double>(stats.total_tests) : 0.0;

    auto color = (success_rate >= 80.0) ? fmt::fg(fmt::color::green) :
                 (success_rate >= 60.0) ? fmt::fg(fmt::color::yellow) : fmt::fg(fmt::color::red);

    fmt::print("{:<24} {:>8} ", stats.name, stats.total_tests);
    fmt::print(color, "{:>9.2f}%", success_rate);
    fmt::print(" {:>12.2f} {:>12.2f}\n", avg_iter, avg_fevals);
  }

  fmt::print("{:=^100}\n", "");
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva
// -------------------------------------------------------------------
void print_summary_table() {
  fmt::print("\n\n{:=^120}\n", " SUMMARY TEST RESULTS ");
  fmt::print("{:<28} {:<20} {:>10} {:>12} {:>18} {:>12}\n",
             "Problem", "Optimizer", "Dimension", "Iterations", "final f(x)", "Status");
  fmt::print("{:-<120}\n", "");

  for ( auto const & result : global_test_results ) {
    std::string status_str = result.converged ? "CONVERGED" : "NOT_CONV";
    auto const & GREEN { fmt::fg(fmt::color::green) };
    auto const & RED   { fmt::fg(fmt::color::red)   };

    fmt::print("{:<28} {:<20} {:>10} {:>12} {:>18.6e} ",
               result.problem_name,
               result.optimizer_name,
               result.dimension,
               result.iterations,
               static_cast<double>(result.final_value));

    if ( result.converged ) fmt::print( GREEN, "{:<12}\n", status_str );
    else                    fmt::print( RED,   "{:<12}\n", status_str );
  }

  fmt::print("{:=^120}\n", "");
}

// -------------------------------------------------------------------
// Funzione di test Nelder Mead verbosa
// -------------------------------------------------------------------
template <typename Problem>
void test(Problem & prob, std::string const & problem_name) {

  fmt::print( "\n\nSTART: {}\n", problem_name );

  #if 0
  // Parametri Nelder Mead
  Utils::NelderMead_minimizer<Scalar>::Options opts;
  opts.verbose = true;
  opts.strategy = Utils::NelderMead_minimizer<Scalar>::Strategy::RANDOM_SUBSPACE;
  opts.progress_frequency = 10; // print every iteration when verbose
      
  opts.max_dimension_standard=5;
  opts.subspace_min_size=2;
  opts.subspace_max_size=10;

  Utils::NelderMead_minimizer<Scalar> optimizer(opts);
  #else
  // Parametri Nelder Mead
  NM::Options opts;
  opts.verbose = true;
  opts.max_iterations = 10000;
  opts.max_function_evaluations = 50000;
  //opts.progress_frequency = 50; // print every iteration when verbose
  NM optimizer(opts);
  #endif

  Vector x0 = prob.init();

  // Set bounds only if sizes match
  try {
    Vector lower = prob.lower();
    Vector upper = prob.upper();
    if ( lower.size() == x0.size() && upper.size() == x0.size() ) {
      optimizer.set_bounds(lower, upper);
    }
  } catch(...) {
    // ignore if problem does not provide bounds in expected form
  }

  auto callback = [&](Vector const & x)->Scalar { return prob(x); };

  auto iter_data = optimizer.minimize(x0, callback);

  TestResult result;
  result.problem_name         = problem_name;
  result.optimizer_name       = "NelderMead";
  result.iterations           = iter_data.total_iterations;
  result.converged            = iter_data.status == NM::Status::CONVERGED;
  result.function_evaluations = iter_data.function_evaluations;
  result.final_value          = iter_data.final_function_value;
  result.final_solution       = iter_data.solution;
  result.dimension            = static_cast<size_t>(x0.size());

  global_test_results.push_back(result);
  update_optimizer_statistics(result);

  // Print concise final info
  fmt::print("\n{}: final f = {:.6e}, iterations = {}\n", problem_name,
             static_cast<double>(iter_data.final_function_value), iter_data.total_iterations);
  fmt::print("x_final = {}\n", vec_to_string(iter_data.solution));
  fmt::print("x_initial = {}\n", vec_to_string(x0));
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main() {
  fmt::print("Esecuzione test NelderMead_minimizer...\n");

  // Lista dei test: creare istanze e chiamare test(...)
  // Nota: ND_func.cxx deve definire le classi usate qui

  try {
    Rosenbrock2D<Scalar> rosen;
    test( rosen, "Rosenbrock2D" );

    RosenbrockN<Scalar,10> rosenN;
    test( rosenN, "Rosenbrock10D" );

    PowellSingularN<Scalar,16> powellN;
    test( powellN, "PowellSingular16D" );

    ExtendedWoodN<Scalar,16> woodN;
    test( woodN, "ExtendedWood16D" );

    // Altri problemi (se presenti in ND_func.cxx)
    Beale2D<Scalar> beale;
    test( beale, "Beale2D" );

    Himmelblau2D<Scalar> himm;
    test( himm, "Himmelblau2D" );

    FreudensteinRoth2D<Scalar> fr;
    test( fr, "FreudensteinRoth2D" );

    HelicalValley3D<Scalar> heli;
    test( heli, "HelicalValley3D" );

    PowellBadlyScaled2D<Scalar> pbs;
    test( pbs, "PowellBadlyScaled2D" );

    BrownAlmostLinearN<Scalar,10> brown;
    test( brown, "BrownAlmostLinear10D" );

    BroydenTridiagonalN<Scalar,12> broy;
    test( broy, "BroydenTridiagonal12D" );

    IllConditionedQuadraticN<Scalar,20> illq;
    test( illq, "IllConditionedQuadratic20D" );

    TrigonometricSumN<Scalar,15> trig;
    test( trig, "TrigonometricSum15D" );

    SchwefelN<Scalar,15> schwefel;
    test( schwefel, "SchwefelN15D" );

    AckleyN<Scalar,15> ackley;
    test( ackley, "AckleyN15D" );

    RastriginN<Scalar,15> rastrigin;
    test( rastrigin, "RastriginN15D" );

  } catch (std::exception const & e) {
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Errore durante l'esecuzione dei test: {}\n", e.what());
    return 1;
  } catch (...) {
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), "Errore non gestito durante i test.\n");
    return 2;
  }

  print_summary_table();
  print_optimizer_statistics();

  return 0;
}
