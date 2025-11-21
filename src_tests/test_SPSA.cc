/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  SPSA test suite per problemi di ottimizzazione senza gradienti          |
 |                                                                          |
 |  Adattamento aggiornato con verbose e salvataggio x_final                |
\*--------------------------------------------------------------------------*/

#include "Utils_SPSA.hh"
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <map>

using Scalar = double;
using Vector = Utils::SPSA_minimizer<Scalar>::Vector;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  std::string problem_name;
  std::string linesearch_name;
  Utils::SPSA_minimizer<Scalar>::Result iteration_data;
  Scalar final_value;
  Vector final_solution;
  size_t dimension;
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
  auto& stats = line_search_statistics[result.linesearch_name];
  stats.name = result.linesearch_name;
  stats.total_tests++;
  if (result.iteration_data.converged) {
    stats.successful_tests++;
    stats.total_iterations += result.iteration_data.iterations;
  }
}

// -------------------------------------------------------------------
// Stampa statistiche
// -------------------------------------------------------------------
void print_line_search_statistics() {
  fmt::print("\n\n{:=^80}\n", " SPSA STATISTICS ");
  fmt::print("{:<15} {:<8} {:<8} {:<12} {:<10}\n",
             "Optimizer", "Tests", "Success", "Success%", "AvgIter");
  fmt::print("{:-<80}\n", "");

  for (const auto& [name, stats] : line_search_statistics) {
    Scalar success_rate = (stats.total_tests > 0) ?
                          100.0 * stats.successful_tests / stats.total_tests : 0.0;
    Scalar avg_iterations = (stats.successful_tests > 0) ?
                            static_cast<Scalar>(stats.total_iterations) / stats.successful_tests : 0.0;
    auto color = (success_rate >= 80.0) ? fmt::fg(fmt::color::green) :
                 (success_rate >= 60.0) ? fmt::fg(fmt::color::yellow) :
                 fmt::fg(fmt::color::red);

    fmt::print("{:<15} {:<8} {:<8} ", stats.name, stats.total_tests, stats.successful_tests);
    fmt::print(color, "{:<12.1f}", success_rate);
    fmt::print(" {:<10.1f}\n", avg_iterations);
  }
  fmt::print("{:=^80}\n", "");
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva
// -------------------------------------------------------------------
void print_summary_table() {
  fmt::print("\n\n{:=^80}\n", " SUMMARY TEST RESULTS ");
  fmt::print("{:<28} {:<12} {:<8} {:<12} {:<15} {:<10}\n",
             "Problem", "Optimizer", "Dimension", "Iterations", "final f(x)", "Status");
  fmt::print("{:-<80}\n", "");

  for ( auto const & result : global_test_results ) {
    std::string status_str = result.iteration_data.converged ? "CONVERGED" : "MAX_ITER";
    auto const & GREEN { fmt::fg(fmt::color::green) };
    auto const & RED   { fmt::fg(fmt::color::red)   };

    fmt::print("{:<28} {:<12} {:<8} {:<12} {:<15.6e}",
               result.problem_name,
               result.linesearch_name,
               result.dimension,
               result.iteration_data.iterations,
               result.final_value);

    if ( result.iteration_data.converged )
      fmt::print( GREEN, "{}\n", status_str );
    else
      fmt::print( RED,   "{}\n", status_str );
  }

  fmt::print("{:=^80}\n", "");
}

#include "ND_func.cxx"

// -------------------------------------------------------------------
// Funzione di test SPSA verbosa
// -------------------------------------------------------------------
template <typename Problem>
void test(Problem & prob, std::string const & problem_name) {
  
  fmt::print( "\n\nSTART: {}\n", problem_name );

  // Parametri SPSA più robusti
  Utils::SPSA_minimizer<Scalar>::Options opts;
  opts.max_iter = 500;        // più iterazioni
  opts.a0 = 0.1;              // learning rate più piccolo
  opts.c0 = 0.5;             // perturbazioni più piccole
  opts.alpha = 0.602;
  opts.gamma = 0.101;
  opts.gradient_avg = 1;       // più medie per il gradiente
  opts.verbose = true;

  Utils::SPSA_minimizer<Scalar> optimizer(opts);
  optimizer.set_bounds(prob.lower(), prob.upper());

  Vector x0 = prob.init();
  Vector x_final;

  auto iter_data = optimizer.minimize(x0, prob);

  TestResult result;
  result.problem_name    = problem_name;
  result.linesearch_name = "SPSA";
  result.iteration_data  = iter_data;
  result.final_value     = iter_data.final_f;
  result.final_solution  = iter_data.final_x;
  result.dimension       = x0.size();

  global_test_results.push_back(result);
  update_line_search_statistics(result);

  fmt::print(
    "{}: final f = {:.6e}, iterations = {}\n{}\n\n\n",
    problem_name, iter_data.final_f,
    iter_data.iterations,
    iter_data.final_x.transpose()
  );
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int main() {
  fmt::print("Esecuzione test SPSA_minimizer...\n");

  // Test originali
  Rosenbrock2D<Scalar> rosen;
  test( rosen, "Rosenbrock2D" );

  NesterovChebyshevRosenbrock<Scalar,128> nesterov;
  test( nesterov, "NesterovChebyshevRosenbrock" );

  RosenbrockN<Scalar,10> rosenN;
  test( rosenN, "Rosenbrock10D" );

  PowellSingularN<Scalar,16> powerllN;
  test( powerllN, "PowellSingular16D" );

  ExtendedWoodN<Scalar,16> woodN;
  test( woodN, "ExtendedWood16D" );

  // Nuovi problemi
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

  SchwefelN<Scalar,15> Schwefel;
  test( Schwefel, "SchwefelN15D" );

  AckleyN<Scalar,15> Ackley;
  test( Ackley, "AckleyN15D" );

  RastriginN<Scalar,15> Rastrigin;
  test( Rastrigin, "RastriginN15D" );

  print_summary_table();
  print_line_search_statistics();

  return 0;
}
