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
  bool   converged;
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
  auto& stats = line_search_statistics["MADS"];
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

#include "ND_func.cxx"

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
  opts.print_every = 50;

  Utils::MADS_minimizer<Scalar> optimizer(opts);
  optimizer.set_bounds(prob.lower(), prob.upper());

  Vector x0 = prob.init();
  Scalar initial_f = prob(x0);

  auto result_data = optimizer.minimize(x0, prob);

  TestResult result;
  result.test_name = problem_name;
  result.initial_f = initial_f;
  result.final_f = result_data.final_f;
  result.error = 0.0;
  result.iterations = result_data.iterations;
  result.f_eval_count = result_data.f_eval_count;
  result.dimension = static_cast<size_t>(x0.size());
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
  
  if (x0.size() <= 10) {
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
