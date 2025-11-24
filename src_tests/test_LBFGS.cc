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

// Usa gli Status definiti in Utils_LBFGS.hh
using Status = Utils::LBFGS_minimizer<Scalar>::Status;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  std::string                                   problem_name;
  std::string                                   linesearch_name;
  Utils::LBFGS_minimizer<Scalar>::IterationData iteration_data;
  Scalar                                        final_value;
  Vector                                        final_solution;
  size_t                                        dimension;
  Status                                        status;
};

// Struttura per statistiche delle line search
struct LineSearchStats {
  std::string name;
  size_t total_tests{0};
  size_t successful_tests{0};
  size_t total_iterations{0};
  size_t total_function_evals{0};
  Scalar average_iterations{0};
  Scalar success_rate{0};
};

// Collettore globale dei risultati
std::vector<TestResult> global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

#include "ND_func_grad.cxx"

// -------------------------------------------------------------------
// Funzione per aggiornare le statistiche delle line search
// -------------------------------------------------------------------
void
update_line_search_statistics(const TestResult& result) {
  auto& stats = line_search_statistics[result.linesearch_name];
  stats.name = result.linesearch_name;
  stats.total_tests++;
  
  bool success = (result.status == Status::CONVERGED ||
                  result.status == Status::GRADIENT_TOO_SMALL);
  
  if (success) {
    stats.successful_tests++;
    stats.total_iterations     += result.iteration_data.iterations;
    stats.total_function_evals += result.iteration_data.function_evaluations;
  }
}

// -------------------------------------------------------------------
// Funzione per stampare le statistiche delle line search
// -------------------------------------------------------------------
void
print_line_search_statistics() {
  fmt::print("\n\n{:=^80}\n", " LINE SEARCH STATISTICS ");
  fmt::print("{:<15} {:<8} {:<8} {:<12} {:<10} {:<10}\n",
             "LineSearch", "Tests", "Success", "Success%", "AvgIter", "AvgFuncEval" );
  fmt::print("{:-<80}\n", "");
  
  for (const auto& [name, stats] : line_search_statistics) {
    Scalar success_rate = (stats.total_tests > 0) ? 
      (100.0 * stats.successful_tests) / stats.total_tests : 0.0;
    Scalar avg_iterations = (stats.successful_tests > 0) ?
      static_cast<Scalar>(stats.total_iterations) / stats.successful_tests : 0.0;
    Scalar avg_func_evals = (stats.successful_tests > 0) ?
      static_cast<Scalar>(stats.total_function_evals) / stats.successful_tests : 0.0;
    
    // Colore in base al success rate
    auto color = (success_rate >= 80.0) ? fmt::fg(fmt::color::green) :
                 (success_rate >= 60.0) ? fmt::fg(fmt::color::yellow) :
                 fmt::fg(fmt::color::red);
    
    fmt::print("{:<15} {:<8} {:<8} ", 
               stats.name, stats.total_tests, stats.successful_tests);
    fmt::print(color, "{:<12} ", fmt::format( "{:.1f}%",success_rate));
    fmt::print("{:<10.1f} {:<12.1f}\n", avg_iterations, avg_func_evals);
  }
  fmt::print("{:=^80}\n", "");
}

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
    case Status::CONVERGED:
      status_str = "CONVERGED";
      converged = true;
      break;
    case Status::MAX_ITERATIONS:
      status_str = "MAX_ITER";
      converged = false;
      break;
    case Status::LINE_SEARCH_FAILED:
      status_str = "LINE_SEARCH_FAILED";
      converged = false;
      break;
    case Status::GRADIENT_TOO_SMALL:
      status_str = "GRAD_SMALL";
      converged = true;
      break;
    case Status::FAILED:
      status_str = "FAILED";
      converged = false;
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
       return r.status == Status::CONVERGED ||
              r.status == Status::GRADIENT_TOO_SMALL;
  });
  size_t accumulated_iter{0};
  for ( auto const & r : global_test_results ) {
     if ( r.status == Status::CONVERGED ||
          r.status == Status::GRADIENT_TOO_SMALL ) accumulated_iter += r.iteration_data.iterations;
  }
    
  fmt::print("\nFinal Stats: {} / {} test converged ({:.1f}%), accumulated iter: {}\n",
             converged_tests, total_tests, (100.0 * converged_tests / total_tests), accumulated_iter );
}

// -------------------------------------------------------------------
// test runner modificato per raccogliere risultati
// -------------------------------------------------------------------
template <typename T, size_t N>
static
void
test( OptimizationProblem<T,N> const * tp, const std::string& problem_name ){

  fmt::print( "\n\n\n\nTEST: {}\n", problem_name );

  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  Utils::LBFGS_minimizer<Scalar>::Options opts;
  opts.max_iter       = 20000;
  opts.m              = 20;
  opts.verbose        = true;
  opts.use_projection = true;

  Vector x0 = tp->init();

  auto cb = [&tp]( Vector const & x, Vector * g ) -> Scalar { return (*tp)( x, g ); };

  // Lista di line search da testare
  std::vector<std::pair<std::string, std::function<std::optional<std::tuple<Scalar,size_t>>(
    Scalar, Scalar, Vector const&, Vector const&, 
    std::function<Scalar(Vector const&, Vector*)>, Scalar)>>> line_searches;

  // Inizializza le line search
  Utils::WeakWolfeLineSearch<Scalar>   wolfe_weak;
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

  line_searches.emplace_back("WolfeWeak", [&wolfe_weak](
    Scalar f0, Scalar Df0, Vector const& x, Vector const& d,
    std::function<Scalar(Vector const&, Vector*)> callback, Scalar alpha0) {
      return wolfe_weak(f0, Df0, x, d, callback, alpha0);
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
    
    auto iter_data = minimizer.minimize( x0, cb, line_search );
    
    // Salva il risultato
    TestResult result;
    result.problem_name    = problem_name;
    result.linesearch_name = ls_name;
    result.iteration_data  = iter_data;
    result.final_value     = iter_data.final_function_value;
    result.final_solution  = minimizer.solution();
    result.dimension       = N;
    result.status          = iter_data.status;
    
    global_test_results.push_back(result);
    update_line_search_statistics(result);
    
    // Converti status in stringa
    std::string status_str;
    switch (result.status) {
    case Status::CONVERGED:
      status_str = "CONVERGED";
      break;
    case Status::MAX_ITERATIONS:
      status_str = "MAX_ITERATIONS";
      break;
    case Status::LINE_SEARCH_FAILED:
      status_str = "LINE_SEARCH_FAILED";
      break;
    case Status::GRADIENT_TOO_SMALL:
      status_str = "GRADIENT_TOO_SMALL";
      break;
    case Status::FAILED:
      status_str = "FAILED";
      break;
    default:
      status_str = "UNKNOWN";
    }
      
    fmt::print("{} - {}: {} after {} iterations, f = {:.6e}\n{}\n",
                problem_name, ls_name, status_str,
                iter_data.iterations,
                iter_data.final_function_value,
                result.final_solution.transpose() );
  }
  fmt::print("\n\n");
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

  NesterovChebyshevRosenbrock<Scalar,128> nesterov;
  test( &nesterov, "NesterovChebyshevRosenbrock" );

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
  
  // Stampa statistiche delle line search
  print_line_search_statistics();

  return 0;
}
