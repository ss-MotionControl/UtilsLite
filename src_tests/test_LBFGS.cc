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

using std::string;
using std::vector;
using std::map;
using std::pair;
using Scalar    = double;
//using MINIMIZER = Utils::LBFGS_minimizer<Scalar>;
using MINIMIZER = Utils::LBFGS_BlockCoordinate<Scalar>;
using Vector    = typename MINIMIZER::Vector;

// Usa gli Status definiti in Utils_LBFGS.hh
using Status = MINIMIZER::Status;

// Struttura per raccogliere i risultati dei test
struct TestResult {
  string            problem_name;
  string            linesearch_name;
  MINIMIZER::Result result_data;
  Scalar            final_value;
  Vector            final_solution;
  size_t            dimension;
  Status            status;
};

// Struttura per statistiche delle line search
struct LineSearchStats {
  string name;
  size_t total_tests{0};
  size_t successful_tests{0};
  size_t total_iterations{0};
  size_t total_function_evals{0};
  Scalar average_iterations{0};
  Scalar success_rate{0};
};

// Collettore globale dei risultati
vector<TestResult> global_test_results;
map<string, LineSearchStats> line_search_statistics;

#include "ND_func_grad.cxx"

// -------------------------------------------------------------------
// Funzione per formattare il vettore (simile a NelderMead)
// -------------------------------------------------------------------
inline
string
format_reduced_vector( Vector const & v, size_t max_size = 10 ) {
  string tmp{"["};
  size_t v_size = v.size();
  if ( v_size <= max_size ) {
    for ( size_t i = 0; i < v_size; ++i)
      tmp += fmt::format("{:.4f}, ", v(i));
  } else {
    for ( size_t i{0}; i < max_size-3; ++i)
      tmp += fmt::format("{:.4f}, ", v(i));
    tmp += "..., ";
    for ( size_t i{v_size-3}; i < v_size; ++i )
      tmp += fmt::format("{:.4f}, ", v(i));
  }
  tmp.pop_back();
  tmp.pop_back();
  tmp += "]";
  return tmp;
}

// -------------------------------------------------------------------
// Funzione per convertire status in stringa
// -------------------------------------------------------------------
string
status_to_string( Status status ) {
  switch (status) {
    case Status::CONVERGED:            return "CONVERGED";
    case Status::MAX_OUTER_ITERATIONS: return "MAX_OUTER_ITER";
    case Status::MAX_INNER_ITERATIONS: return "MAX_INNER_ITER";
    case Status::LINE_SEARCH_FAILED:   return "LINE_SEARCH_FAILED";
    case Status::GRADIENT_TOO_SMALL:   return "GRAD_SMALL";
    case Status::FAILED:               return "FAILED";
    default:                           return "UNKNOWN";
  }
}

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
    stats.total_iterations     += result.result_data.total_iterations;
    stats.total_function_evals += result.result_data.total_evaluations;
  }
}

// -------------------------------------------------------------------
// Funzione per stampare le statistiche delle line search
// -------------------------------------------------------------------
void
print_line_search_statistics() {
  fmt::print(fmt::fg(fmt::color::light_blue),
    "\n\n"
    "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                  LINE SEARCH STATISTICS                                  ║\n"
    "╠══════════════════╤══════════╤══════════╤════════════╤═════════════╤══════════════════════╣\n"
    "║    LineSearch    │ Tests    │ Success  │  Success % │   AvgIter   │     AvgFuncEval      ║\n"
    "╠══════════════════╪══════════╪══════════╪════════════╪═════════════╪══════════════════════╣\n"
  );

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
    
    fmt::print("║ {:<16} │ {:>8} │ {:>8} │ ",  stats.name, stats.total_tests, stats.successful_tests);
    fmt::print(color, " {:>8} ", fmt::format( "{:.1f}%", success_rate) );
    fmt::print(" │ {:>11.1f} │ {:>20.1f} ║\n", avg_iterations, avg_func_evals);
  }
  
  fmt::print(fmt::fg(fmt::color::light_blue),
    "╚══════════════════╧══════════╧══════════╧════════════╧═════════════╧══════════════════════╝\n"
  );
}

// -------------------------------------------------------------------
// Funzione per stampare la tabella riassuntiva
// -------------------------------------------------------------------
void
print_summary_table() {
  fmt::print(fmt::fg(fmt::color::light_blue),
    "\n\n"
    "╔══════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                  L-BFGS SUMMARY RESULTS                                  ║\n"
    "╠════════════════════════╤════════╤══════════════╤══════════╤════════════════╤═════════════╣\n"
    "║ Function               │ Dim    │ LineSearch   │ Iter     │ Final Value    │ Status      ║\n"
    "╠════════════════════════╪════════╪══════════════╪══════════╪════════════════╪═════════════╣\n"
  );
    
  for ( auto const & result : global_test_results ) {
    string status_str = status_to_string(result.status);
    bool converged = (result.status == Status::CONVERGED ||
                      result.status == Status::GRADIENT_TOO_SMALL);
        
    // Usa colori: verde per convergenza, giallo per warning, rosso per fallimento
    auto status_color = converged ? fmt::fg(fmt::color::green) :
                        (result.status == Status::MAX_OUTER_ITERATIONS) ?
                        fmt::fg(fmt::color::yellow) : fmt::fg(fmt::color::red);
    
    // Tronca il nome del problema se troppo lungo
    string problem_name = result.problem_name;
    if (problem_name.length() > 22) {
      problem_name = problem_name.substr(0, 19) + "...";
    }
    
    fmt::print(
      "║ {:<22} │ {:>6} │ {:<12} │ {:>8} │ {:<14.4e} │ ",
      problem_name,
      result.dimension,
      result.linesearch_name,
      result.result_data.total_iterations,
      result.final_value
    );

    fmt::print(status_color, "{:<11}", status_str);
    fmt::print(fmt::fg(fmt::color::light_blue), " ║\n");
  }

  fmt::print(fmt::fg(fmt::color::light_blue),
    "╚════════════════════════╧════════╧══════════════╧══════════╧════════════════╧═════════════╝\n"
  );

  // Statistiche finali
  size_t total_tests     = global_test_results.size();
  size_t converged_tests = std::count_if(global_test_results.begin(), global_test_results.end(),
    [](const TestResult& r) {
       return r.status == Status::CONVERGED ||
              r.status == Status::GRADIENT_TOO_SMALL;
  });
  
  size_t accumulated_iter{0};
  size_t accumulated_evals{0};
  for ( auto const & r : global_test_results ) {
     if ( r.status == Status::CONVERGED ||
          r.status == Status::GRADIENT_TOO_SMALL ) {
       accumulated_iter += r.result_data.total_iterations;
       accumulated_evals += r.result_data.total_evaluations;
     }
  }
  
  fmt::print(fmt::fg(fmt::color::light_blue), "\n📊 Global Statistics:\n");
  fmt::print("   • Total problems: {}\n", total_tests);
  fmt::print("   • Converged: {} ({:.1f}%)\n", converged_tests, 
             (100.0 * converged_tests / total_tests));
  fmt::print("   • Total iterations: {}\n", accumulated_iter);
  fmt::print("   • Total function evaluations: {}\n", accumulated_evals);
}

// -------------------------------------------------------------------
// test runner modificato per raccogliere risultati
// -------------------------------------------------------------------
template <typename T, size_t N>
static
void
test( OptimizationProblem<T,N> const * tp, string const & problem_name ){

  fmt::print(fmt::fg(fmt::color::cyan),
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║ TEST FUNCTION: {:<47} ║\n"
    "║ Dimension:     {:<47} ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n",
    problem_name, N
  );

  using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  MINIMIZER::Options opts;
  //opts.max_iter        = 20000;
  //opts.m               = 20;
  opts.verbosity_level = 2;  // Verbosity like NelderMead: 0=quiet, 1=outer, 2=inner, 3=detailed
  //opts.use_projection  = true;

  Vector x0 = tp->init();

  auto cb = [&tp]( Vector const & x, Vector * g ) -> Scalar { return (*tp)( x, g ); };

  // Lista di line search da testare
  vector<pair<string, std::function<std::optional<std::tuple<Scalar,size_t>>(
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

#if 0

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
#endif

  for (const auto& [ls_name, line_search] : line_searches) {
  
    MINIMIZER minimizer(opts);
    minimizer.set_bounds( tp->lower(), tp->upper() );
    
    auto solution_data = minimizer.minimize( x0, cb, line_search );
    
    // Salva il risultato
    TestResult result;
    result.problem_name    = problem_name;
    result.linesearch_name = ls_name;
    result.result_data     = solution_data;
    result.final_value     = solution_data.final_function_value;
    result.final_solution  = solution_data.solution;
    result.dimension       = N;
    result.status          = solution_data.status;
    
    global_test_results.push_back(result);
    update_line_search_statistics(result);
    
    string status_str = status_to_string(result.status);
    
    fmt::print("{} - {}: {} after {} iterations, f = {:.6e}\n",
                problem_name, ls_name, status_str,
                solution_data.total_iterations,
                solution_data.final_function_value);
    fmt::print("Solution: {}\n\n", format_reduced_vector(result.final_solution,10));
  }
  fmt::print("\n");
}

// -------------------------------------------------------------------
// main: run original 4 tests plus the 10 new ones
// -------------------------------------------------------------------
int
main(){

  fmt::print(fmt::fg(fmt::color::light_blue),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║                 L-BFGS Optimization Test Suite                 ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n\n"
  );

#if 0
  // Test originali
  Rosenbrock2D<Scalar> rosen;
  test( &rosen, "Rosenbrock2D" );

  NesterovChebyshevRosenbrock<Scalar,128> nesterov;
  test( &nesterov, "NesterovChebyshevRosenbrock" );
#endif

  RosenbrockN<Scalar,30> rosenN;
  test( &rosenN, "Rosenbrock30D" );

#if 0
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
#endif

  // Stampa tabella riassuntiva
  print_summary_table();
  
  // Stampa statistiche delle line search
  print_line_search_statistics();

  return 0;
}
