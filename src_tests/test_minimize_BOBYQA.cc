/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  BOBYQA test suite per problemi di ottimizzazione senza gradienti        |
 |                                                                          |
 |  Adattamento per la classe BOBYQA_minimizer di Utils_minimize_BOBYQA.hh  |
\*--------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "Utils_minimize_BOBYQA.hh"

using std::string;

using Scalar  = double;
using Vector  = Utils::BOBYQA_minimizer<Scalar>::Vector;
using Status  = Utils::BOBYQA_minimizer<Scalar>::Status;
using integer = Utils::BOBYQA_minimizer<Scalar>::integer;

// Risultato dell'ottimizzazione per il test
struct Result
{
  bool   converged;
  size_t iterations;
  Scalar final_f;
  Vector final_x;
};

// Struttura per raccogliere i risultati dei test
struct TestResult
{
  string  problem_name;
  string  linesearch_name;
  Result  iteration_data;
  Scalar  final_value;
  Vector  final_solution;
  integer dimension;
};

// Statistiche line search
struct LineSearchStats
{
  std::string name;
  size_t      total_tests{ 0 };
  size_t      successful_tests{ 0 };
  size_t      total_iterations{ 0 };
  size_t      total_function_evals{ 0 };
};

// Collettore globale
std::vector<TestResult>                global_test_results;
std::map<std::string, LineSearchStats> line_search_statistics;

// -------------------------------------------------------------------
// Aggiorna statistiche
// -------------------------------------------------------------------
void
update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics[result.linesearch_name];
  stats.name   = result.linesearch_name;
  stats.total_tests++;
  if ( result.iteration_data.converged )
  {
    stats.successful_tests++;
    stats.total_iterations += result.iteration_data.iterations;
  }
}

// -------------------------------------------------------------------
// Stampa statistiche
// -------------------------------------------------------------------
void
print_line_search_statistics()
{
  fmt::print( "\n\n{:=^80}\n", " BOBYQA STATISTICS " );
  fmt::print( "{:<15} {:<8} {:<8} {:<12} {:<10}\n", "Optimizer", "Tests", "Success", "Success%", "AvgIter" );
  fmt::print( "{:-<80}\n", "" );

  for ( const auto & [name, stats] : line_search_statistics )
  {
    Scalar success_rate   = ( stats.total_tests > 0 ) ? 100.0 * stats.successful_tests / stats.total_tests : 0.0;
    Scalar avg_iterations = ( stats.successful_tests > 0 )
                                ? static_cast<Scalar>( stats.total_iterations ) / stats.successful_tests
                                : 0.0;
    auto   color          = ( success_rate >= 80.0 )   ? fmt::fg( fmt::color::green )
                            : ( success_rate >= 60.0 ) ? fmt::fg( fmt::color::yellow )
                                                       : fmt::fg( fmt::color::red );

    fmt::print( "{:<15} {:<8} {:<8} ", stats.name, stats.total_tests, stats.successful_tests );
    fmt::print( color, "{:<12.1f}", success_rate );
    fmt::print( " {:<10.1f}\n", avg_iterations );
  }
  fmt::print( "{:=^80}\n", "" );
}

// -------------------------------------------------------------------
// Stampa tabella riassuntiva
// -------------------------------------------------------------------
void
print_summary_table()
{
  fmt::print( "\n\n{:=^80}\n", " SUMMARY TEST RESULTS " );
  fmt::print( "{:<28} {:<12} {:<8} {:<12} {:<15} {:<10}\n", "Problem", "Optimizer", "Dimension", "Iterations",
              "final f(x)", "Status" );
  fmt::print( "{:-<80}\n", "" );

  for ( auto const & result : global_test_results )
  {
    std::string  status_str = result.iteration_data.converged ? "CONVERGED" : "MAX_ITER";
    auto const & GREEN      = fmt::fg( fmt::color::green );
    auto const & RED        = fmt::fg( fmt::color::red );

    fmt::print( "{:<28} {:<12} {:<8} {:<12} {:<15.6e}", result.problem_name, result.linesearch_name, result.dimension,
                result.iteration_data.iterations, result.final_value );

    if ( result.iteration_data.converged )
      fmt::print( GREEN, "{}\n", status_str );
    else
      fmt::print( RED, "{}\n", status_str );
  }

  fmt::print( "{:=^80}\n", "" );
}

#include "ND_func.cxx"

// -------------------------------------------------------------------
// Funzione di test BOBYQA
// -------------------------------------------------------------------
template <typename Problem>
void
test( Problem & prob, std::string const & problem_name )
{
  fmt::print( "\n\nSTART: {}\n", problem_name );

  // Dimensione del problema e punto iniziale
  Vector        x0 = prob.init();
  integer const n  = x0.size();

  // Bound del problema
  Vector lower = prob.lower();
  Vector upper = prob.upper();

  // Calcolo NPT (numero di punti di interpolazione)
  // Raccomandato: 2*n+1, ma deve essere tra n+2 e (n+1)(n+2)/2
  integer npt     = 2 * n + 1;
  integer npt_min = n + 2;
  integer npt_max = ( n + 1 ) * ( n + 2 ) / 2;

  if ( npt < npt_min ) npt = npt_min;
  if ( npt > npt_max ) npt = npt_max;

  // Wrapper per la funzione obiettivo che conta le valutazioni
  size_t nfev   = 0;
  auto   objfun = [&prob, &nfev]( Vector const & x ) -> Scalar
  {
    ++nfev;
    return prob( x );
  };

  // Creazione dell'ottimizzatore BOBYQA
  Utils::BOBYQA_minimizer<Scalar> optimizer;

  // Impostazione dei parametri
  optimizer.set_verbosity( 1 );     // Output di base
  optimizer.set_maxfun( 5000 );     // Massimo numero di valutazioni
  optimizer.set_rho( 0.1, 0.001 );  // RHOBEG e RHOEND

  // Chiamata all'algoritmo di ottimizzazione
  Status status = optimizer.minimize( n, npt, objfun, x0, lower, upper );

  // Valutazione finale nel punto ottimizzato
  Scalar final_f = prob( x0 );

  // Costruzione del risultato per il test
  Result iteration_data;
  iteration_data.converged  = ( status == Status::BOBYQA_SUCCESS );
  iteration_data.iterations = nfev;
  iteration_data.final_f    = final_f;
  iteration_data.final_x    = x0;

  TestResult result;
  result.problem_name    = problem_name;
  result.linesearch_name = "BOBYQA";
  result.iteration_data  = iteration_data;
  result.final_value     = final_f;
  result.final_solution  = x0;
  result.dimension       = n;

  global_test_results.push_back( result );
  update_line_search_statistics( result );

  // Output dei risultati
  std::string status_str = iteration_data.converged ? "CONVERGED" : "FAILED";
  fmt::print( "{}: status = {}, final f = {:.6e}, nfev = {}\n", problem_name, status_str, final_f, nfev );
  fmt::print( "Solution: {}\n\n", x0.transpose() );
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int
main()
{
  fmt::print( "Esecuzione test BOBYQA_minimizer...\n" );

  // Test con problemi standard
  Rosenbrock2D<Scalar> rosen;
  test( rosen, "Rosenbrock2D" );

  NesterovChebyshevRosenbrock<Scalar, 128> nesterov;
  test( nesterov, "NesterovChebyshevRosenbrock" );

  RosenbrockN<Scalar, 10> rosenN;
  test( rosenN, "Rosenbrock10D" );

  PowellSingularN<Scalar, 16> powerllN;
  test( powerllN, "PowellSingular16D" );

  ExtendedWoodN<Scalar, 16> woodN;
  test( woodN, "ExtendedWood16D" );

  // Problemi aggiuntivi
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

  BrownAlmostLinearN<Scalar, 10> brown;
  test( brown, "BrownAlmostLinear10D" );

  BroydenTridiagonalN<Scalar, 12> broy;
  test( broy, "BroydenTridiagonal12D" );

  IllConditionedQuadraticN<Scalar, 20> illq;
  test( illq, "IllConditionedQuadratic20D" );

  TrigonometricSumN<Scalar, 15> trig;
  test( trig, "TrigonometricSum15D" );

  SchwefelN<Scalar, 15> Schwefel;
  test( Schwefel, "SchwefelN15D" );

  AckleyN<Scalar, 15> Ackley;
  test( Ackley, "AckleyN15D" );

  RastriginN<Scalar, 15> Rastrigin;
  test( Rastrigin, "RastriginN15D" );

  // Stampa dei risultati
  print_summary_table();
  print_line_search_statistics();

  return 0;
}
