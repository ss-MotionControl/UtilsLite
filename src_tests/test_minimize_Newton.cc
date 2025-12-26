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

#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <vector>
#include <optional>
#include <functional>
#include <limits>
#include <algorithm>

#include "Utils_minimize_Newton.hh"

using std::map;
using std::pair;
using std::string;
using std::vector;
using Scalar       = double;
using MINIMIZER    = Utils::Newton_minimizer<Scalar>;
using Vector       = typename MINIMIZER::Vector;
using SparseMatrix = typename MINIMIZER::SparseMatrix;

// Usa gli Status definiti in Utils_minimize_Newton.hh
using Status = MINIMIZER::Status;

// Struttura per raccogliere i risultati dei test
struct TestResult
{
  string            problem_name;
  string            linesearch_name;
  MINIMIZER::Result result_data;
  Vector            final_solution;
  size_t            dimension;
  Status            status;
  Scalar            final_gradient_norm{ 0 };
};

// Struttura per statistiche delle line search
struct LineSearchStats
{
  string name;
  size_t total_tests{ 0 };
  size_t successful_tests{ 0 };
  size_t total_iterations{ 0 };
  size_t total_evaluations{ 0 };
  Scalar total_final_gradient_norm{ 0 };

  Scalar average_iterations{ 0 };
  Scalar average_final_gradient_norm{ 0 };
  Scalar success_rate{ 0 };
};

// Collettore globale dei risultati
vector<TestResult>           global_test_results;
map<string, LineSearchStats> line_search_statistics;

#include "ND_func.cxx"

// -------------------------------------------------------------------
// Funzione per formattare il vettore (simile a NelderMead)
// -------------------------------------------------------------------
inline string
format_reduced_vector( Vector const & v, size_t max_size = 10 )
{
  string tmp{ "[" };
  size_t v_size = v.size();

  if ( v_size == 0 ) { return "[]"; }

  if ( v_size <= max_size )
  {
    for ( size_t i = 0; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }
  else
  {
    for ( size_t i{ 0 }; i < max_size - 3; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
    tmp += "..., ";
    for ( size_t i{ v_size - 3 }; i < v_size; ++i ) tmp += fmt::format( "{:.4f}, ", v( i ) );
  }

  // Rimuovi la virgola finale solo se ci sono elementi
  if ( v_size > 0 )
  {
    tmp.pop_back();
    tmp.pop_back();
  }
  tmp += "]";
  return tmp;
}

// -------------------------------------------------------------------
// Funzione per aggiornare le statistiche delle line search
// -------------------------------------------------------------------
void
update_line_search_statistics( const TestResult & result )
{
  auto & stats = line_search_statistics[result.linesearch_name];
  stats.name   = result.linesearch_name;
  stats.total_tests++;

  bool success = ( result.status == Status::CONVERGED || result.status == Status::GRADIENT_TOO_SMALL );

  if ( success )
  {
    stats.successful_tests++;
    stats.total_iterations += result.result_data.total_iterations;
    stats.total_evaluations += result.result_data.total_evaluations;

    // Calcola le medie
    stats.average_iterations = static_cast<Scalar>( stats.total_iterations ) / stats.successful_tests;
    stats.success_rate       = ( 100.0 * stats.successful_tests ) / stats.total_tests;
  }
  else
  {
    // Aggiorna solo il tasso di successo per i test falliti
    stats.success_rate = ( 100.0 * stats.successful_tests ) / stats.total_tests;
  }
}

// -------------------------------------------------------------------
// Tabella riassuntiva per Line Search
// -------------------------------------------------------------------

void
print_line_search_statistics()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════"
    "══════════════════════╗\n"
    "║                            LINE SEARCH SUMMARY                 "
    "                      ║\n"
    "╠══════════════════╤══════════╤════════════╤════════════╤════════"
    "════╤═════════════════╣\n"
    "║ LineSearch       │ Tests    │ Success %  │ Avg Iter   │ Avg "
    "Eval   │ Avg ‖g_proj‖    ║\n"
    "╠══════════════════╪══════════╪════════════╪════════════╪════════"
    "════╪═════════════════╣\n" );

  for ( auto const & [_, s] : line_search_statistics )
  {
    auto color = ( s.success_rate >= 80.0 )   ? fmt::fg( fmt::color::green )
                 : ( s.success_rate >= 60.0 ) ? fmt::fg( fmt::color::yellow )
                                              : fmt::fg( fmt::color::red );

    fmt::print( "║ {:<16} │ {:>8} │ ", s.name, s.total_tests );
    fmt::print( color, "{:>9.1f}% ", s.success_rate );
    fmt::print(
      "│ {:>10.1f} │ {:>10.1f} │ {:>15.3e} ║\n",
      s.average_iterations,
      Scalar( s.total_evaluations ) / std::max<size_t>( s.successful_tests, 1 ),
      s.average_final_gradient_norm );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚══════════════════╧══════════╧════════════╧════════════╧════════"
    "════╧═════════════════╝\n" );
}

// -------------------------------------------------------------------
// Funzione per stampare la tabella riassuntiva
// -------------------------------------------------------------------
void
print_summary_table()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "\n\n"
    "╔════════════════════════════════════════════════════════════════"
    "════════════════════════════════════╗\n"
    "║                                      NEWTON SUMMARY RESULTS    "
    "                                    ║\n"
    "╠════════════════════════╤════════╤══════════════╤══════════╤════"
    "════════════╤═══════════════════════╣\n"
    "║ Function               │ Dim    │ LineSearch   │ Iter     │ "
    "Final Value    │ Status                ║\n"
    "╠════════════════════════╪════════╪══════════════╪══════════╪════"
    "════════════╪═══════════════════════╣"
    "\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = MINIMIZER::to_string( result.status );
    bool   converged  = result.status == Status::CONVERGED || result.status == Status::GRADIENT_TOO_SMALL;

    // Usa colori: verde per convergenza, rosso per fallimento
    auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

    // Tronca il nome del problema se troppo lungo
    string problem_name = result.problem_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:<12} │ {:>8} │ {:<14.4e} │ ",
      problem_name,
      result.dimension,
      result.linesearch_name,
      result.result_data.total_iterations,
      result.result_data.final_function_value );

    fmt::print( status_color, "{:<21}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╚════════════════════════╧════════╧══════════════╧══════════╧════"
    "════════════╧═══════════════════════╝"
    "\n" );

  // Statistiche finali
  size_t total_tests     = global_test_results.size();
  size_t converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.status == Status::CONVERGED || r.status == Status::GRADIENT_TOO_SMALL; } );

  size_t accumulated_iter{ 0 };
  size_t accumulated_evals{ 0 };
  for ( auto const & r : global_test_results )
  {
    if ( r.status == Status::CONVERGED || r.status == Status::GRADIENT_TOO_SMALL )
    {
      accumulated_iter += r.result_data.total_iterations;
      accumulated_evals += r.result_data.total_evaluations;
    }
  }

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print(
    "   • Converged: {} ({:.1f}%)\n",
    converged_tests,
    ( 100.0 * converged_tests / std::max<size_t>( total_tests, 1 ) ) );
  fmt::print( "   • Total iterations: {}\n", accumulated_iter );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
}

// -------------------------------------------------------------------
// Template helper per verificare se un tipo ha metodo hessian
// -------------------------------------------------------------------
template <typename T, typename = void>
struct has_hessian : std::false_type
{
};

template <typename T>
struct has_hessian<T, std::void_t<decltype( std::declval<T>().hessian( std::declval<Vector>() ) )>> : std::true_type
{
};

// -------------------------------------------------------------------
// Test runner
// -------------------------------------------------------------------

template <typename Problem>
static void
test( Problem & tp, string const & name )
{
  MINIMIZER::Options opts;
  opts.max_iter        = 20000;
  opts.g_tol           = 1e-6;
  opts.g_tol_weak      = 1e-4;
  opts.f_tol           = 1e-10;
  opts.x_tol           = 1e-8;
  opts.step_max        = 10.0;
  opts.very_small_step = 1e-8;
  opts.verbosity_level = 2;

  Vector x0 = tp.init();

  auto cb = [&tp]( Vector const & x, Vector * g, SparseMatrix * H ) -> Scalar
  {
    if ( g ) *g = tp.gradient( x );

    if ( H )
    {
      if constexpr ( has_hessian<Problem>::value ) { *H = tp.hessian( x ); }
      else
      {
        H->resize( x.size(), x.size() );
        H->setZero();

        Scalar eps = std::sqrt( std::numeric_limits<Scalar>::epsilon() );
        Vector xp = x, xm = x;

        for ( int i = 0; i < x.size(); ++i )
        {
          xp( i ) += eps;
          xm( i ) -= eps;

          Vector gp = tp.gradient( xp );
          Vector gm = tp.gradient( xm );

          for ( int j = 0; j < x.size(); ++j ) H->coeffRef( j, i ) = ( gp( j ) - gm( j ) ) / ( 2 * eps );

          xp( i ) = xm( i ) = x( i );
        }

        SparseMatrix Ht = H->transpose();
        *H              = 0.5 * ( ( *H ) + Ht );
      }
    }
    return tp( x );
  };

  vector<pair<
    string,
    std::function<std::optional<std::tuple<Scalar, size_t>>(
      Scalar,
      Scalar,
      Vector const &,
      Vector const &,
      std::function<Scalar( Vector const &, Vector * )>,
      Scalar )>>>
    line_searches;

  Utils::ArmijoLineSearch<Scalar>      armijo;
  Utils::WeakWolfeLineSearch<Scalar>   wolfe_w;
  Utils::StrongWolfeLineSearch<Scalar> wolfe_s;
  Utils::GoldsteinLineSearch<Scalar>   gold;
  Utils::HagerZhangLineSearch<Scalar>  hz;
  Utils::MoreThuenteLineSearch<Scalar> more;

  line_searches.emplace_back( "Armijo", [&]( auto... a ) { return armijo( a... ); } );
  line_searches.emplace_back( "WeakWolfe", [&]( auto... a ) { return wolfe_w( a... ); } );
  line_searches.emplace_back( "StrongWolfe", [&]( auto... a ) { return wolfe_s( a... ); } );
  line_searches.emplace_back( "Goldstein", [&]( auto... a ) { return gold( a... ); } );
  line_searches.emplace_back( "HagerZhang", [&]( auto... a ) { return hz( a... ); } );
  line_searches.emplace_back( "MoreThuente", [&]( auto... a ) { return more( a... ); } );

  for ( auto const & [ls_name, ls] : line_searches )
  {
    MINIMIZER m( opts );
    try
    {
      m.set_bounds( tp.lower(), tp.upper() );
    }
    catch ( ... )
    {
    }

    auto res = m.minimize( x0, cb, ls );

    TestResult tr;
    tr.problem_name        = name;
    tr.linesearch_name     = ls_name;
    tr.result_data         = res;
    tr.final_solution      = m.solution();
    tr.dimension           = tr.final_solution.size();
    tr.status              = res.status;
    tr.final_gradient_norm = res.final_gradient_norm;

    global_test_results.push_back( tr );
    update_line_search_statistics( tr );

    fmt::print(
      "{} - {}: {} | iter = {} | f = {:.6e} | ‖g_proj‖ = {:.3e}\n",
      name,
      ls_name,
      MINIMIZER::to_string( tr.status ),
      res.total_iterations,
      res.final_function_value,
      tr.final_gradient_norm );

    fmt::print( "Solution: {}\n\n", format_reduced_vector( tr.final_solution ) );
  }
}

// ===========================================================================
// MIGLIORAMENTO: Tabella riassuntiva raggruppata per line search
// ===========================================================================
void
print_summary_table_by_linesearch()
{
  // Raggruppa per line search
  map<string, vector<TestResult const *>> grouped;

  for ( auto const & r : global_test_results ) grouped[r.linesearch_name].push_back( &r );

  for ( auto const & [ls_name, results] : grouped )
  {
    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "\n\n"
      "╔══════════════════════════════════════════════════════════════"
      "════════════════════════════════════╗\n"
      "║  LINE SEARCH: {:<82} ║\n"
      "╠════════════════════════╤════════╤══════════╤════════════════╤"
      "═══════════════╤════════════════════╣\n"
      "║ Function               │ Dim    │ Iter     │ Final Value    "
      "│ ‖g_final‖     │ Status             ║\n"
      "╠════════════════════════╪════════╪══════════╪════════════════╪"
      "═══════════════╪════════════════════╣\n",
      ls_name );

    for ( auto const * rp : results )
    {
      auto const & r = *rp;

      string status_str = MINIMIZER::to_string( r.status );
      bool   converged  = r.status == Status::CONVERGED || r.status == Status::GRADIENT_TOO_SMALL;

      // Colori per lo status
      auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

      // Colori per la norma del gradiente
      auto grad_color = ( r.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                        : ( r.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                           : fmt::fg( fmt::color::red );

      string fname = r.problem_name;
      if ( fname.size() > 22 ) fname = fname.substr( 0, 19 ) + "...";

      fmt::print(
        "║ {:<22} │ {:>6} │ {:>8} │ {:<14.4e} │ ",
        fname,
        r.dimension,
        r.result_data.total_iterations,
        r.result_data.final_function_value );

      // Norma gradiente colorata
      fmt::print( grad_color, "{:<13.2e}", r.final_gradient_norm );
      fmt::print( " │ " );

      // Status colorato
      fmt::print( status_color, "{:<18}", status_str );
      fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
    }

    fmt::print(
      fmt::fg( fmt::color::light_blue ),
      "╚════════════════════════╧════════╧══════════╧════════════════╧"
      "═══════════════╧════════════════════╝\n" );
  }
}

// -------------------------------------------------------------------
// MAIN
// -------------------------------------------------------------------
int
main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║                 NEWTON Optimization Test Suite                 ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  // Test in ordine alfabetico
  AckleyN<Scalar, 15> ackley;
  test( ackley, "AckleyN15D" );

  Beale2D<Scalar> beale;
  test( beale, "Beale2D" );

  Booth2D<Scalar> booth;
  test( booth, "Booth2D" );

  BroydenTridiagonalN<Scalar, 12> broy;
  test( broy, "BroydenTridiagonal12D" );

  BrownAlmostLinearN<Scalar, 10> brown;
  test( brown, "BrownAlmostLinear10D" );

  ExtendedWoodN<Scalar, 16> woodN;
  test( woodN, "ExtendedWood16D" );

  FreudensteinRoth2D<Scalar> fr;
  test( fr, "FreudensteinRoth2D" );

  GriewankN<Scalar, 10> griewank;
  test( griewank, "GriewankN10D" );

  HelicalValley3D<Scalar> heli;
  test( heli, "HelicalValley3D" );

  Himmelblau2D<Scalar> himm;
  test( himm, "Himmelblau2D" );

  IllConditionedQuadraticN<Scalar, 20> illq;
  test( illq, "IllConditionedQuadratic20D" );

  LevyN<Scalar, 10> levy;
  test( levy, "LevyN10D" );

  Matyas2D<Scalar> matyas;
  test( matyas, "Matyas2D" );

  McCormick2D<Scalar> mccormick;
  test( mccormick, "McCormick2D" );

  MichalewiczN<Scalar, 10> michalewicz;
  test( michalewicz, "MichalewiczN10D" );

  NesterovChebyshevRosenbrock<Scalar, 128> nesterov;
  test( nesterov, "NesterovChebyshevRosenbrock128D" );

  PowellBadlyScaled2D<Scalar> pbs;
  test( pbs, "PowellBadlyScaled2D" );

  PowellSingularN<Scalar, 16> powellN;
  test( powellN, "PowellSingular16D" );

  RastriginN<Scalar, 15> rastrigin;
  test( rastrigin, "RastriginN15D" );

  Rosenbrock2D<Scalar> rosen;
  test( rosen, "Rosenbrock2D" );

  RosenbrockN<Scalar, 10> rosenN;
  test( rosenN, "Rosenbrock10D" );

  Schaffer2D<Scalar> schaffer;
  test( schaffer, "Schaffer2D" );

  SchwefelN<Scalar, 15> schwefel;
  test( schwefel, "SchwefelN15D" );

  ThreeHumpCamel2D<Scalar> camel;
  test( camel, "ThreeHumpCamel2D" );

  TrigonometricSumN<Scalar, 15> trig;
  test( trig, "TrigonometricSum15D" );

  // Stampa dei risultati
  // print_summary_table();
  print_summary_table_by_linesearch();
  print_line_search_statistics();

  return 0;
}
