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
#include <numeric>

#include "Utils_minimize_Newton.hh"

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif

using std::map;
using std::pair;
using std::string;
using std::vector;
using Scalar       = double;
using MINIMIZER    = Utils::Newton_minimizer<Scalar>;
using integer      = MINIMIZER::integer;
using Vector       = typename MINIMIZER::Vector;
using SparseMatrix = typename MINIMIZER::SparseMatrix;

using Status = MINIMIZER::Status;

struct TestResult
{
  string  problem_name;
  Vector  final_solution;
  size_t  dimension;
  Status  status;
  integer iterations;
  integer function_evals;
  integer hessian_evals;
  Scalar  final_f;
  Scalar  initial_f;
  Scalar  final_gradient_norm;
};

vector<TestResult> global_test_results;

#include "ND_func.cxx"

template <typename T, typename = void> struct has_hessian : std::false_type
{
};

template <typename T>
struct has_hessian<T, std::void_t<decltype( std::declval<T>().hessian( std::declval<Vector>() ) )>> : std::true_type
{
};

template <typename Problem> static void test( Problem & tp, string const & name )
{
  MINIMIZER::Options opts;
  opts.verbosity = 3;    // Reduced from 2 to avoid excessive output
  opts.max_iter  = 500;  // Reduced from 2 to avoid excessive output

  Vector x0 = tp.init();

  Vector final_solution = x0;

  auto cb = [&tp, &final_solution]( Vector const & x, Vector * g, SparseMatrix * H ) -> Scalar
  {
    final_solution = x;

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

  MINIMIZER m( opts );
  try
  {
    m.set_bounds( tp.lower(), tp.upper() );
  }
  catch ( ... )
  {
    // If no bounds, do nothing
  }

  // Esegue l'ottimizzazione
  m.minimize( x0, cb );

  // Ottiene i risultati dai metodi della classe
  TestResult tr;
  tr.problem_name        = name;
  tr.final_solution      = m.solution();
  tr.dimension           = tr.final_solution.size();
  tr.status              = m.status();
  tr.iterations          = m.iterations();
  tr.function_evals      = m.function_evals();
  tr.hessian_evals       = m.hessian_evals();
  tr.final_f             = m.final_f();
  tr.initial_f           = m.initial_f();
  tr.final_gradient_norm = m.final_grad_norm();

  global_test_results.push_back( tr );

  fmt::print(
    "{}: {} | iter = {} | f = {:.6e} | ‖g_proj‖ = {:.3e}\n",
    name,
    MINIMIZER::to_string( tr.status ),
    tr.iterations,
    tr.final_f,
    tr.final_gradient_norm );

  if ( opts.verbosity >= 2 ) { fmt::print( "Solution: {}\n\n", Utils::format_reduced_vector( tr.final_solution ) ); }
}

void print_summary_table()
{
  fmt::print(
    "\n\n"
    "╔══════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                      NEWTON SUMMARY RESULTS                                      ║\n"
    "╠════════════════════════╤════════╤══════════╤══════════════╤══════════════╤═══════════════════════╣\n"
    "║ Function               │ Dim    │ Iter     │ Final Value  │ ‖g_final‖    │ Status                ║\n"
    "╠════════════════════════╪════════╪══════════╪══════════════╪══════════════╪═══════════════════════╣"
    "\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = MINIMIZER::to_string( result.status );
    bool   converged  = result.status == Status::CONVERGED;

    auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

    auto grad_color = ( result.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( result.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                              : fmt::fg( fmt::color::red );

    string problem_name = result.problem_name;
    if ( problem_name.length() > 22 ) { problem_name = problem_name.substr( 0, 19 ) + "..."; }

    fmt::print(
      "║ {:<22} │ {:>6} │ {:>8} │ {:<12.4e} │ ",
      problem_name,
      result.dimension,
      result.iterations,
      result.final_f );

    fmt::print( grad_color, "{:<12.4e}", result.final_gradient_norm );
    fmt::print( " │ " );

    fmt::print( status_color, "{:<21}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    "╚════════════════════════╧════════╧══════════╧══════════════╧══════════════╧═══════════════════════╝"
    "\n" );

  MINIMIZER::integer total_tests     = global_test_results.size();
  MINIMIZER::integer converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.status == Status::CONVERGED; } );

  MINIMIZER::integer accumulated_iter{ 0 };
  MINIMIZER::integer accumulated_evals{ 0 };
  MINIMIZER::integer accumulated_hess_evals{ 0 };

  for ( auto const & r : global_test_results )
  {
    if ( r.status == Status::CONVERGED )
    {
      accumulated_iter += r.iterations;
      accumulated_evals += r.function_evals;
    }
    accumulated_hess_evals += r.hessian_evals;
  }

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print(
    "   • Converged: {} ({:.1f}%)\n",
    converged_tests,
    ( 100.0 * converged_tests / std::max<size_t>( total_tests, 1 ) ) );
  fmt::print( "   • Total iterations: {}\n", accumulated_iter );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
  fmt::print( "   • Total hessian evaluations: {}\n", accumulated_hess_evals );
}

int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║                 NEWTON Optimization Test Suite                 ║\n"
    "║               (Trust-Region with BFGS fallback)                ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
    "\n" );

#if 1
  integer k{ 0 };
  for ( auto [ptr, name] : NL_list ) test( *ptr, fmt::format( "N.{} {}", ++k, name ) );
#else
  auto ilist = { 6, 12, 16, 19, 26, 27, 28, 29, 37, 41 };
  for ( auto k : ilist )
  {
    auto & NL = NL_list[k - 1];
    test( *NL.first, fmt::format( "N.{} {}", k, NL.second ) );
  }
#endif

  print_summary_table();

  return 0;
}
