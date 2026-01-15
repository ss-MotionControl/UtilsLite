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

#include "Utils_minimize_SQP.hh"

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
using MINIMIZER    = Utils::SQP_minimizer<Scalar>;
using integer      = MINIMIZER::integer;
using Vector       = typename MINIMIZER::Vector;
using SparseMatrix = typename MINIMIZER::SparseMatrix;

using Status = MINIMIZER::Status;

struct TestResult
{
  string  problem_name;
  Status  status;
  integer iterations;
  integer function_evals;
  integer hessian_evals;
  integer qp_solves;
  integer line_searches;
  Scalar  final_f;
  Scalar  initial_f;
  Scalar  final_gradient_norm;
  Scalar  final_kkt_norm;
  Vector  final_solution;
  Vector  multipliers;
  integer active_constraints;
  integer dimension;
};

vector<TestResult> global_test_results;

#include "ND_func.cxx"

template <typename Problem> static void test( Problem & tp, string const & name )
{
  fmt::print( "\n\n{:─^{}}\n\n", name, 80 );

  MINIMIZER::Options opts;
  opts.max_iter = 20000;
  // opts.g_tol           = 1e-9;
  // opts.f_tol           = 1e-10;
  // opts.x_tol           = 1e-8;
  // opts.kkt_tol         = 1e-9;
  // opts.alpha_init      = 1.0;
  // opts.rho             = 0.5;
  // opts.c1              = 1e-4;
  // opts.max_line_search = 30;
  // opts.lambda_init     = 1e-8;
  // opts.active_tol      = 1e-6;
  opts.verbosity = 3;  // Reduced for cleaner output

  Vector x0 = tp.init();

  Vector final_solution = x0;

  auto cb = [&tp, &final_solution]( Vector const & x, Vector * g, SparseMatrix * H ) -> Scalar
  {
    final_solution = x;
    if ( g ) *g = tp.gradient( x );
    if ( H ) *H = tp.hessian( x );
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

  m.minimize( x0, cb );

  TestResult tr;
  tr.problem_name        = name;
  tr.status              = m.status();
  tr.iterations          = m.iterations();
  tr.function_evals      = m.function_evals();
  tr.hessian_evals       = m.hessian_evals();
  tr.qp_solves           = m.qp_solves();
  tr.line_searches       = m.line_searches();
  tr.final_f             = m.final_f();
  tr.initial_f           = m.initial_f();
  tr.final_gradient_norm = m.final_grad_norm();
  tr.final_kkt_norm      = m.final_kkt_norm();
  tr.final_solution      = m.solution();
  tr.multipliers         = m.multipliers();
  tr.active_constraints  = m.active_constraints();
  tr.dimension           = tr.final_solution.size();

  global_test_results.push_back( tr );

  fmt::print(
    "{}: {} | iter = {} | f = {:.6e} | ‖g‖ = {:.3e} | KKT = {:.3e} | Active = {}\n",
    name,
    MINIMIZER::to_string( tr.status ),
    tr.iterations,
    tr.final_f,
    tr.final_gradient_norm,
    tr.final_kkt_norm,
    tr.active_constraints );

  if ( opts.verbosity >= 2 ) { fmt::print( "Solution: {}\n\n", Utils::format_reduced_vector( tr.final_solution ) ); }
}

void print_summary_table()
{
  fmt::print(
    "\n\n"
    "╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
    "║                                         SQP SUMMARY RESULTS                                            ║\n"
    "╠═══════════════════════╤═══════╤═════════╤═══════════════╤════════════╤════════════╤════════════════════╣\n"
    "║ Function              │ Dim   │ Iter    │ Final Value   │ ‖g_final‖  │ KKT        │ Status             ║\n"
    "╠═══════════════════════╪═══════╪═════════╪═══════════════╪════════════╪════════════╪════════════════════╣\n" );

  for ( auto const & result : global_test_results )
  {
    string status_str = MINIMIZER::to_string( result.status );
    bool   converged  = result.status == Status::CONVERGED || result.status == Status::GRADIENT_TOO_SMALL;

    auto status_color = converged ? fmt::fg( fmt::color::green ) : fmt::fg( fmt::color::red );

    auto grad_color = ( result.final_gradient_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                      : ( result.final_gradient_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                              : fmt::fg( fmt::color::red );

    auto kkt_color = ( result.final_kkt_norm < 1e-8 )   ? fmt::fg( fmt::color::green )
                     : ( result.final_kkt_norm < 1e-6 ) ? fmt::fg( fmt::color::yellow )
                                                        : fmt::fg( fmt::color::red );

    string problem_name = result.problem_name;
    if ( problem_name.length() > 21 ) { problem_name = problem_name.substr( 0, 18 ) + "..."; }

    fmt::print(
      "║ {:<21} │ {:>5} │ {:>7} │ {:<13.4g} │ ",
      problem_name,
      result.dimension,
      result.iterations,
      result.final_f );

    fmt::print( grad_color, "{:<10.2g}", result.final_gradient_norm );
    fmt::print( " │ " );

    fmt::print( kkt_color, "{:<10.2g}", result.final_kkt_norm );
    fmt::print( " │ " );

    fmt::print( status_color, "{:<18}", status_str );
    fmt::print( fmt::fg( fmt::color::light_blue ), " ║\n" );
  }

  fmt::print(
    "╚═══════════════════════╧═══════╧═════════╧═══════════════╧════════════╧════════════╧════════════════════╝\n" );

  integer total_tests     = global_test_results.size();
  integer converged_tests = std::count_if(
    global_test_results.begin(),
    global_test_results.end(),
    []( const TestResult & r ) { return r.status == Status::CONVERGED || r.status == Status::GRADIENT_TOO_SMALL; } );

  integer accumulated_iter{ 0 };
  integer accumulated_evals{ 0 };
  integer accumulated_hess_evals{ 0 };
  integer accumulated_qp_solves{ 0 };
  integer accumulated_line_searches{ 0 };
  integer accumulated_active{ 0 };

  for ( auto const & r : global_test_results )
  {
    if ( r.status == Status::CONVERGED || r.status == Status::GRADIENT_TOO_SMALL )
    {
      accumulated_iter += r.iterations;
      accumulated_evals += r.function_evals;
    }
    accumulated_hess_evals += r.hessian_evals;
    accumulated_qp_solves += r.qp_solves;
    accumulated_line_searches += r.line_searches;
    accumulated_active += r.active_constraints;
  }

  auto perc = 100.0 * converged_tests / std::max<integer>( total_tests, 1 );

  fmt::print( fmt::fg( fmt::color::light_blue ), "\n📊 Global Statistics:\n" );
  fmt::print( "   • Total problems: {}\n", total_tests );
  fmt::print( "   • Converged: {} ({:.1f}%)\n", converged_tests, perc );
  fmt::print( "   • Total iterations: {}\n", accumulated_iter );
  fmt::print( "   • Total function evaluations: {}\n", accumulated_evals );
  fmt::print( "   • Total hessian evaluations: {}\n", accumulated_hess_evals );
  fmt::print( "   • Total QP solves: {}\n", accumulated_qp_solves );
  fmt::print( "   • Total line searches: {}\n", accumulated_line_searches );
  fmt::print( "   • Total active constraints: {}\n", accumulated_active );
}

int main()
{
  fmt::print(
    fmt::fg( fmt::color::light_blue ),
    "╔═══════════════════════════════════════════════════════════════╗\n"
    "║               SQP Optimization Test Suite                     ║\n"
    "║            (Sequential Quadratic Programming)                 ║\n"
    "╚═══════════════════════════════════════════════════════════════╝\n"
    "\n" );

  integer k = 0;
  for ( auto [ptr, name] : NL_list ) test( *ptr, fmt::format( " n.{} {} ", ++k, name ) );
  // auto [ptr, name] = NL_list[12];
  // test( *ptr, name );

  print_summary_table();

  return 0;
}
