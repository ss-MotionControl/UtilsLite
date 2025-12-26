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

#include "Utils_AlgoBracket.hh"
#include "Utils_fmt.hh"

using namespace std;

using Utils::AlgoBracket;

#include "1D_fun.cxx"

constexpr int NA{ 8 };

static int ntest{ 0 };
static int nfuneval[NA]{ 0, 0, 0, 0, 0, 0, 0, 0 };

template <typename FUN>
void
do_solve( string_view name, real_type a, real_type b, FUN f )
{
  fmt::print( "\n#{:<3} {}\n", ntest, name );
  ++ntest;

  for ( const unsigned ialgo : { 0, 1, 2, 3, 4, 5, 6 } )
  {
    AlgoBracket<real_type> solver;
    solver.select( ialgo );
    real_type res  = solver.eval2( a, b, f );
    real_type fres = f( res );
    nfuneval[ialgo] += solver.num_fun_eval();
    fmt::print(
      "{:<15} f:{:<3} it:{:<3} {} x = {:12} f(x) = {:15}  b-a={}\n",
      solver.algo(),
      solver.num_fun_eval(),
      solver.used_iter(),
      solver.converged() ? "YES" : "NO ",
      fmt::format( "{:.6}", res ),
      fmt::format( "{:.6}", fres ),
      fmt::format( "{:.6}", solver.b() - solver.a() ) );
  }
}

/*
template <typename FUN>
void
do_solve2( real_type a, real_type b, real_type amin, real_type bmax, FUN f ) {
  AlgoBracket<real_type> solver;
  real_type res = solver.eval2( a, b, amin, bmax, f );
  ++ntest;
  nfuneval += solver.num_fun_eval();
  fmt::print(
    "#{:<3} iter = {:<3} #nfun = {:<3} {} x = {:12} f(x) = {:15} b-a={}\n",
    ntest, solver.used_iter(), solver.num_fun_eval(), solver.converged() ? "YES"
: "NO ", fmt::format("{:.6}",res), fmt::format("{:.3}",f(res)),
    fmt::format("{:.6}",solver.b()-solver.a())
  );
}
*/

int
main()
{
  std::vector<std::unique_ptr<fun1D>> f_list;

  build_1dfun_list( f_list );

  for ( const auto & f : f_list ) do_solve( f->info(), f->a0(), f->b0(), f->function() );

  // do_solve( "fun_penalty(x,0)",              -1.0, 1.0,    [] ( real_type x )
  // { return fun_penalty(x,0); } ); do_solve( "fun_penalty(x,-10)", -1.0, 1.0,
  // [] ( real_type x ) { return fun_penalty(x,-10); } ); do_solve2(
  // -1, 1.1498547501802843, -100, 100, [] ( real_type x ) { return
  // fun_penalty(x,-229.970950036057); } );

  for ( unsigned const ialgo : { 0, 1, 2, 3, 4, 5, 6 } )
  {
    AlgoBracket<real_type> solver;
    solver.select( ialgo );
    fmt::print( "{:15} = {}\n", solver.algo(), nfuneval[ialgo] );
  }

  cout << "\nAll Done Folks!\n";

  return 0;
}
