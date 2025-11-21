/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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

#include "Utils_NelderMead.hh"
#include "Utils_fmt.hh"

#include <cmath>

using namespace Utils;
using namespace std;

using real_type = double;
using Vector = Utils::NelderMead_minimizer<real_type>::Vector;

static inline real_type power2( real_type const x ) { return x*x; }
//static inline real_type power3( real_type x ) { return x*x*x; }
//static inline real_type power4( real_type x ) { return power2(power2(x)); }
//static inline real_type power5( real_type x ) { return power4(x)*x; }

#if 0
static
real_type
fun1( Vector const & X ) {
  real_type x = X(0);
  real_type y = X(1);
  if ( x                              > -1      ) return Utils::Inf<real_type>();
  if ( x                              < -17.001 ) return Utils::Inf<real_type>();
  if ( y                              > -1      ) return Utils::Inf<real_type>();
  if ( y                              < -x/3-28 ) return Utils::Inf<real_type>();
  if ( power2(y+20)-3*x               < 51      ) return Utils::Inf<real_type>();
  if ( abs(x+14.5)+power2(y+15)       < 3       ) return Utils::Inf<real_type>();
  if ( power2(x+16)+pow(abs(y+8),1.5) < 20      ) return Utils::Inf<real_type>();
  if ( power2(x+9.2)+abs(y+12)        < 7       ) return Utils::Inf<real_type>();
  if ( power2(x+6)+power2(y+15)       < 29.8    ) return Utils::Inf<real_type>();
  if ( power2(x+6)+pow(abs(y+1),1.5)  < 15      ) return Utils::Inf<real_type>();
  return pow(abs(y-x),2.07) + pow(abs(x*y),1.07);
}
#endif

#if 0
static
real_type
fun2( Vector const & X ) {
  real_type x = X(0);
  real_type y = X(1);
  if ( x > 100    ) return Utils::Inf<real_type>();
  if ( x < 0      ) return Utils::Inf<real_type>();
  if ( y > 101.01 ) return Utils::Inf<real_type>();
  if ( y < 0      ) return Utils::Inf<real_type>();
  {
    real_type const v[] = {5,19,33,47,61,75,89};
    for ( int i = 0; i < 7; ++i ) {
      if ( abs(x)+pow( abs(y-v[i]), 3.5 ) < 99.9 )
        return Utils::Inf<real_type>();
    }
  }
  {
    real_type const v[] = {12,26,40,54,68,82,96};
    for ( int i = 0; i < 7; ++i ) {
      if ( abs(x-100)+pow( abs(y-v[i]), 3 ) < 99.9 )
        return Utils::Inf<real_type>();
    }
  }
  return abs(x-100)/100 + abs(y-101);
}
#endif

static
real_type
fun3( Vector const & X ) {
  real_type const x { X(0) };
  real_type const y { X(1) };
  return 100*power2(y-x*x)+power2(1-x);
}

static
real_type
fun4( Vector const & X ) {
  real_type const x { X(0) };
  real_type const y { X(1) };
  real_type const e { 1e-8 };
  real_type const w { 1-x*x-y*y };
  if ( w > 0 ) return x+y+e/w;
  return Utils::Inf<real_type>();
}

template <typename FUN>
void
do_solve( FUN f, Vector const & X0, real_type const delta ) {
  NelderMead_minimizer<real_type>::Options opts;
  opts.verbose = true;
  opts.initial_step = delta; // Usa delta come passo iniziale
  NelderMead_minimizer<real_type> solver(opts);
  auto result = solver.minimize( X0, f );
  
  fmt::print( "Solution found:\n" );
  fmt::print( "  Status: {}\n", NelderMead_minimizer<real_type>::status_to_string(result.status) );
  fmt::print( "  Solution: [{:.8f}, {:.8f}]\n", result.solution(0), result.solution(1) );
  fmt::print( "  Function value: {:.8f}\n", result.final_function_value );
  fmt::print( "  Iterations: {}\n", result.iterations );
  fmt::print( "  Function evaluations: {}\n", result.function_evaluations );
  fmt::print( "  Simplex diameter: {:.2e}\n", result.simplex_diameter );
  fmt::print( "  Simplex volume: {:.2e}\n", result.simplex_volume );
  fmt::print( "\n" );
}

int
main() {
  #if 0
  {
    real_type X0[2]{-1.1,-27.0};
    real_type delta = 1;
    do_solve( fun1, X0, delta );
  }
  #endif
  #if 0
  {
    real_type X0[2]{1,1};
    real_type delta = 1;
    do_solve( fun2, X0, delta );
  }
  #endif
  #if 1
  {
    Vector X0(2);
    X0 << -1, 1;
    real_type delta = 0.1;
    do_solve( fun3, X0, delta );
  }
  #endif
  #if 1
  {
    Vector X0(2);
    X0 << 0, 0;
    real_type delta = 0.01;
    do_solve( fun4, X0, delta );
  }
  #endif

  cout << "\nAll Done Folks!\n";

  return 0;
}
