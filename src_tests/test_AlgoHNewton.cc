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
 |      Universita` degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils_AlgoHNewton.hh"
#include "Utils_fmt.hh"

using namespace std;

using Utils::AlgoHNewton;
using Utils::m_pi;

using real_type = double;

static inline real_type power2( real_type x ) { return x*x; }
static inline real_type power3( real_type x ) { return x*x*x; }
static inline real_type power4( real_type x ) { return power2(power2(x)); }
static inline real_type power5( real_type x ) { return power4(x)*x; }

static int ntest{0};
static int nfuneval{0};

class FUN1 : public Utils::AlgoHNewton_base_fun<real_type> {
public:
  real_type eval( real_type x ) const override {
    real_type res{0};
    for ( int i{1}; i <= 20; ++i ) {
      res += power2(2*i-5)/power3( x - i*i );
    }
    return -2*res;
  }
  
  real_type D( real_type x ) const override {
    real_type res{0};
    for ( int i{1}; i <= 20; ++i ) {
      res += power2(2*i-5)/power4( x - i*i );
    }
    return 6*res;
  }
};
FUN1 fun1;

class FUN2 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type a;
  real_type b;
public:
  FUN2() = delete;
  explicit FUN2( real_type _a, real_type _b ) : a(_a), b(_b) {}
  real_type eval( real_type x ) const override { return a*x*exp(b*x); }
  real_type D   ( real_type x ) const override { return a*exp(b*x)*(b*x+1); }
};

class FUN3 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
  real_type a;
public:
  FUN3() = delete;
  explicit FUN3( real_type _n, real_type _a ) : n(_n), a(_a) {}
  real_type eval( real_type x ) const override { return pow(x,n)-a; }
  real_type D   ( real_type x ) const override { return n*pow(x,n-1); }
};

class FUN4 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN4() = delete;
  explicit FUN4( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return x*exp(-n)-exp(-n*x)+0.5; }
  real_type D   ( real_type x ) const override { return exp(-n) + n*exp(-n*x); }
};

class FUN5 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN5() = delete;
  explicit FUN5( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return (1+power2(1-n))*x-power2(1-n*x); }
  real_type D   ( real_type x ) const override { return 2 + (1-2*x)*power2(n); }
};

class FUN6 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN6() = delete;
  explicit FUN6( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return power2(x)-pow(1-x,n); }
  real_type D   ( real_type x ) const override { return 2*x+n*pow(1-x,n-1); }
};

class FUN7 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN7() = delete;
  explicit FUN7( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return (1+power4(1-n))*x-power4(1-n*x); }
  real_type D   ( real_type x ) const override { return 1 + power4(1 - n) + 4*power3(1-n*x)*n ; }
};

class FUN8 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN8() = delete;
  explicit FUN8( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return (1+power4(1-n))*x-power4(1-n*x); }
  real_type D   ( real_type x ) const override { return (1 + (1 - x)*n)*exp(-n*x) + n*pow(x,n - 1); }
};

class FUN9 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN9() = delete;
  explicit FUN9( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return (n*x-1)/((n-1)*x); }
  real_type D   ( real_type x ) const override { return 1/((n - 1)*x*x); }
};

class FUN10 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN10() = delete;
  explicit FUN10( real_type _n ) : n(_n) {}
  real_type eval( real_type x ) const override { return pow(x,1/n)-pow(n,1/n); }
  real_type D   ( real_type x ) const override { return pow(x,(1-n)/n)/n; }
};

class FUN11 : public Utils::AlgoHNewton_base_fun<real_type> {
public:
  real_type
  eval( real_type x ) const override {
    if ( x == 0 ) return 0;
    return x*exp(-1/(x*x));
  }

  real_type
  D( real_type x ) const override {
    if ( x == 0 ) return 0;
    real_type x2{x*x};
    return exp(-1/x2)*(1+2/x2);
  }
};
FUN11 fun11;

class FUN12 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN12() = delete;
  explicit FUN12( real_type _n ) : n(_n) {}
  real_type
  eval( real_type x ) const override {
    if ( x < 0 ) return -n/20.0;
    return (n/20.0)*(x/1.5+sin(x)-1);
  }

  real_type
  D( real_type x ) const override {
    if ( x < 0 ) return 0;
    return n*cos(x)/20.0 + n/30.0;
  }
};

class FUN13 : public Utils::AlgoHNewton_base_fun<real_type> {
  real_type n;
public:
  FUN13() = delete;
  explicit FUN13( real_type _n ) : n(_n) {}
  real_type
  eval( real_type x ) const override {
    if ( x > 2e-3/(1+n) ) return exp(1)-1.859;
    if ( x < 0          ) return -0.859;
    return exp( (n+1)*0.5e3*x )-1.859;
  }

  real_type
  D( real_type x ) const override {
    if ( x > 2e-3/(1+n) ) return 0;
    if ( x < 0          ) return 0;
    return 500*(n+1)*exp(500*(n+1)*x);
  }
};

static
void
do_solve( real_type a, real_type b, Utils::AlgoHNewton_base_fun<real_type> const * f ) {
  AlgoHNewton<real_type> solver;
  real_type res  = solver.eval( a, b, f );
  real_type fres = f->eval(res);
  ++ntest;
  nfuneval += solver.num_fun_eval();
  fmt::print(
    "#{:<3} iter = {:<3} #nfun = {:<3} #nfun_D = {:<3} converged={} x = {:12} f(x) = {}\n",
    ntest, solver.used_iter(), solver.num_fun_eval(), solver.num_fun_D_eval(), solver.converged(),
    fmt::format("{:.6}",res),
    fmt::format("{:.3}",fres)
  );
}

int
main() {
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return x > 0 ? 1/(1-x) : x-1; }
      real_type D   ( real_type x ) const override { return x > 0 ? 1/power2(1-x) : 1; }
    };
    FUN f;
    do_solve( -1.0, 1.0, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return sin(x)-x/2; }
      real_type D   ( real_type x ) const override { return cos(x)-0.5; }
    };
    FUN f;
    do_solve( m_pi/2, m_pi, &f );
  }

  for ( int i = 1; i <= 10; ++i )
    do_solve( power2(i)+1e-9, power2(i+1)-1e-9, &fun1 );
  
  {
    FUN2 f1(-40,-1);
    FUN2 f2(-100,-2);
    FUN2 f3(-200,-3);
    do_solve( -9, 31, &f1 );
    do_solve( -9, 31, &f2 );
    do_solve( -9, 31, &f3 );
  }
  {
    FUN3 f1(4,0.2);
    FUN3 f2(6,0.2);
    FUN3 f3(8,0.2);
    FUN3 f4(10,0.2);
    FUN3 f5(12,0.2);
    FUN3 f6(8,1);
    FUN3 f7(10,1);
    FUN3 f8(12,1);
    FUN3 f9(14,1);
    do_solve( 0, 5, &f1 );
    do_solve( 0, 5, &f2 );
    do_solve( 0, 5, &f3 );
    do_solve( 0, 5, &f4 );
    do_solve( 0, 5, &f5 );
    do_solve( 0, 5, &f6 );
    do_solve( 0, 5, &f7 );
    do_solve( 0, 5, &f8 );
    do_solve( 0, 5, &f9 );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return sin(x)-0.5; }
      real_type D   ( real_type x ) const override { return cos(x); }
    };
    FUN f;
    do_solve( 0, 1.5, &f );
  }

  for ( int i = 1;  i <= 5;   ++i     ) { FUN4 f(i); do_solve( 0, 1, &f ); }
  for ( int i = 20; i <= 100; i += 20 ) { FUN4 f(i); do_solve( 0, 1, &f ); }

  {
    FUN5 f1(5);
    FUN5 f2(10);
    FUN5 f3(20);
    do_solve( 0, 1, &f1 );
    do_solve( 0, 1, &f2 );
    do_solve( 0, 1, &f3 );
  }
  {
    FUN6 f1(2);
    FUN6 f2(5);
    FUN6 f3(10);
    FUN6 f4(15);
    FUN6 f5(20);
    do_solve( 0, 1, &f1 );
    do_solve( 0, 1, &f2 );
    do_solve( 0, 1, &f3 );
    do_solve( 0, 1, &f4 );
    do_solve( 0, 1, &f5 );
  }
  {
    FUN7 f1(1);
    FUN7 f2(2);
    FUN7 f3(3);
    FUN7 f4(5);
    FUN7 f5(8);
    FUN7 f6(15);
    FUN7 f7(20);
    do_solve( 0, 1, &f1 );
    do_solve( 0, 1, &f2 );
    do_solve( 0, 1, &f3 );
    do_solve( 0, 1, &f4 );
    do_solve( 0, 1, &f5 );
    do_solve( 0, 1, &f6 );
    do_solve( 0, 1, &f7 );
  }
  {
    FUN8 f1(1);
    FUN8 f2(5);
    FUN8 f3(10);
    FUN8 f4(15);
    FUN8 f5(20);
    do_solve( 0, 1, &f1 );
    do_solve( 0, 1, &f2 );
    do_solve( 0, 1, &f3 );
    do_solve( 0, 1, &f4 );
    do_solve( 0, 1, &f5 );
  }
  {
    FUN9 f1(2);
    FUN9 f2(5);
    FUN9 f3(15);
    FUN9 f4(20);
    do_solve( 0.01, 1, &f1 );
    do_solve( 0.01, 1, &f2 );
    do_solve( 0.01, 1, &f3 );
    do_solve( 0.01, 1, &f4 );
  }
  for ( int i = 2;  i <= 9;  ++i    ) { FUN10 f(i); do_solve( 1, 100, &f ); }
  for ( int i = 11; i <= 33; i += 2 ) { FUN10 f(i); do_solve( 1, 100, &f ); }

  do_solve( -1, 4, &fun11 );

  for ( int i{1}; i <= 8;  ++i     ) { FUN12 f(i); do_solve( -1e4, m_pi/2, &f ); }
  for ( int i{0}; i <= 40; i += 10 ) { FUN12 f(i); do_solve( -1e4, m_pi/2, &f ); }

  for ( int i{20};  i <= 40;   ++i      ) { FUN13 f(i); do_solve( -1e4, 1e-4, &f ); }
  for ( int i{100}; i <= 1000; i += 100 ) { FUN13 f(i); do_solve( -1e4, 1e-4, &f ); }

  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return log(x); }
      real_type D   ( real_type x ) const override { return 1/x; }
    };
    FUN f;
    do_solve( 0.5,  5, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return (10-x)*exp(-10*x)-pow(x,10)+1; }
      real_type D   ( real_type x ) const override { return (10*x - 101)*exp(-10*x) - 10*pow(x,9); }
    };
    FUN f;
    do_solve( 0.5,  8, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return exp(sin(x))-x-1; }
      real_type D   ( real_type x ) const override { return cos(x)*exp(sin(x)) - 1; }
    };
    FUN f;
    do_solve( 1, 4, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return 11*pow(x,11)-1; }
      real_type D   ( real_type x ) const override { return 121*pow(x,10); }
    };
    FUN f;
    do_solve( 0.5,  1, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return 2*sin(x)-1; }
      real_type D   ( real_type x ) const override { return 2*cos(x); }
    };
    FUN f;
    do_solve( 0.1,  m_pi/3, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power2(x)+sin(x/10)-0.25; }
      real_type D   ( real_type x ) const override { return 2*x + cos(x/10)/10; }
    };
    FUN f;
    do_solve( 0, 1, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return (x-1)*exp(x); }
      real_type D   ( real_type x ) const override { return exp(x)*x; }
    };
    FUN f;
    do_solve( 0, 1.5, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return cos(x)-x; }
      real_type D   ( real_type x ) const override { return -sin(x)-1; }
    };
    FUN f;
    do_solve( 0, 1.7, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power3(x-1)-1; }
      real_type D   ( real_type x ) const override { return 3*power2(x-1); }
    };
    FUN f;
    do_solve( 1.5, 3, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return exp(x*x+7*x-30)-1; }
      real_type D   ( real_type x ) const override { return (2*x + 7)*exp((x + 10)*(x - 3)); }
    };
    FUN f;
    do_solve( 2.6, 3.5, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return atan(x)-1; }
      real_type D   ( real_type x ) const override { return 1/(x*x + 1); }
    };
    FUN f;
    do_solve( 1, 8, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return exp(x)-2*x-1; }
      real_type D   ( real_type x ) const override { return exp(x) - 2; }
    };
    FUN f;
    do_solve( 0.2, 3, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return exp(-x)-x-sin(x); }
      real_type D   ( real_type x ) const override { return -exp(-x) - 1 - cos(x); }
    };
    FUN f;
    do_solve( 0, 0.5, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power3(x)-1; }
      real_type D   ( real_type x ) const override { return 3*power2(x); }
    };
    FUN f;
    do_solve( 0.1,  1.5, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power2(x)-power2(sin(x))-1; }
      real_type D   ( real_type x ) const override { return 2*x - sin(2*x); }
    };
    FUN f;
    do_solve( -1, 2, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power3(x); }
      real_type D   ( real_type x ) const override { return 3*power2(x); }
    };
    FUN f;
    do_solve( -0.5, 1/3.0, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type eval( real_type x ) const override { return power5(x); }
      real_type D   ( real_type x ) const override { return 5*power4(x); }
    };
    FUN f;
    do_solve( -0.5, 1/3.0, &f );
  }
  {
    class FUN : public Utils::AlgoHNewton_base_fun<real_type> {
    public:
      real_type
      eval( real_type x ) const override {
        real_type x2{x*x};
        real_type x4{x2*x2};
        real_type x8{x4*x4};
        return tan(m_pi*(x8-0.5));
      }
      real_type
      D( real_type x ) const override {
        real_type x2{x*x};
        real_type x4{x2*x2};
        real_type x7{x4*x2*x};
        real_type x8{x4*x4};
        return 8*m_pi*x7*power2( 1/cos(m_pi*(x8 - 0.5)) );
      }
    };
    FUN f;
    do_solve( 0.0, 1.0, &f );
  }

  fmt::print( "nfuneval {}\n", nfuneval );

  cout << "\nAll Done Folks!\n";

  return 0;
}
