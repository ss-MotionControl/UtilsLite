/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2017                                                      |
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

#include "Utils.hh"
#include "Utils_fmt.hh"

using std::cout;

static void do_test( int, int ) { return; }

template <class TP>
void
test_TP( int NN, int nt, int sz ) {
  Utils::TicToc tm;

  double t_launch, t_wait, t_delete;
  {
    TP pool(nt);

    tm.tic();
    for ( int i{0}; i < NN; ++i ) pool.run( do_test, i, sz );
    tm.toc();
    t_launch = tm.elapsed_mus()/NN;

    tm.tic();
    pool.wait();
    tm.toc();
    t_wait = tm.elapsed_mus();

    tm.tic();
  }
  tm.toc();
  t_delete = tm.elapsed_mus();

  fmt::print(
    "[{:30}] [LAUNCH: {:12.8}mus, WAIT {:12.8}mus, DELETE {:12.8}mus] {:.8}mus\n",
    TP::Name(), t_launch, t_wait, t_delete, t_launch+t_wait
  );
}

int
main( int argc, char *argv[] ) {
  Utils::TicToc tm;

  int nt = 16;
  int sz = 200;

  if ( argc >= 2 ) nt = atoi( argv[1] );
  if ( argc >= 3 ) sz = atoi( argv[2] );

  fmt::print( "NT = {}\n", nt );

  for ( int NN : { 16, 100, 1000, 10000 } ) {
    fmt::print("\n\nNN = {}\n\n", NN );
    test_TP<Utils::ThreadPool0>( NN, nt, sz );
    test_TP<Utils::ThreadPool1>( NN, nt, sz );
    test_TP<Utils::ThreadPool2>( NN, nt, sz );
    test_TP<Utils::ThreadPool3>( NN, nt, sz );
    test_TP<Utils::ThreadPool4>( NN, nt, sz );
    test_TP<Utils::ThreadPool5>( NN, nt, sz );
  }
  fmt::print("All done folks!\n\n");

  #if 0

  fmt::print("ThreadPool1\n");
  Utils::ThreadPool1 TP1(16); // 0%
  Utils::sleep_for_seconds(4);

  fmt::print("ThreadPool2\n");
  Utils::ThreadPool2 TP2(16); // 0%
  Utils::sleep_for_seconds(4);

  fmt::print("ThreadPool3\n");
  Utils::ThreadPool3 TP3(16); // 100%
  Utils::sleep_for_seconds(4);

  fmt::print("ThreadPool4\n");
  Utils::ThreadPool4 TP4(16); // 100%
  Utils::sleep_for_seconds(4);

  fmt::print("ThreadPool5\n");
  Utils::ThreadPool5 TP5(16); // 0%
  Utils::sleep_for_seconds(4);
  #endif

  cout << "All done folks!\n\n";

  return 0;
}
