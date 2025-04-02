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
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

#include "Utils.hh"
#include "Utils_fmt.hh"

using std::cout;

static std::atomic<unsigned> accumulator;

class Counter {
  Utils::BinarySearch<int> bs;
public:
  Counter() {
    bool ok ;
    int * pdata = bs.search( std::this_thread::get_id(), ok );
    *pdata = 0;
  }

  void
  inc() const {
    bool ok;
    int * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    ++(*pdata);
  }

  int
  get() const {
    bool ok;
    int const * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    return *pdata;
  }

  void
  print() const {
    bool ok;
    int * pdata{ bs.search( std::this_thread::get_id(), ok ) };
    if ( !ok ) fmt::print("Counter::inc failed thread\n");
    fmt::print(
      "thread {}, counter = {}\n",
      std::this_thread::get_id(),
      *pdata
    );
  }
};

static
void
do_test( int const n, int const sz ) {
  Counter c;
  int const nn{ 1+((n*111)%sz) };
  //int nn = 40;
  for ( int i{0}; i < nn; ++i ) {
    //Utils::sleep_for_nanoseconds(1);
    int const mm{ 1+((i*11)%64) };
    for ( int j{0}; j < mm; ++j ) c.inc();
  }
  accumulator += c.get();
  //c.print();
}


template <class TP>
void
test_TP( int const NN, int nt, int sz ) {
  Utils::TicToc tm;

  accumulator = 0;
  double t_launch, t_wait;
  {
    TP pool(nt);

    tm.tic();
    for ( int i{0}; i < NN; ++i) pool.run( do_test, i, sz );
    tm.toc();
    t_launch = tm.elapsed_mus()/NN;

    tm.tic();
    pool.wait();
    tm.toc();
    t_wait = tm.elapsed_mus();

    tm.tic();
  }
  tm.toc();
  double t_delete{ tm.elapsed_mus() };

  fmt::print(
     "[{:30}] result {} [LAUNCH (AVE): {:12.8} mus, WAIT {:12.8} mus, DELETE {:12.8} mus] {:.8} mus\n",
     TP::Name(), accumulator.load(), t_launch, t_wait, t_delete, t_launch*NN+t_wait
  );
}

int
main( int const argc, char *argv[] ) {
  Utils::TicToc tm;

  int nt = 16;
  int sz = 200;

  if ( argc >= 2 ) nt = atoi( argv[1] );
  if ( argc >= 3 ) sz = atoi( argv[2] );

  fmt::print( "NT = {}\n", nt );

  for ( int NN : { 16, 100, 1000, 10000 } ) {
    accumulator = 0;
    tm.tic();
    for ( int i{0}; i < NN; ++i) do_test(i,sz);
    tm.toc();
    fmt::print(
      "[No Thread]   result {} [{:.6} mus, AVE = {:.6} mus]\n",
      accumulator.load(), tm.elapsed_mus(), tm.elapsed_mus()/NN
    );

    fmt::print("\n\nNN = {}\n\n", NN );
    test_TP<Utils::ThreadPool0>( NN, nt, sz );
    test_TP<Utils::ThreadPool1>( NN, nt, sz );
    test_TP<Utils::ThreadPool2>( NN, nt, sz );
    test_TP<Utils::ThreadPool3>( NN, nt, sz );
    test_TP<Utils::ThreadPool4>( NN, nt, sz );
    test_TP<Utils::ThreadPool5>( NN, nt, sz );
  }

  cout << "All done folks!\n\n";

  return 0;
}
