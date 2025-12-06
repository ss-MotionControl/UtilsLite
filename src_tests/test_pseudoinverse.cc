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

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>
#include <tuple>

#include "Utils_fmt.hh"
#include "Utils_pseudoinverse.hh"

using namespace Eigen;
using std::pair;
using std::string;
using std::vector;

//==============================================================================
static vector<double>         lambda_values = { 0.0, 1e-2, 1e-6, 1e-10 };
static vector<pair<int, int>> sizes         = {
  { 50, 50 }, { 100, 50 }, { 500, 50 }, { 100, 100 }, { 500, 100 }, { 3000, 200 }
};

//==============================================================================
// Funzione per generare una matrice sparsa casuale
template <typename Scalar>
Eigen::SparseMatrix<Scalar>
generate_random_sparse_matrix( int m, int n, double density = 0.3 )
{
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  SparseMatrix mat( m, n );

  size_t nnz = static_cast<size_t>( density * m * n );

  static std::random_device              rd;
  static std::mt19937                    gen( rd() );
  std::uniform_int_distribution<>        row_dist( 0, m - 1 );
  std::uniform_int_distribution<>        col_dist( 0, n - 1 );
  std::uniform_real_distribution<Scalar> val_dist( -1.0, 1.0 );

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve( nnz );

  std::set<std::pair<int, int>> positions;

  // Nota: std::set è lento per matrici grandi, ma ok per test.
  // Per performance migliori usare direttamente triplette con duplicati e sumup di Eigen.
  while ( triplets.size() < nnz )
  {
    int i = row_dist( gen );
    int j = col_dist( gen );
    if ( positions.find( { i, j } ) == positions.end() )
    {
      positions.insert( { i, j } );
      triplets.emplace_back( i, j, val_dist( gen ) );
    }
  }

  mat.setFromTriplets( triplets.begin(), triplets.end() );
  mat.makeCompressed();

  return mat;
}

//==============================================================================
// Genera un vettore casuale con elementi in [min_val, max_val]
template <typename Scalar>
Matrix<Scalar, Dynamic, 1>
generate_random_vector( int n, Scalar min_val = 0.1, Scalar max_val = 10.0 )
{
  static std::random_device              rd;
  static std::mt19937                    gen( rd() );
  std::uniform_real_distribution<Scalar> dist( min_val, max_val );

  Matrix<Scalar, Dynamic, 1> vec( n );
  for ( int i = 0; i < n; ++i ) { vec( i ) = dist( gen ); }

  return vec;
}

//==============================================================================
// FORMATTER UTILS
//==============================================================================
inline void
header( std::string const & msg )
{
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold,
              "\n"
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              "━━━━━━━━━━━━━━━┓\n" );
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "┃ {:^76} ┃\n", msg );
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold,
              "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              "━━━━━━━━━━━━━━━┛\n" );
}

inline std::string
sci( double v )
{
  return fmt::format( "{:>12.3e}", v );
}
inline std::string
ms_format( double v )
{
  return fmt::format( "    {}", fmt::format( "{:.2f} ms", v ) );
}
inline std::string
ratio_format( double v )
{
  return fmt::format( "    {}", fmt::format( "{:.2f}x", v ) );
}
inline std::string
small_ms_format( double v )
{
  return fmt::format( "    {}", fmt::format( "{:.4f} ms", v ) );
}

//==============================================================================
// RUN TEST STANDARD (D = I)
//==============================================================================
static void
run_test_standard( int m, int n, double lambda )
{
  Utils::TicToc tm;

  header( fmt::format( "STANDARD TEST (D=I): m={}, n={}, λ={}", m, n, lambda ) );

  // Genera matrice sparsa casuale
  SparseMatrix<double> AS = generate_random_sparse_matrix<double>( m, n, 0.3 );
  // Converti in densa per i solver dense
  MatrixXd AD = MatrixXd( AS );

  VectorXd b = VectorXd::Random( m );
  VectorXd c = VectorXd::Random( n );

  //--------------------------------------------------------------------------
  // Build TikhonovSolver (QR) - DENSE
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver<double> qr_solver( AD, lambda );
  tm.toc();
  double t_build_qr = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build TikhonovSolver2 (KKT dense)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver2<double> kkt_dense_solver( AD, lambda );
  tm.toc();
  double t_build_kkt_dense = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver (sparse QR)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver<double> sp_solver( AS, lambda );
  tm.toc();
  double t_build_sp = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver2 (sparse KKT)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver2<double> sp2_solver( AS, lambda );
  tm.toc();
  double t_build_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Accuracy - Solve min ||A x - b||^2 + λ||x||^2
  //--------------------------------------------------------------------------
  VectorXd x_qr = qr_solver.solve( b );
  double   r_qr = ( AD * x_qr - b ).norm();

  VectorXd x_kkt_dense = kkt_dense_solver.solve( b );
  double   r_kkt_dense = ( AD * x_kkt_dense - b ).norm();

  VectorXd x_sp = sp_solver.solve( b );
  double   r_sp = ( AS * x_sp - b ).norm();

  VectorXd x_sp2 = sp2_solver.solve( b );
  double   r_sp2 = ( AS * x_sp2 - b ).norm();

  double dx_qr_kkt  = ( x_qr - x_kkt_dense ).norm();
  double dx_qr_sp   = ( x_qr - x_sp ).norm();
  double dx_qr_sp2  = ( x_qr - x_sp2 ).norm();
  double dx_kkt_sp  = ( x_kkt_dense - x_sp ).norm();
  double dx_kkt_sp2 = ( x_kkt_dense - x_sp2 ).norm();
  double dx_sp_sp2  = ( x_sp - x_sp2 ).norm();

  //--------------------------------------------------------------------------
  // Accuracy - Transpose solve: y = A * (A^T A + λ I)^{-1} * c
  //--------------------------------------------------------------------------
  VectorXd y_qr        = qr_solver.solve_transpose( c );
  VectorXd y_kkt_dense = kkt_dense_solver.solve_transpose( c );
  VectorXd y_sp        = sp_solver.solve_transpose( c );
  VectorXd y_sp2       = sp2_solver.solve_transpose( c );

  double dy_qr_kkt  = ( y_qr - y_kkt_dense ).norm();
  double dy_qr_sp   = ( y_qr - y_sp ).norm();
  double dy_qr_sp2  = ( y_qr - y_sp2 ).norm();
  double dy_kkt_sp  = ( y_kkt_dense - y_sp ).norm();
  double dy_kkt_sp2 = ( y_kkt_dense - y_sp2 ).norm();
  double dy_sp_sp2  = ( y_sp - y_sp2 ).norm();

  //--------------------------------------------------------------------------
  // Objective function values
  //--------------------------------------------------------------------------
  auto compute_objective = [&]( const MatrixXd & A_mat, const VectorXd & x, double lam ) -> double
  { return ( A_mat * x - b ).squaredNorm() + lam * x.squaredNorm(); };

  double obj_qr  = compute_objective( AD, x_qr, lambda );
  double obj_kkt = compute_objective( AD, x_kkt_dense, lambda );
  double obj_sp  = compute_objective( AD, x_sp, lambda );
  double obj_sp2 = compute_objective( AD, x_sp2, lambda );

  //--------------------------------------------------------------------------
  // Speed test - Multiple solves
  //--------------------------------------------------------------------------
  const int NTEST   = 100;
  double    t100_qr = 0.0, t100_kkt_dense = 0.0, t100_sp = 0.0, t100_sp2 = 0.0;

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = qr_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_qr = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = kkt_dense_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_kkt_dense = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = sp_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_sp = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = sp2_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // OUTPUT TABLE
  //--------------------------------------------------------------------------

  // Accuracy table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " ACCURACY METRICS " );
  fmt::print( "{:<54} {:<25}\n", "    Metric", "    Value" );
  fmt::print( "{:─<54} {:─<25}\n", "─", "─" );

  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_qr − b‖ (QR Solver)", sci( r_qr ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_kkt − b‖ (KKT dense)", sci( r_kkt_dense ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp − b‖ (SP QR Solver)", sci( r_sp ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp2 − b‖ (SP KKT Solver)", sci( r_sp2 ) );

  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_kkt‖ (QR vs KKT dense)", sci( dx_qr_kkt ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp‖ (QR vs SP QR)", sci( dx_qr_sp ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp2‖ (QR vs SP KKT)", sci( dx_qr_sp2 ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_kkt − x_sp‖ (KKT dense vs SP QR)",
              sci( dx_kkt_sp ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_kkt − x_sp2‖ (KKT dense vs SP KKT)",
              sci( dx_kkt_sp2 ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_sp − x_sp2‖ (SP QR vs SP KKT)", sci( dx_sp_sp2 ) );

  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_qr − y_kkt‖ (Transpose QR vs KKT)",
              sci( dy_qr_kkt ) );
  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_qr − y_sp‖ (Transpose QR vs SP QR)",
              sci( dy_qr_sp ) );
  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_qr − y_sp2‖ (Transpose QR vs SP KKT)",
              sci( dy_qr_sp2 ) );
  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_kkt − y_sp‖ (Transpose KKT vs SP QR)",
              sci( dy_kkt_sp ) );
  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_kkt − y_sp2‖ (Transpose KKT vs SP KKT)",
              sci( dy_kkt_sp2 ) );
  fmt::print( fg( fmt::color::orange ), "{:<54} {:<25}\n", "    ‖y_sp − y_sp2‖ (Transpose SP QR vs SP KKT)",
              sci( dy_sp_sp2 ) );

  // Objective function values
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " OBJECTIVE FUNCTION VALUES " );
  fmt::print( "{:<30} {:<25}\n", "    Solver", "    f(x) = ||Ax-b||² + λ||x||²" );
  fmt::print( "{:─<30} {:─<25}\n", "─", "─" );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    QR Solver", sci( obj_qr ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    KKT Solver (dense)", sci( obj_kkt ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP QR Solver", sci( obj_sp ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP KKT Solver", sci( obj_sp2 ) );

  // Build times table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " BUILD TIMES " );
  fmt::print( "{:<26} {:<26} {:<26}\n", "    Method", "    Time", "    Relative" );
  fmt::print( "{:─<26} {:─<26} {:─<26}\n", "─", "─", "─" );

  std::vector<std::pair<double, std::string>> build_times = { { t_build_qr, "QR Solver (dense)" },
                                                              { t_build_kkt_dense, "KKT Solver (dense)" },
                                                              { t_build_sp, "SP QR Solver (sparse)" },
                                                              { t_build_sp2, "SP KKT Solver (sparse)" } };

  double min_build = std::numeric_limits<double>::infinity();
  for ( const auto & bt : build_times )
  {
    if ( bt.first > 0 && bt.first < min_build ) min_build = bt.first;
  }

  for ( const auto & bt : build_times )
  {
    fmt::print( fg( fmt::color::light_green ), "{:<26} {:<26} {:<26}\n", "    " + bt.second,
                bt.first > 0 ? ms_format( bt.first ) : "    FAILED",
                bt.first > 0 ? ratio_format( bt.first / min_build ) : "    N/A" );
  }

  // Solve times table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " SOLVE TIMES (100 iterations) " );
  fmt::print( "{:<29} {:<16} {:<16} {:<16}\n", "    Method", "    Total Time", "    Time/Solve", "    Speed Ratio" );
  fmt::print( "{:─<29} {:─<16} {:─<16} {:─<16}\n", "─", "─", "─", "─" );

  std::vector<std::tuple<double, std::string>> solve_times = { { t100_qr, "QR Solver (dense)" },
                                                               { t100_kkt_dense, "KKT Solver (dense)" },
                                                               { t100_sp, "SP QR Solver (sparse)" },
                                                               { t100_sp2, "SP KKT Solver (sparse)" } };

  double min_solve = std::numeric_limits<double>::infinity();
  for ( const auto & st : solve_times )
  {
    if ( std::get<0>( st ) < min_solve ) min_solve = std::get<0>( st );
  }

  for ( const auto & st : solve_times )
  {
    double              time = std::get<0>( st );
    const std::string & name = std::get<1>( st );

    fmt::print( fg( fmt::color::light_green ), "{:<29} {:<16} {:<16} {:<16}\n", "    " + name, ms_format( time ),
                small_ms_format( time / NTEST ), ratio_format( time / min_solve ) );
  }
}

//==============================================================================
// RUN TEST WITH DIAGONAL PENALTY MATRIX D
//==============================================================================
static void
run_test_with_diagonal( int m, int n, double lambda )
{
  Utils::TicToc tm;

  header( fmt::format( "TEST WITH DIAGONAL D: m={}, n={}, λ={}", m, n, lambda ) );

  // Genera matrice sparsa casuale
  SparseMatrix<double> AS = generate_random_sparse_matrix<double>( m, n, 0.3 );
  MatrixXd             AD = MatrixXd( AS );

  // Genera matrice diagonale D casuale
  VectorXd D     = generate_random_vector<double>( n, 0.1, 10.0 );
  VectorXd D_inv = D.cwiseInverse();

  // 1. APPROCCIO TRASFORMAZIONE (per i solver QR che non supportano D)
  // Problema: min ||A x - b||^2 + lambda ||D x||^2
  // Sostituzione: y = D x  =>  x = D^-1 y
  // Diventa: min ||(A D^-1) y - b||^2 + lambda ||y||^2
  MatrixXd             AD_tilde = AD * D_inv.asDiagonal();
  SparseMatrix<double> AS_tilde = AS * D_inv.asDiagonal();

  // 2. APPROCCIO NATIVO (per i solver KKT che supportano D)
  // Il solver KKT risolve (A^T A + Diag(reg)) x = A^T b
  // Il problema minimizza ||Ax - b||^2 + ||D_weight x||^2
  // => Equazioni normali: (A^T A + D_weight^T D_weight) x = A^T b
  // Quindi il vettore di regolarizzazione da passare è lambda * D.^2
  VectorXd regularization_vec = lambda * D.array().square();

  VectorXd b = VectorXd::Random( m );
  // c non usato per test trasposta nel caso diagonale complesso

  //--------------------------------------------------------------------------
  // Build TikhonovSolver (QR) - DENSE con A_tilde (MANUALE)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver<double> qr_solver( AD_tilde, lambda );
  VectorXd                      y_qr = qr_solver.solve( b );
  VectorXd                      x_qr = D_inv.asDiagonal() * y_qr;  // Trasformazione inversa
  tm.toc();
  double t_build_qr = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build TikhonovSolver2 (KKT dense) - DENSE (NATIVO)
  //--------------------------------------------------------------------------
  tm.tic();
  // Passiamo la matrice originale AD e il vettore di regolarizzazione calcolato
  Utils::TikhonovSolver2<double> kkt_dense_solver( AD, regularization_vec );
  VectorXd                       x_kkt_dense = kkt_dense_solver.solve( b );
  tm.toc();
  double t_build_kkt_dense = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver (sparse QR) - SPARSE con A_tilde (MANUALE)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver<double> sp_solver( AS_tilde, lambda );
  VectorXd                         y_sp = sp_solver.solve( b );
  VectorXd                         x_sp = D_inv.asDiagonal() * y_sp;
  tm.toc();
  double t_build_sp = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver2 (sparse KKT) - SPARSE (NATIVO)
  //--------------------------------------------------------------------------
  tm.tic();
  // Passiamo la matrice originale AS e il vettore di regolarizzazione calcolato
  Utils::SP_TikhonovSolver2<double> sp2_solver( AS, regularization_vec );
  VectorXd                          x_sp2 = sp2_solver.solve( b );
  tm.toc();
  double t_build_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Accuracy - Solve min ||A x - b||^2 + λ||D x||^2
  //--------------------------------------------------------------------------
  auto compute_residual = [&]( const MatrixXd & A_mat, const VectorXd & x ) -> double
  { return ( A_mat * x - b ).norm(); };

  double r_qr        = compute_residual( AD, x_qr );
  double r_kkt_dense = compute_residual( AD, x_kkt_dense );
  double r_sp        = compute_residual( AD, x_sp );
  double r_sp2       = compute_residual( AD, x_sp2 );

  double dx_qr_kkt  = ( x_qr - x_kkt_dense ).norm();
  double dx_qr_sp   = ( x_qr - x_sp ).norm();
  double dx_qr_sp2  = ( x_qr - x_sp2 ).norm();
  double dx_kkt_sp  = ( x_kkt_dense - x_sp ).norm();
  double dx_kkt_sp2 = ( x_kkt_dense - x_sp2 ).norm();
  double dx_sp_sp2  = ( x_sp - x_sp2 ).norm();

  //--------------------------------------------------------------------------
  // Objective function values con D
  //--------------------------------------------------------------------------
  auto compute_objective_D = [&]( const MatrixXd & A_mat, const VectorXd & x, double lam ) -> double
  { return ( A_mat * x - b ).squaredNorm() + lam * ( D.asDiagonal() * x ).squaredNorm(); };

  double obj_qr  = compute_objective_D( AD, x_qr, lambda );
  double obj_kkt = compute_objective_D( AD, x_kkt_dense, lambda );
  double obj_sp  = compute_objective_D( AD, x_sp, lambda );
  double obj_sp2 = compute_objective_D( AD, x_sp2, lambda );

  //--------------------------------------------------------------------------
  // Speed test - Multiple solves
  //--------------------------------------------------------------------------
  const int NTEST   = 100;
  double    t100_qr = 0.0, t100_kkt_dense = 0.0, t100_sp = 0.0, t100_sp2 = 0.0;

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    // FIX: tmp_y non deve essere volatile per supportare l'operazione di Eigen
    VectorXd          tmp_y = qr_solver.solve( b );
    volatile VectorXd tmp_x = D_inv.asDiagonal() * tmp_y;
    (void) tmp_x;
  }
  tm.toc();
  t100_qr = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = kkt_dense_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_kkt_dense = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    // FIX: tmp_y non deve essere volatile
    VectorXd          tmp_y = sp_solver.solve( b );
    volatile VectorXd tmp_x = D_inv.asDiagonal() * tmp_y;
    (void) tmp_x;
  }
  tm.toc();
  t100_sp = tm.elapsed_ms();

  tm.tic();
  for ( int i = 0; i < NTEST; ++i )
  {
    volatile VectorXd tmp = sp2_solver.solve( b );
    (void) tmp;
  }
  tm.toc();
  t100_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // OUTPUT TABLE
  //--------------------------------------------------------------------------

  // Diagonal matrix info
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " DIAGONAL MATRIX D INFO " );
  fmt::print( "{:<20} {:<20} {:<20}\n", "    Min(D)", "    Max(D)", "    Cond(D)" );
  fmt::print( "{:─<20} {:─<20} {:─<20}\n", "─", "─", "─" );
  double min_D  = D.minCoeff();
  double max_D  = D.maxCoeff();
  double cond_D = max_D / min_D;
  fmt::print( fg( fmt::color::light_blue ), "{:<20} {:<20} {:<20}\n", fmt::format( "{:.4e}", min_D ),
              fmt::format( "{:.4e}", max_D ), fmt::format( "{:.4e}", cond_D ) );

  // Accuracy table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " ACCURACY METRICS (with D) " );
  fmt::print( "{:<54} {:<25}\n", "    Metric", "    Value" );
  fmt::print( "{:─<54} {:─<25}\n", "─", "─" );

  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_qr − b‖ (QR Solver - Transformed)",
              sci( r_qr ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_kkt − b‖ (KKT dense - Native)",
              sci( r_kkt_dense ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp − b‖ (SP QR Solver - Transformed)",
              sci( r_sp ) );
  fmt::print( fg( fmt::color::light_blue ), "{:<54} {:<25}\n", "    ‖A x_sp2 − b‖ (SP KKT Solver - Native)",
              sci( r_sp2 ) );

  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_kkt‖ (QR vs KKT dense)", sci( dx_qr_kkt ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp‖ (QR vs SP QR)", sci( dx_qr_sp ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_qr − x_sp2‖ (QR vs SP KKT)", sci( dx_qr_sp2 ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_kkt − x_sp‖ (KKT dense vs SP QR)",
              sci( dx_kkt_sp ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_kkt − x_sp2‖ (KKT dense vs SP KKT)",
              sci( dx_kkt_sp2 ) );
  fmt::print( fg( fmt::color::violet ), "{:<54} {:<25}\n", "    ‖x_sp − x_sp2‖ (SP QR vs SP KKT)", sci( dx_sp_sp2 ) );

  // Objective function values
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " OBJECTIVE FUNCTION VALUES (with D) " );
  fmt::print( "{:<30} {:<25}\n", "    Solver", "    f(x) = ||Ax-b||² + λ||Dx||²" );
  fmt::print( "{:─<30} {:─<25}\n", "─", "─" );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    QR Solver (Transformed)", sci( obj_qr ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    KKT Solver (Native)", sci( obj_kkt ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP QR Solver (Transformed)", sci( obj_sp ) );
  fmt::print( fg( fmt::color::light_green ), "{:<30} {:<25}\n", "    SP KKT Solver (Native)", sci( obj_sp2 ) );

  // Build times table
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " BUILD TIMES " );
  fmt::print( "{:<26} {:<26} {:<26}\n", "    Method", "    Time", "    Relative" );
  fmt::print( "{:─<26} {:─<26} {:─<26}\n", "─", "─", "─" );

  std::vector<std::pair<double, std::string>> build_times = { { t_build_qr, "QR Solver (dense)" },
                                                              { t_build_kkt_dense, "KKT Solver (dense)" },
                                                              { t_build_sp, "SP QR Solver (sparse)" },
                                                              { t_build_sp2, "SP KKT Solver (sparse)" } };

  double min_build = std::numeric_limits<double>::infinity();
  for ( const auto & bt : build_times )
  {
    if ( bt.first > 0 && bt.first < min_build ) min_build = bt.first;
  }

  for ( const auto & bt : build_times )
  {
    fmt::print( fg( fmt::color::light_green ), "{:<26} {:<26} {:<26}\n", "    " + bt.second,
                bt.first > 0 ? ms_format( bt.first ) : "    FAILED",
                bt.first > 0 ? ratio_format( bt.first / min_build ) : "    N/A" );
  }

  // Solve times table (with transformation overhead)
  fmt::print( fg( fmt::color::yellow ) | fmt::emphasis::bold, "\n{:─^80}\n", " SOLVE TIMES (100 iterations) " );
  fmt::print( "{:<29} {:<16} {:<16} {:<16}\n", "    Method", "    Total Time", "    Time/Solve", "    Speed Ratio" );
  fmt::print( "{:─<29} {:─<16} {:─<16} {:─<16}\n", "─", "─", "─", "─" );

  std::vector<std::tuple<double, std::string>> solve_times = { { t100_qr, "QR Solver (dense)" },
                                                               { t100_kkt_dense, "KKT Solver (dense)" },
                                                               { t100_sp, "SP QR Solver (sparse)" },
                                                               { t100_sp2, "SP KKT Solver (sparse)" } };

  double min_solve = std::numeric_limits<double>::infinity();
  for ( const auto & st : solve_times )
  {
    if ( std::get<0>( st ) < min_solve ) min_solve = std::get<0>( st );
  }

  for ( const auto & st : solve_times )
  {
    double              time = std::get<0>( st );
    const std::string & name = std::get<1>( st );

    fmt::print( fg( fmt::color::light_green ), "{:<29} {:<16} {:<16} {:<16}\n", "    " + name, ms_format( time ),
                small_ms_format( time / NTEST ), ratio_format( time / min_solve ) );
  }
}

// Funzione per stampare la legenda dei metodi
static void
print_solver_legend()
{
  using namespace fmt;

  print( fg( color::white ) | emphasis::bold, "\n{:━^80}\n", " LEGENDA METODI E FORMULAZIONE MATEMATICA " );

  // Descrizione del problema generale
  print( " Problema: " );
  print( fg( color::cyan ), "argmin_x ‖Ax - b‖² + λ‖Dx‖²\n\n" );

  // -----------------------------------------------------------
  // 1. x_qr (Dense QR)
  // -----------------------------------------------------------
  print( fg( color::lime_green ) | emphasis::bold, " 1. x_qr (Dense QR)\n" );
  print( "    • Metodo: Fattorizzazione QR su matrice 'stacked' (densità piena).\n" );
  print( "    • Trasforma il problema in minimi quadrati standard:\n" );
  print( fg( color::yellow ),
         "      min ‖ ⎡   A   ⎤     ⎡ b ⎤ ‖ 2\n"
         "          ‖ ⎢       ⎥ x - ⎢   ⎥ ‖\n"
         "          ‖ ⎣ √λ D  ⎦     ⎣ 0 ⎦ ‖\n" );
  print( "    • Nota: Molto robusto (condizionamento √K), ma lento per grandi dimensioni.\n\n" );

  // -----------------------------------------------------------
  // 2. x_kkt (Dense KKT)
  // -----------------------------------------------------------
  print( fg( color::lime_green ) | emphasis::bold, " 2. x_kkt (Dense KKT / Normal Eq)\n" );
  print( "    • Metodo: Risoluzione diretta delle Equazioni Normali (densità piena).\n" );
  print( "    • Risolve il sistema lineare simmetrico definito positivo:\n" );
  print( fg( color::yellow ), "      ( AᵀA + λ DᵀD ) x = Aᵀb\n" );
  print(
      "    • Nota: Più veloce del QR (costruisce la matrice AᵀA), ma il condizionamento\n"
      "      è al quadrato (K) rispetto al QR. Meno stabile numericamente.\n\n" );

  // -----------------------------------------------------------
  // 3. x_sp (Sparse QR)
  // -----------------------------------------------------------
  print( fg( color::lime_green ) | emphasis::bold, " 3. x_sp (Sparse QR)\n" );
  print( "    • Metodo: Come x_qr, ma sfrutta la sparsità di A e D.\n" );
  print( "    • Usa Eigen::SPQR o Eigen::SparseQR sulla matrice impilata:\n" );
  print( fg( color::yellow ),
         "      min ‖ ⎡ A ⎤ x - ⎡ b ⎤ ‖\n"
         "          ‖ ⎣ Γ ⎦     ⎣ 0 ⎦ ‖   (dove Γ = √λ D)\n" );
  print( "    • Nota: Ottimo per matrici sparse mal condizionate.\n\n" );

  // -----------------------------------------------------------
  // 4. x_sp2 (Sparse KKT)
  // -----------------------------------------------------------
  print( fg( color::lime_green ) | emphasis::bold, " 4. x_sp2 (Sparse KKT / Cholesky)\n" );
  print( "    • Metodo: Equazioni normali sparse con decomposizione di Cholesky (SimplicialLDLT).\n" );
  print( "    • Risolve:\n" );
  print( fg( color::yellow ), "      ( AᵀA + diag(λ dᵢ²) ) x = Aᵀb\n" );
  print( "    • Nota: Generalmente il metodo più veloce per problemi sparsi ben condizionati.\n" );

  print( fg( color::white ) | emphasis::bold, "{:━^80}\n", "" );
}

//==============================================================================
// MAIN
//==============================================================================
int
main()
{
  print_solver_legend();

  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "\n{:━^80}\n",
              " BENCHMARK TIKHONOV SOLVERS: QR vs KKT (dense) vs SP QR vs SP KKT " );

  // Test standard (D = I)
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{:━^80}\n", " STANDARD TESTS (D = I) " );

  size_t total_tests  = sizes.size() * lambda_values.size();
  size_t current_test = 0;

  for ( auto sz : sizes )
  {
    for ( double lambda : lambda_values )
    {
      ++current_test;
      fmt::print( fg( fmt::color::gray ) | fmt::emphasis::faint, "\n[{:2d}/{:2d}] ", static_cast<int>( current_test ),
                  static_cast<int>( total_tests ) );
      try
      {
        run_test_standard( sz.first, sz.second, lambda );
      }
      catch ( const std::exception & e )
      {
        fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\nTest failed with exception: {}\n", e.what() );
      }
    }
  }

  // Test con matrice diagonale D (solo per lambda=0.1 per ridurre il numero di test)
  fmt::print( fg( fmt::color::cyan ) | fmt::emphasis::bold, "\n{:━^80}\n",
              " TESTS WITH DIAGONAL PENALTY MATRIX D (λ=0.1) " );

  double lambda_diag      = 0.1;
  current_test            = 0;
  size_t total_diag_tests = sizes.size();

  for ( auto sz : sizes )
  {
    ++current_test;
    fmt::print( fg( fmt::color::gray ) | fmt::emphasis::faint, "\n[DIAG {:2d}/{:2d}] ",
                static_cast<int>( current_test ), static_cast<int>( total_diag_tests ) );
    try
    {
      run_test_with_diagonal( sz.first, sz.second, lambda_diag );
    }
    catch ( const std::exception & e )
    {
      fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "\nTest with diagonal failed with exception: {}\n",
                  e.what() );
    }
  }

  fmt::print( fg( fmt::color::lime_green ) | fmt::emphasis::bold, "\n{:━^80}\n", " BENCHMARK COMPLETED " );

  return 0;
}
