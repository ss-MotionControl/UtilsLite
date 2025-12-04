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

/*--------------------------------------------------------------------------*\
 |
 |  Benchmark comparison: TikhonovSolver vs TikhonovSolver2 (KKT dense) 
 |                         vs SP_TikhonovSolver vs SP_TikhonovSolver2
 |
\*--------------------------------------------------------------------------*/

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <chrono>
#include <random>
#include <set>
#include <cstddef>
#include <limits>

#include "Utils_pseudoinverse.hh"
#include "Utils_fmt.hh"

using namespace Eigen;
using std::vector;
using std::pair;

//==============================================================================
static vector<double> lambda_values = {0.0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10};
static vector<pair<int,int>> sizes = {{50,50}, {100,50}, {200,50}, {100,100}, {200,100}, {300,100}, {3000,200}};

//==============================================================================
// Funzione per generare una matrice sparsa casuale
template <typename Scalar>
Eigen::SparseMatrix<Scalar>
generate_random_sparse_matrix(int m, int n, double density = 0.3) {
  using SparseMatrix = Eigen::SparseMatrix<Scalar>;
  SparseMatrix mat(m, n);

  size_t nnz = static_cast<size_t>(density * m * n);

  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> row_dist(0, m-1);
  std::uniform_int_distribution<> col_dist(0, n-1);
  std::uniform_real_distribution<Scalar> val_dist(-1.0, 1.0);

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(nnz);

  std::set<std::pair<int, int>> positions;

  for (size_t k = 0; k < nnz; ++k) {
    int i, j;
    do {
      i = row_dist(gen);
      j = col_dist(gen);
    } while (positions.count({i, j}) > 0);

    positions.insert({i, j});
    triplets.emplace_back(i, j, val_dist(gen));
  }

  mat.setFromTriplets(triplets.begin(), triplets.end());
  mat.makeCompressed();

  return mat;
}

//==============================================================================
// FORMATTER UTILS
//==============================================================================
inline void header(std::string const& msg) {
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "\n"
             "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n");
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "┃ {:^76} ┃\n", msg);
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n");
}

inline void subheader(std::string const& msg) {
  fmt::print(fg(fmt::color::light_blue) | fmt::emphasis::bold,
             "\n"
             "╭{0:─^{2}}╮\n"
             "│{1:^{2}}│\n"
             "╰{0:─^{2}}╯\n", "", msg, 78);
}

inline std::string sci(double v) { return fmt::format("{:>12.3e}", v); }
inline std::string ms_format(double v) { return fmt::format("    {}",fmt::format("{:.2f} ms", v)); }
inline std::string ratio_format(double v) { return fmt::format("    {}",fmt::format("{:.2f}x", v)); }
inline std::string small_ms_format(double v) { return fmt::format("    {}",fmt::format("{:.4f} ms", v)); }

//==============================================================================
// RUN TEST
//==============================================================================
static void run_test(int m, int n, double lambda) {

  Utils::TicToc tm;

  header(fmt::format("TEST: m={}, n={}, λ={}", m, n, lambda));

  // Genera matrice sparsa casuale
  SparseMatrix<double> AS = generate_random_sparse_matrix<double>(m, n, 0.3);
  // Converti in densa per i solver dense
  MatrixXd AD = MatrixXd(AS);

  VectorXd b = VectorXd::Random(m);
  VectorXd c = VectorXd::Random(n);

  //--------------------------------------------------------------------------
  // Build TikhonovSolver (QR) - DENSE
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver<double> qr_solver(AD, lambda);
  tm.toc();
  double t_build_qr = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build TikhonovSolver2 (KKT dense)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver2<double> kkt_dense_solver(AD, lambda);
  tm.toc();
  double t_build_kkt_dense = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver (sparse QR)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver<double> sp_solver(AS, lambda);
  tm.toc();
  double t_build_sp = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build SP_TikhonovSolver2 (sparse KKT)
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::SP_TikhonovSolver2<double> sp2_solver(AS, lambda);
  tm.toc();
  double t_build_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Accuracy - Solve min ||A x - b||^2 + λ||x||^2
  //--------------------------------------------------------------------------
  VectorXd x_qr = qr_solver.solve(b);
  double r_qr = (AD*x_qr - b).norm();

  VectorXd x_kkt_dense = kkt_dense_solver.solve(b);
  double r_kkt_dense = (AD*x_kkt_dense - b).norm();

  VectorXd x_sp = sp_solver.solve(b);
  double r_sp = (AS*x_sp - b).norm();

  VectorXd x_sp2 = sp2_solver.solve(b);
  double r_sp2 = (AS*x_sp2 - b).norm();

  double dx_qr_kkt = (x_qr - x_kkt_dense).norm();
  double dx_qr_sp = (x_qr - x_sp).norm();
  double dx_qr_sp2 = (x_qr - x_sp2).norm();
  double dx_kkt_sp = (x_kkt_dense - x_sp).norm();
  double dx_kkt_sp2 = (x_kkt_dense - x_sp2).norm();
  double dx_sp_sp2 = (x_sp - x_sp2).norm();

  //--------------------------------------------------------------------------
  // Accuracy - Transpose solve: y = A * (A^T A + λ I)^{-1} * c
  //--------------------------------------------------------------------------
  VectorXd y_qr = qr_solver.solve_transpose(c);
  VectorXd y_kkt_dense = kkt_dense_solver.solve_transpose(c);
  VectorXd y_sp = sp_solver.solve_transpose(c);
  VectorXd y_sp2 = sp2_solver.solve_transpose(c);

  double dy_qr_kkt = (y_qr - y_kkt_dense).norm();
  double dy_qr_sp = (y_qr - y_sp).norm();
  double dy_qr_sp2 = (y_qr - y_sp2).norm();
  double dy_kkt_sp = (y_kkt_dense - y_sp).norm();
  double dy_kkt_sp2 = (y_kkt_dense - y_sp2).norm();
  double dy_sp_sp2 = (y_sp - y_sp2).norm();

  //--------------------------------------------------------------------------
  // Speed test - Multiple solves
  //--------------------------------------------------------------------------
  const int NTEST = 100;
  double t100_qr = 0.0, t100_kkt_dense = 0.0, t100_sp = 0.0, t100_sp2 = 0.0;

  tm.tic();
  for (int i=0; i<NTEST; ++i) { volatile VectorXd tmp = qr_solver.solve(b); (void)tmp; }
  tm.toc();
  t100_qr = tm.elapsed_ms();

  tm.tic();
  for (int i=0; i<NTEST; ++i) { volatile VectorXd tmp = kkt_dense_solver.solve(b); (void)tmp; }
  tm.toc();
  t100_kkt_dense = tm.elapsed_ms();

  tm.tic();
  for (int i=0; i<NTEST; ++i) { volatile VectorXd tmp = sp_solver.solve(b); (void)tmp; }
  tm.toc();
  t100_sp = tm.elapsed_ms();

  tm.tic();
  for (int i=0; i<NTEST; ++i) { volatile VectorXd tmp = sp2_solver.solve(b); (void)tmp; }
  tm.toc();
  t100_sp2 = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // OUTPUT TABLE
  //--------------------------------------------------------------------------

  // Accuracy table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
             "\n{:─^80}\n", " ACCURACY METRICS ");
  fmt::print("{:<54} {:<25}\n", "    Metric", "    Value");
  fmt::print("{:─<54} {:─<25}\n", "─", "─");

  fmt::print(fg(fmt::color::light_blue),
             "{:<54} {:<25}\n", "    ‖A x_qr − b‖ (QR Solver)", sci(r_qr));
  fmt::print(fg(fmt::color::light_blue),
             "{:<54} {:<25}\n", "    ‖A x_kkt − b‖ (KKT dense)", sci(r_kkt_dense));
  fmt::print(fg(fmt::color::light_blue),
             "{:<54} {:<25}\n", "    ‖A x_sp − b‖ (SP QR Solver)", sci(r_sp));
  fmt::print(fg(fmt::color::light_blue),
             "{:<54} {:<25}\n", "    ‖A x_sp2 − b‖ (SP KKT Solver)", sci(r_sp2));

  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_qr − x_kkt‖ (QR vs KKT dense)", sci(dx_qr_kkt));
  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_qr − x_sp‖ (QR vs SP QR)", sci(dx_qr_sp));
  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_qr − x_sp2‖ (QR vs SP KKT)", sci(dx_qr_sp2));
  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_kkt − x_sp‖ (KKT dense vs SP QR)", sci(dx_kkt_sp));
  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_kkt − x_sp2‖ (KKT dense vs SP KKT)", sci(dx_kkt_sp2));
  fmt::print(fg(fmt::color::violet),
             "{:<54} {:<25}\n", "    ‖x_sp − x_sp2‖ (SP QR vs SP KKT)", sci(dx_sp_sp2));

  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_qr − y_kkt‖ (Transpose QR vs KKT)", sci(dy_qr_kkt));
  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_qr − y_sp‖ (Transpose QR vs SP QR)", sci(dy_qr_sp));
  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_qr − y_sp2‖ (Transpose QR vs SP KKT)", sci(dy_qr_sp2));
  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_kkt − y_sp‖ (Transpose KKT vs SP QR)", sci(dy_kkt_sp));
  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_kkt − y_sp2‖ (Transpose KKT vs SP KKT)", sci(dy_kkt_sp2));
  fmt::print(fg(fmt::color::orange),
             "{:<54} {:<25}\n", "    ‖y_sp − y_sp2‖ (Transpose SP QR vs SP KKT)", sci(dy_sp_sp2));

  // Build times table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold, "\n{:─^80}\n", " BUILD TIMES ");
  fmt::print("{:<26} {:<26} {:<26}\n", "    Method", "    Time", "    Relative");
  fmt::print("{:─<26} {:─<26} {:─<26}\n", "─", "─", "─");

  std::vector<std::pair<double, std::string>> build_times = {
    {t_build_qr, "QR Solver (dense)"},
    {t_build_kkt_dense, "KKT Solver (dense)"},
    {t_build_sp, "SP QR Solver (sparse)"},
    {t_build_sp2, "SP KKT Solver (sparse)"}
  };

  double min_build = std::numeric_limits<double>::infinity();
  for (const auto& bt : build_times) { 
    if (bt.first > 0 && bt.first < min_build) min_build = bt.first; 
  }

  for (const auto& bt : build_times) {
    fmt::print(fg(fmt::color::light_green),
               "{:<26} {:<26} {:<26}\n",
               "    " + bt.second,
               bt.first > 0 ? ms_format(bt.first) : "    FAILED",
               bt.first > 0 ? ratio_format(bt.first/min_build) : "    N/A");
  }

  // Solve times table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
             "\n{:─^80}\n", " SOLVE TIMES (100 iterations) ");
  fmt::print("{:<29} {:<16} {:<16} {:<16}\n",
             "    Method", "    Total Time", "    Time/Solve", "    Speed Ratio");
  fmt::print("{:─<29} {:─<16} {:─<16} {:─<16}\n", "─", "─", "─", "─");

  std::vector<std::tuple<double, std::string>> solve_times = {
    {t100_qr, "QR Solver (dense)"},
    {t100_kkt_dense, "KKT Solver (dense)"},
    {t100_sp, "SP QR Solver (sparse)"},
    {t100_sp2, "SP KKT Solver (sparse)"}
  };

  double min_solve = std::numeric_limits<double>::infinity();
  for (const auto& st : solve_times) { 
    if (std::get<0>(st) < min_solve) min_solve = std::get<0>(st); 
  }

  for (const auto& st : solve_times) {
    double time = std::get<0>(st);
    const std::string& name = std::get<1>(st);

    fmt::print(fg(fmt::color::light_green),
               "{:<29} {:<16} {:<16} {:<16}\n",
               "    " + name,
               ms_format(time),
               small_ms_format(time/NTEST),
               ratio_format(time/min_solve));
  }
}

//==============================================================================
// MAIN
//==============================================================================
int main() {

  fmt::print(fg(fmt::color::lime_green) | fmt::emphasis::bold,
             "\n{:━^80}\n", " BENCHMARK TIKHONOV SOLVERS: QR vs KKT (dense) vs SP QR vs SP KKT ");

  size_t total_tests = sizes.size() * lambda_values.size();
  size_t current_test = 0;

  for (auto sz : sizes) {
    for (double lambda : lambda_values) {
      ++current_test;
      fmt::print(fg(fmt::color::gray) | fmt::emphasis::faint,
                 "\n[{:2d}/{:2d}] ", static_cast<int>(current_test), static_cast<int>(total_tests));
      try {
        run_test(sz.first, sz.second, lambda);
      } catch (const std::exception& e) {
        fmt::print(fg(fmt::color::red) | fmt::emphasis::bold,
                   "\nTest failed with exception: {}\n", e.what());
      }
    }
  }

  fmt::print(fg(fmt::color::lime_green) | fmt::emphasis::bold,
             "\n{:━^80}\n", " BENCHMARK COMPLETED ");

  // Test aggiuntivo: confronto tra tutti i solver per matrice densa piena
  fmt::print(fg(fmt::color::lime_green) | fmt::emphasis::bold,
             "\n{:━^80}\n", " ADDITIONAL TEST: FULL DENSE MATRIX COMPARISON ");
  
  {
    int m = 100, n = 50;
    double lambda = 0.1;
    
    // Genera matrice densa casuale
    MatrixXd AD_dense = MatrixXd::Random(m, n);
    VectorXd b = VectorXd::Random(m);
    VectorXd c = VectorXd::Random(n);
    
    Utils::TicToc tm;
    
    // QR Solver
    tm.tic();
    Utils::TikhonovSolver<double> qr_solver(AD_dense, lambda);
    VectorXd x_qr = qr_solver.solve(b);
    tm.toc();
    double t_qr = tm.elapsed_ms();
    
    // KKT Dense Solver
    tm.tic();
    Utils::TikhonovSolver2<double> kkt_solver(AD_dense, lambda);
    VectorXd x_kkt = kkt_solver.solve(b);
    tm.toc();
    double t_kkt = tm.elapsed_ms();
    
    // Converti in sparsa per testare i solver sparse (anche se non efficiente)
    SparseMatrix<double> AS = AD_dense.sparseView();
    
    // SP QR Solver
    tm.tic();
    Utils::SP_TikhonovSolver<double> sp_solver(AS, lambda);
    VectorXd x_sp = sp_solver.solve(b);
    tm.toc();
    double t_sp = tm.elapsed_ms();
    
    // SP KKT Solver
    tm.tic();
    Utils::SP_TikhonovSolver2<double> sp_kkt_solver(AS, lambda);
    VectorXd x_sp_kkt = sp_kkt_solver.solve(b);
    tm.toc();
    double t_sp_kkt = tm.elapsed_ms();
    
    fmt::print(fg(fmt::color::cyan),
               "\nFull dense matrix ({}x{}) with λ = {}:\n", m, n, lambda);
    fmt::print(fg(fmt::color::light_blue),
               "    QR Solver (dense): {:.4f} ms, solution norm: {:.6e}\n", 
               t_qr, x_qr.norm());
    fmt::print(fg(fmt::color::light_blue),
               "    KKT Solver (dense): {:.4f} ms, solution norm: {:.6e}\n", 
               t_kkt, x_kkt.norm());
    fmt::print(fg(fmt::color::light_blue),
               "    SP QR Solver (sparse): {:.4f} ms, solution norm: {:.6e}\n", 
               t_sp, x_sp.norm());
    fmt::print(fg(fmt::color::light_blue),
               "    SP KKT Solver (sparse): {:.4f} ms, solution norm: {:.6e}\n", 
               t_sp_kkt, x_sp_kkt.norm());
    
    double diff_qr_kkt = (x_qr - x_kkt).norm();
    double diff_qr_sp = (x_qr - x_sp).norm();
    double diff_qr_sp_kkt = (x_qr - x_sp_kkt).norm();
    
    fmt::print(fg(fmt::color::violet),
               "\n    Differences:\n");
    fmt::print(fg(fmt::color::violet),
               "        QR vs KKT: {:.6e}\n", diff_qr_kkt);
    fmt::print(fg(fmt::color::violet),
               "        QR vs SP QR: {:.6e}\n", diff_qr_sp);
    fmt::print(fg(fmt::color::violet),
               "        QR vs SP KKT: {:.6e}\n", diff_qr_sp_kkt);
  }

  return 0;
}
