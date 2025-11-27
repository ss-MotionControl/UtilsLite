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
 |  Benchmark comparison: TikhonovSolver vs TikhonovPseudoInverse
 |
\*--------------------------------------------------------------------------*/

#include <Eigen/Dense>
#include <iostream>
#include <chrono>

#include "Utils_pseudoinverse.hh"
#include "Utils_fmt.hh"

using namespace Eigen;
using std::vector;
using std::pair;

//==============================================================================
static vector<double> lambda_values = {
  0.0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10
};

static vector<pair<int,int>> sizes = {
  {50,50}, {100,50}, {200,50},
  {100,100}, {200,100}, {300,100}
};

//==============================================================================
//  FORMATTER UTILS
//==============================================================================
inline void header(std::string const& msg) {
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "  {:^80}  \n", msg);
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,
             "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

inline void subheader(std::string const& msg) {
  fmt::print(fg(fmt::color::light_blue) | fmt::emphasis::bold,
             "\n┌{0:─^{2}}┐\n"
             "│{1:^{2}}│\n"
             "└{0:─^{2}}┘\n", "", msg, 78);
}

inline std::string sci(double v) {
  return fmt::format("{:>12.3e}", v);
}

inline std::string ms_format(double v) {
  return fmt::format("    {}",fmt::format("{:.2f} ms", v) );
}

inline std::string ratio_format(double v) {
  return fmt::format("    {}",fmt::format("{:.2f}x", v) );
}

inline std::string small_ms_format(double v) {
  return fmt::format("    {}",fmt::format("{:.4f} ms", v) );
}

//==============================================================================
//  RUN TEST
//==============================================================================
static void run_test(int m, int n, double lambda) {

  Utils::TicToc tm;

  header(fmt::format("TEST: m={}, n={}, λ={}", m, n, lambda));

  MatrixXd A = MatrixXd::Random(m,n);
  VectorXd b = VectorXd::Random(m);
  VectorXd c = VectorXd::Random(n);

  //--------------------------------------------------------------------------
  // Build pseudo-inverse
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovPseudoInverse pinv(A, lambda);
  tm.toc();
  double t_build_pinv = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Build QR solver
  //--------------------------------------------------------------------------
  tm.tic();
  Utils::TikhonovSolver solver(A, lambda);
  tm.toc();
  double t_build_solver = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // Accuracy
  //--------------------------------------------------------------------------
  VectorXd x1 = pinv.apply(b);
  VectorXd x2 = solver.solve(b);
  double r1 = (A*x1 - b).norm();
  double r2 = (A*x2 - b).norm();
  double dx = (x1 - x2).norm();

  VectorXd y1 = pinv.apply_transpose(c);
  VectorXd y2 = solver.solve_transpose(c);
  double dy = (y1 - y2).norm();

  //--------------------------------------------------------------------------
  // Speed test
  //--------------------------------------------------------------------------
  const int NTEST = 100;

  tm.tic();
  for (int i=0; i<NTEST; ++i) volatile auto tmp = pinv.apply(b);
  tm.toc();
  double t100_pinv = tm.elapsed_ms();

  tm.tic();
  for (int i=0; i<NTEST; ++i) volatile auto tmp = solver.solve(b);
  tm.toc();
  double t100_solver = tm.elapsed_ms();

  //--------------------------------------------------------------------------
  // OUTPUT TABLE
  //--------------------------------------------------------------------------

  subheader("PERFORMANCE RESULTS");

  // Build times table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold, "\n{:─^80}\n", " BUILD TIMES ");
  fmt::print("{:<26} {:<26} {:<26}\n", "    Method", "    Time", "    Relative");
  fmt::print("{:─<26} {:─<26} {:─<26}\n", "─", "─", "─");
  
  double min_build = std::min(t_build_pinv, t_build_solver);
  fmt::print(fg(fmt::color::light_green),
             "{:<26} {:<26} {:<26}\n",
             "    PseudoInverse", 
             ms_format(t_build_pinv),
             ratio_format(t_build_pinv/min_build));
  
  fmt::print(fg(fmt::color::light_green),
             "{:<26} {:<26} {:<26}\n",
             "    QR Solver", 
             ms_format(t_build_solver),
             ratio_format(t_build_solver/min_build));

  // Accuracy table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
             "\n{:─^80}\n", " ACCURACY METRICS ");
  fmt::print("{:<44} {:<35}\n", "    Metric", "    Value");
  fmt::print("{:─<44} {:─<35}\n", "─", "─");
  
  fmt::print(fg(fmt::color::light_blue), "{:<44} {:<35}\n",  "    ‖A x₁ − b‖ (PseudoInverse)", sci(r1));
  
  fmt::print(fg(fmt::color::light_blue),
             "{:<44} {:<35}\n",
             "    ‖A x₂ − b‖ (QR Solver)", sci(r2));
  
  fmt::print(fg(fmt::color::violet),
             "{:<44} {:<35}\n",
             "    ‖x₁ − x₂‖ (Solution Difference)", sci(dx));
  
  fmt::print(fg(fmt::color::orange),
             "{:<44} {:<35}\n",
             "    ‖y₁ − y₂‖ (Transpose Difference)", sci(dy));

  // Solve times table
  fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
             "\n{:─^80}\n", " SOLVE TIMES (100 iterations) ");
  fmt::print("{:<20} {:<18} {:<18} {:<18}\n",
             "    Method", "    Total Time", "    Time/Solve", "    Speed Ratio");
  fmt::print("{:─<20} {:─<18} {:─<18} {:─<18}\n", "─", "─", "─", "─");
  
  fmt::print(fg(fmt::color::pink),
             "{:<20} {:<18} {:<18} {:<18}\n",
             "    PseudoInverse",
             ms_format(t100_pinv),
             small_ms_format(t100_pinv/NTEST),
             "");
  
  fmt::print(fg(fmt::color::pink),
             "{:<20} {:<18} {:<18} {:<18}\n",
             "    QR Solver", 
             ms_format(t100_solver),
             small_ms_format(t100_solver/NTEST),
             ratio_format(t100_pinv/t100_solver));
  
  fmt::print(fg(fmt::color::gray),
             "\n{}\n", "────────────────────────────────────────────────────────────────────────────────");
}

//==============================================================================
//  MAIN
//==============================================================================
int main() {

  fmt::print(fg(fmt::color::lime_green) | fmt::emphasis::bold,
             "\n{:=^80}\n", " BENCHMARK TIKHONOV SOLVERS ");

  int total_tests = sizes.size() * lambda_values.size();
  int current_test = 0;

  for (auto sz : sizes) {
    for (double lambda : lambda_values) {
      ++current_test;
      fmt::print(fg(fmt::color::gray) | fmt::emphasis::faint,
                 "\n[{:2d}/{:2d}] ", current_test, total_tests);
      run_test(sz.first, sz.second, lambda);
    }
  }

  fmt::print(fg(fmt::color::lime_green) | fmt::emphasis::bold,
             "\n{:=^80}\n", " BENCHMARK COMPLETED ");

  return 0;
}
