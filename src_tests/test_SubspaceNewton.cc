// -----------------------------------------------------------------------------
// File: test_SubspaceNewton.cc
// -----------------------------------------------------------------------------

/*--------------------------------------------------------------------------*\
 |  Driver program for testing Subspace Newton with Kaczmarz-style        |
 |  selection (cyclic/random/greedy) and optional line-search on          |
 |  all nonlinear system test problems.                                   |
\*--------------------------------------------------------------------------*/

#include "Utils_fmt.hh"
#include "Utils_SubspaceNewton.hh"
#include "Utils_nonlinear_system.hh"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

using namespace Utils;
using namespace std;

//  ---------- same TestResult structure ----------
struct TestResult {
  string test_name;
  int    num_equations;
  bool   converged;
  int    iterations;
  int    function_evals;
  int    jacobian_evals;
  double final_residual;
  double elapsed_time_ms;
  int    initial_point_index;

  TestResult()
  : num_equations(0)
  , converged(false)
  , iterations(0)
  , function_evals(0)
  , jacobian_evals(0)
  , final_residual(0.0)
  , elapsed_time_ms(0.0)
  , initial_point_index(-1)
  {}
};

// ---------- same Statistics struct ----------
struct Statistics {
  int    total_tests;
  int    converged_tests;
  int    failed_tests;
  double success_rate;
  double avg_iterations;
  double avg_function_evals;
  double avg_jacobian_evals;
  double avg_time_ms;
  double max_time_ms;
  double min_time_ms;

  Statistics()
  : total_tests(0)
  , converged_tests(0)
  , failed_tests(0)
  , success_rate(0.0)
  , avg_iterations(0.0)
  , avg_function_evals(0.0)
  , avg_jacobian_evals(0.0)
  , avg_time_ms(0.0)
  , max_time_ms(0.0)
  , min_time_ms(numeric_limits<double>::max())
  {}
};

// ---------- same utility functions ----------
string truncate_string(string const & str, size_t max_length) {
  if (str.length() <= max_length) return str;
  return str.substr(0, max_length - 3) + "...";
}

void print_progress(int current, int total) {
  double progress = static_cast<double>(current)/static_cast<double>(total);
  Utils::progress_bar(std::cout, progress, 50, "Progress:");
}


// Funzione per stampare la tabella riassuntiva usando fmt con allineamento perfetto
void
print_summary_table(const vector<TestResult>& results) {
  // Dimensioni delle colonne
  constexpr int col_idx      = 5;  // # (indice)
  constexpr int col_status   = 8;  // Status (incluso spazio per i colori)
  constexpr int col_neq      = 7;  // NEQ
  constexpr int col_iter     = 5;  // Iter
  constexpr int col_feval    = 8;  // F-Eval
  constexpr int col_jeval    = 8;  // J-Eval
  constexpr int col_residual = 12; // Residual
  constexpr int col_time     = 10; // Time(ms)
  constexpr int col_name     = 45; // Test Name
  
  // Calcola la larghezza totale della tabella
  constexpr int total_width = 2 + col_idx + 3 + col_status + 3 + col_neq + 3 +
                             col_iter + 3 + col_feval + 3 + col_jeval + 3 +
                             col_residual + 3 + col_time + 3 + col_name + 2;
  
  // Intestazione della tabella
  fmt::print("\n\n");
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold, "{:━^{}}\n", " NEWTON METHOD TEST RESULTS ", total_width);
  
  // Linea divisoria
  fmt::print(fg(fmt::color::cyan), "┏{}┓\n", fmt::format("{:━^{}}", "", total_width - 2));
  
  // Intestazione delle colonne
  fmt::print(fg(fmt::color::cyan), "┃ ");
  fmt::print("{:>{}} │ ", "#",         col_idx);
  fmt::print("{:^{}} │ ", "Status",    col_status);
  fmt::print("{:>{}} │ ", "NEQ",       col_neq);
  fmt::print("{:>{}} │ ", "Iter",      col_iter);
  fmt::print("{:>{}} │ ", "F-Eval",    col_feval);
  fmt::print("{:>{}} │ ", "J-Eval",    col_jeval);
  fmt::print("{:>{}} │ ", "Residual",  col_residual);
  fmt::print("{:>{}} │ ", "Time(ms)",  col_time);
  fmt::print("{:<{}} ",   "Test Name", col_name);
  fmt::print(fg(fmt::color::cyan), "┃\n");
  
  // Linea divisoria
  fmt::print(fg(fmt::color::cyan), "┠{}┨\n", fmt::format("{:─^{}}", "", total_width - 2));
  
  // Dati
  for (size_t i = 0; i < results.size(); ++i) {
    auto const & r = results[i];
    
    // Colonna indice
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:>{}} │ ", i + 1, col_idx);
    
    // Colonna status
    if (r.converged) {
      fmt::print(fg(fmt::color::green), "{:^{}}", "✓ OK", col_status);
    } else {
      fmt::print(fg(fmt::color::red), "{:^{}}", "✗ FAIL", col_status);
    }
    fmt::print(fg(fmt::color::cyan), " │ ");
    
    // Colonna NEQ
    fmt::print("{:>{}} │ ", r.num_equations, col_neq);
    
    // Colonna Iter
    fmt::print("{:>{}} │ ", r.iterations, col_iter);
    
    // Colonna F-Eval
    fmt::print("{:>{}} │ ", r.function_evals, col_feval);
    
    // Colonna J-Eval
    fmt::print("{:>{}} │ ", r.jacobian_evals, col_jeval);
    
    // Colonna Residual
    if (r.final_residual == 0.0) {
      fmt::print("{:>12} │ ", "0.00e+00");
    } else {
      fmt::print("{:>12.2e} │ ", r.final_residual);
    }
    
    // Colonna Time
    if (r.elapsed_time_ms < 0.01) {
      fmt::print("{:>10.3f} │ ", r.elapsed_time_ms);
    } else if (r.elapsed_time_ms < 1.0) {
      fmt::print("{:>10.2f} │ ", r.elapsed_time_ms);
    } else if (r.elapsed_time_ms < 10.0) {
      fmt::print("{:>10.2f} │ ", r.elapsed_time_ms);
    } else if (r.elapsed_time_ms < 100.0) {
      fmt::print("{:>10.2f} │ ", r.elapsed_time_ms);
    } else {
      fmt::print("{:>10.0f} │ ", r.elapsed_time_ms);
    }
    
    // Colonna Test Name
    string test_name = truncate_string(r.test_name, col_name);
    fmt::print("{:<{}} ", test_name, col_name);
    
    fmt::print(fg(fmt::color::cyan), "┃\n");
  }
  
  // Linea finale
  fmt::print(fg(fmt::color::cyan), "┗{}┛\n", fmt::format("{:━^{}}", "", total_width - 2));
  //fmt::print(fmt::text_style());
}

// Funzione per calcolare e stampare le statistiche con allineamento perfetto
void
print_statistics(const vector<TestResult>& results) {
  Statistics stats;
  stats.total_tests = results.size();
  
  double total_iterations     = 0.0;
  double total_function_evals = 0.0;
  double total_jacobian_evals = 0.0;
  double total_time           = 0.0;
  
  for ( auto const & r : results) {
    if (r.converged) {
      stats.converged_tests++;
      total_iterations     += r.iterations;
      total_function_evals += r.function_evals;
      total_jacobian_evals += r.jacobian_evals;
      total_time           += r.elapsed_time_ms;
      
      if (r.elapsed_time_ms > stats.max_time_ms) stats.max_time_ms = r.elapsed_time_ms;
      if (r.elapsed_time_ms < stats.min_time_ms) stats.min_time_ms = r.elapsed_time_ms;
    } else {
      stats.failed_tests++;
    }
  }
  
  stats.success_rate = 100.0 * stats.converged_tests / stats.total_tests;
  
  if (stats.converged_tests > 0) {
      stats.avg_iterations     = total_iterations / stats.converged_tests;
      stats.avg_function_evals = total_function_evals / stats.converged_tests;
      stats.avg_jacobian_evals = total_jacobian_evals / stats.converged_tests;
      stats.avg_time_ms        = total_time / stats.converged_tests;
  }
  
  // Dimensioni per la tabella delle statistiche
  constexpr int stat_col_label   = 25;
  constexpr int stat_col_value   = 12;
  constexpr int stat_total_width = stat_col_label + stat_col_value + 4; // +6 per bordi e spazi
    
  // Stampa delle statistiche
  fmt::print("\n");
  fmt::print(fg(fmt::color::cyan) | fmt::emphasis::bold,  "{:━^{}}\n", " STATISTICAL SUMMARY ", stat_total_width);
  fmt::print(fg(fmt::color::cyan), "┏{}┓\n", fmt::format("{:━^{}}", "", stat_total_width - 2));
  fmt::print(fg(fmt::color::cyan), "┃");
  fmt::print("{:^{}}", "", stat_total_width - 2);
  fmt::print(fg(fmt::color::cyan), "┃\n");
  
  // Total Tests
  fmt::print(fg(fmt::color::cyan), "┃ ");
  fmt::print("{:<{}}", "Total Tests:", stat_col_label);
  fmt::print(fg(fmt::color::white), "{:>{}}", stats.total_tests, stat_col_value);
  fmt::print(fg(fmt::color::cyan), " ┃\n");
  
  // Converged Tests
  fmt::print(fg(fmt::color::cyan), "┃ ");
  fmt::print("{:<{}}", "Converged Tests:", stat_col_label);
  fmt::print(fg(fmt::color::green), "{:>{}}", fmt::format("{} ({:.1f}%)", stats.converged_tests, stats.success_rate), stat_col_value);
  fmt::print(fg(fmt::color::cyan), " ┃\n");
  
  // Failed Tests
  fmt::print(fg(fmt::color::cyan), "┃ ");
  fmt::print("{:<{}}", "Failed Tests:", stat_col_label);
  fmt::print(fg(fmt::color::red), "{:>{}}", fmt::format("{} ({:.1f}%)", stats.failed_tests, 100.0 - stats.success_rate), stat_col_value);
  fmt::print(fg(fmt::color::cyan), " ┃\n");
  
  // Linea divisoria
  fmt::print(fg(fmt::color::cyan), "┠{}┨\n", fmt::format("{:─^{}}", "", stat_total_width - 2));
  
  if (stats.converged_tests > 0) {
    // Average Iterations
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Average Iterations:", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.avg_iterations, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
    
    // Average Function Evals
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Average Function Evals:", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.avg_function_evals, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
    
    // Average Jacobian Evals
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Average Jacobian Evals:", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.avg_jacobian_evals, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
    
    // Average Time
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Average Time (ms):", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.avg_time_ms, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
    
    // Min Time
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Min Time (ms):", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.min_time_ms, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
    
    // Max Time
    fmt::print(fg(fmt::color::cyan), "┃ ");
    fmt::print("{:<{}}", "Max Time (ms):", stat_col_label);
    fmt::print(fg(fmt::color::white), "{:>{}.2f}", stats.max_time_ms, stat_col_value);
    fmt::print(fg(fmt::color::cyan), " ┃\n");
  }
    
  fmt::print(fg(fmt::color::cyan), "┃");
  fmt::print("{:^{}}", "", stat_total_width - 2);
  fmt::print(fg(fmt::color::cyan), "┃\n");
  fmt::print(fg(fmt::color::cyan), "┗{}┛\n", fmt::format("{:━^{}}", "", stat_total_width - 2));
  //fmt::print(fmt::text_style());
}


int main(int argc, char* argv[]) {

  Utils::TicToc tm;

  fmt::print("\n");
  fmt::print(
    fg(fmt::color::cyan) | fmt::emphasis::bold,
    "\n"
    "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n"
    "┃ SUBSPACE NEWTON WITH SELECTION AND LINE SEARCH - TEST SUITE ┃\n"
    "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n"
    "\n"
  );

  // Inizializza database test
  init_nonlinear_system_tests();

  fmt::print(fg(fmt::color::yellow), "Total number of test problems: ");
  fmt::print(fg(fmt::color::white),  "{}\n\n", nonlinear_system_tests.size());

  // Default params
  double tolerance      = 1e-8;
  int max_iterations    = 100;
  bool verbose_mode     = false;
  int block_size        = 1;
  bool use_line_search  = true;
  string strategy_name  = "random";

  // usage message identical but extended
  auto print_usage = [&]() {
    fmt::print("Usage: {} [options]\n", argv[0]);
    fmt::print("Options:\n");
    fmt::print("  --help                    Show help\n");
    fmt::print("  --verbose                 Enable verbose output\n");
    fmt::print("  --max-iter=N              Set max Subspace Newton iterations\n");
    fmt::print("  --tolerance=VAL           Set residual tolerance\n");
    fmt::print("  --block-size=N            Set subspace size (1 => coordinate Newton)\n");
    fmt::print("  --strategy=cyclic|random|greedy\n");
    fmt::print("  --no-linesearch           Disable line search\n");
  };

  // parse args
  for (int i=1;i<argc;++i) {
    string arg = argv[i];
    if (arg=="--help"||arg=="-h") {
      print_usage();
      return 0;
    }
    else if(arg=="--verbose") {
      verbose_mode=true;
    }
    else if(arg.rfind("--max-iter=",0)==0) {
      max_iterations=stoi(arg.substr(11));
    }
    else if(arg.rfind("--tolerance=",0)==0) {
      tolerance=stod(arg.substr(12));
    }
    else if(arg.rfind("--block-size=",0)==0) {
      block_size=stoi(arg.substr(13));
    }
    else if(arg.rfind("--strategy=",0)==0) {
      strategy_name=arg.substr(11);
    }
    else if(arg=="--no-linesearch") {
      use_line_search=false;
    }
    else {
      fmt::print(fg(fmt::color::yellow),"Unknown option: {}\n",arg);
      print_usage();
      return 1;
    }
  }

  fmt::print(fg(fmt::color::yellow), "Tolerance: "); fmt::print("{}\n", tolerance);
  fmt::print(fg(fmt::color::yellow), "Max iterations: "); fmt::print("{}\n", max_iterations);
  fmt::print(fg(fmt::color::yellow), "Block size: "); fmt::print("{}\n", block_size);
  fmt::print(fg(fmt::color::yellow), "Strategy: "); fmt::print("{}\n", strategy_name);
  fmt::print(fg(fmt::color::yellow), "Line search: "); fmt::print("{}\n\n", use_line_search?"ON":"OFF");

  // convert strategy
  SubspaceNewton::SelectionStrategy strategy = SubspaceNewton::RANDOM_UNIFORM;
  if(strategy_name=="cyclic") strategy=SubspaceNewton::CYCLIC;
  else if(strategy_name=="greedy") strategy=SubspaceNewton::GREEDY;

  // results
  vector<TestResult> all_results;

  // Instantiate solver
  SubspaceNewton solver;
  solver.set_max_iterations(max_iterations);
  solver.set_tolerance(tolerance);
  solver.set_block_size(block_size);
  solver.set_verbose_level(verbose_mode ? 2 : 0);
  solver.enable_line_search(use_line_search);
  solver.set_strategy(strategy);

  // Loop test suite
  for(size_t test_idx=0; test_idx<nonlinear_system_tests.size(); ++test_idx) {

    if(!verbose_mode) print_progress(test_idx, nonlinear_system_tests.size());

    NonlinearSystem * system = nonlinear_system_tests[test_idx];

    vector<SubspaceNewton::Vector> initial_points;
    system->initial_points(initial_points);

    if(initial_points.empty()) {
      TestResult r;
      r.test_name     = system->title();
      r.num_equations = system->num_equations();
      all_results.push_back(r);
      continue;
    }

    for(size_t ip=0; ip<initial_points.size(); ++ip) {
      SubspaceNewton::Vector x = initial_points[ip];

      tm.tic();
      bool converged = solver.solve(*system, x);
      tm.toc();

      TestResult r;
      r.test_name       = system->title();
      r.num_equations   = system->num_equations();
      r.converged       = converged;
      r.iterations      = solver.get_iterations();
      r.function_evals  = solver.get_function_evals();
      r.jacobian_evals  = solver.get_jacobian_evals();
      r.final_residual  = solver.final_residual();
      r.elapsed_time_ms = tm.elapsed_ms();
      r.initial_point_index = ip;

      all_results.push_back(r);
    }
  }

  if(!verbose_mode) print_progress(nonlinear_system_tests.size(),nonlinear_system_tests.size());
  fmt::print(fg(fmt::color::green)|fmt::emphasis::bold,"\n\nAll tests completed!\n");

  print_summary_table(all_results);
  print_statistics(all_results);

  for(auto* sys: nonlinear_system_tests) delete sys;
  fmt::print(fg(fmt::color::cyan),"\nTest suite completed successfully.\n\n");
  return 0;
}
