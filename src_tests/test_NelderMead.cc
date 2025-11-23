/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  NelderMead_minimizer Test Harness                                       |
 |                                                                          |
 \*--------------------------------------------------------------------------*/

#include "Utils_NelderMead.hh"
#include "Utils_fmt.hh"
#include "ND_func.cxx" 

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <limits>

using Scalar = double;
using NM_Block = Utils::NelderMead_BlockCoordinate<Scalar>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// --- Helper per formattare vettori ---
std::string format_vector(const Vector& v) {
    std::stringstream ss;
    ss << "[";
    if (v.size() <= 6) {
        for (int i = 0; i < v.size(); ++i) {
            ss << fmt::format("{:.4f}", v(i));
            if (i < v.size() - 1) ss << ", ";
        }
    } else {
        for (int i = 0; i < 3; ++i) ss << fmt::format("{:.4f}, ", v(i));
        ss << "... ";
        for (int i = v.size() - 3; i < v.size(); ++i) {
            ss << fmt::format("{:.4f}", v(i));
            if (i < v.size() - 1) ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

struct TestResult {
  std::string problem_name;
  Scalar final_value{0};
  size_t dimension{0};
  size_t total_evaluations{0}; 
  std::string status_str;
};

static std::vector<TestResult> global_results;

// --- Init Sicuro ---
template <typename ProblemFunc>
Vector get_safe_initial_point(ProblemFunc& problem) {
    try { return problem.init(); } catch(...) {}
    Vector L = problem.lower();
    Vector U = problem.upper();
    size_t dim = L.size();
    Vector x0(dim);
    for (size_t i = 0; i < dim; ++i) {
        if (std::isfinite(L(i)) && std::isfinite(U(i))) x0(i) = (L(i) + U(i)) / 2.0;
        else if (std::isfinite(L(i))) x0(i) = L(i) + 1.0;
        else if (std::isfinite(U(i))) x0(i) = U(i) - 1.0;
        else x0(i) = 0.5; 
    }
    return x0;
}

template <typename ProblemFunc>
void test(ProblemFunc& problem, std::string const & name) {
  
  Vector L = problem.lower();
  Vector U = problem.upper();
  size_t dim = L.size();
  Vector x0 = get_safe_initial_point(problem);

  fmt::print("\n");
  fmt::print("################################################################\n");
  fmt::print("# TEST FUNZIONE: {:<45} #\n", name);
  fmt::print("# Dimensione:    {:<45} #\n", dim);
  fmt::print("################################################################\n");

  fmt::print("-> Punto Iniziale: {}\n", format_vector(x0));
  fmt::print("-> Valore Iniziale: {:.6e}\n", problem(x0));

  NM_Block::Options opts;
  opts.block_size = 10; 
  opts.max_outer_iterations = 100; 
  opts.max_function_evaluations = 200000; 
  opts.tolerance = 1e-7;
  opts.verbose = true;
  
  // Opzioni sub-solver (usate come globali se dim < 10)
  opts.sub_options.tolerance = 1e-7;
  opts.sub_options.initial_step = 0.1; 

  NM_Block solver(opts);
  solver.set_bounds(L, U);

  auto result = solver.minimize(x0, [&](Vector const & x) { return problem(x); });

  fmt::print("\n-> STATO FINALE:   {}\n", Utils::status_to_string(result.status));
  fmt::print("-> Punto Finale:   {}\n", format_vector(result.solution));
  fmt::print("-> Valore Finale:  {:.8e}\n", result.final_function_value);
  fmt::print("-> Totale Evals:   {}\n", result.total_evaluations);

  TestResult tr;
  tr.problem_name = name;
  tr.dimension = dim;
  tr.final_value = result.final_function_value;
  tr.status_str = Utils::status_to_string(result.status);
  tr.total_evaluations = result.total_evaluations;
  global_results.push_back(tr);
}

void print_summary_table() {
    if (global_results.empty()) return;
    fmt::print("\n\n");
    fmt::print("╔═══════════════════════════════════════════════════════════════════════╗\n");
    fmt::print("║                           R I E P I L O G O                           ║\n");
    fmt::print("╠════════════════════════╤══════╤══════════╤══════════════╤═════════════╣\n");
    fmt::print("║ Funzione               │ Dim  │ Tot Evals│ Valore Finale│ Status      ║\n");
    fmt::print("╠════════════════════════╪══════╪══════════╪══════════════╪═════════════╣\n");

    for (const auto& r : global_results) {
        fmt::print("║ {:<22} │ {:>4} │ {:>8} │ {:<12.4e} │ {:<11} ║\n", 
                   r.problem_name.substr(0,22), r.dimension, 
                   r.total_evaluations, r.final_value, r.status_str);
    }
    fmt::print("╚════════════════════════╧══════╧══════════╧══════════════╧═════════════╝\n");
}

int
main() {
  try {
    Rosenbrock2D<Scalar> rosen;
    test( rosen, "Rosenbrock2D" );

    RosenbrockN<Scalar,10> rosenN;
    test( rosenN, "Rosenbrock10D" );

    PowellSingularN<Scalar,16> powellN;
    test( powellN, "PowellSingular16D" );

    ExtendedWoodN<Scalar,16> woodN;
    test( woodN, "ExtendedWood16D" );

    // Altri problemi (se presenti in ND_func.cxx)
    Beale2D<Scalar> beale;
    test( beale, "Beale2D" );

    Himmelblau2D<Scalar> himm;
    test( himm, "Himmelblau2D" );

    FreudensteinRoth2D<Scalar> fr;
    test( fr, "FreudensteinRoth2D" );

    HelicalValley3D<Scalar> heli;
    test( heli, "HelicalValley3D" );

    PowellBadlyScaled2D<Scalar> pbs;
    test( pbs, "PowellBadlyScaled2D" );

    BrownAlmostLinearN<Scalar,10> brown;
    test( brown, "BrownAlmostLinear10D" );

    BroydenTridiagonalN<Scalar,12> broy;
    test( broy, "BroydenTridiagonal12D" );

    IllConditionedQuadraticN<Scalar,20> illq;
    test( illq, "IllConditionedQuadratic20D" );

    TrigonometricSumN<Scalar,15> trig;
    test( trig, "TrigonometricSum15D" );

    SchwefelN<Scalar,15> schwefel;
    test( schwefel, "SchwefelN15D" );

    AckleyN<Scalar,15> ackley;
    test( ackley, "AckleyN15D" );

    RastriginN<Scalar,15> rastrigin;
    test( rastrigin, "RastriginN15D" );

    print_summary_table();

  } catch (std::exception const & e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
