/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2026                                                      |
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

#include "Utils_search_intervals.hh"

// Test utility functions with fmt formatting
template <typename T> void print_array( const char * name, T * arr, int n, const fmt::text_style & style = {} )
{
  fmt::print( style, "{}: [", name );
  for ( int i = 0; i < n; ++i )
  {
    fmt::print( style, "{}", arr[i] );
    if ( i < n - 1 ) fmt::print( style, ", " );
  }
  fmt::print( style, "]\n" );
}

// Color definitions for different test categories
namespace TestColors
{
  constexpr auto SECTION = fmt::emphasis::bold | fg( fmt::color::cyan );
  constexpr auto SUCCESS = fg( fmt::color::green ) | fmt::emphasis::bold;
  constexpr auto WARNING = fg( fmt::color::yellow ) | fmt::emphasis::bold;
  constexpr auto ERROR   = fg( fmt::color::red ) | fmt::emphasis::bold;
  constexpr auto INFO    = fg( fmt::color::blue );
  constexpr auto DATA    = fg( fmt::color::light_gray );
  constexpr auto PASS    = fg( fmt::color::green );
  constexpr auto FAIL    = fg( fmt::color::red );
  // constexpr auto HIGHLIGHT = fg(fmt::color::magenta) | fmt::emphasis::bold;
}  // namespace TestColors

// Test runner class with colored output
class TestRunner
{
private:
  int passed_tests = 0;
  int failed_tests = 0;
  int total_tests  = 0;

public:
  void startSection( const std::string & section_name )
  {
    fmt::print( "\n" );
    fmt::print( fmt::emphasis::bold | fg( fmt::color::white ), "{}\n", std::string( 60, '=' ) );
    fmt::print( TestColors::SECTION, "{} {}\n", "▶", section_name );
    fmt::print( fmt::emphasis::bold | fg( fmt::color::white ), "{}\n\n", std::string( 60, '=' ) );
  }

  void startSubSection( const std::string & subsection_name )
  {
    fmt::print( TestColors::INFO, "  • {}\n", subsection_name );
  }

  template <typename Func> bool runTest( const std::string & test_name, Func test_func )
  {
    total_tests++;
    fmt::print( "    [Test {:2d}] {:.<40}", total_tests, test_name );

    try
    {
      test_func();
      passed_tests++;
      fmt::print( TestColors::PASS, "✓ PASS\n" );
      return true;
    }
    catch ( const std::exception & e )
    {
      failed_tests++;
      fmt::print( TestColors::FAIL, "✗ FAIL\n" );
      fmt::print( TestColors::ERROR, "      Error: {}\n", e.what() );
      return false;
    }
    catch ( ... )
    {
      failed_tests++;
      fmt::print( TestColors::FAIL, "✗ FAIL\n" );
      fmt::print( TestColors::ERROR, "      Unknown error\n" );
      return false;
    }
  }

  void printResults()
  {
    fmt::print( "\n" );
    fmt::print( fmt::emphasis::bold | fg( fmt::color::white ), "{}\n", std::string( 60, '=' ) );
    fmt::print( fmt::emphasis::bold, "TEST SUMMARY:\n" );
    fmt::print( "  Total Tests: {}\n", total_tests );

    if ( passed_tests > 0 ) { fmt::print( TestColors::SUCCESS, "  Passed: {} ✓\n", passed_tests ); }

    if ( failed_tests > 0 ) { fmt::print( TestColors::ERROR, "  Failed: {} ✗\n", failed_tests ); }

    double percentage = total_tests > 0 ? ( 100.0 * passed_tests / total_tests ) : 0.0;

    if ( percentage == 100.0 ) { fmt::print( TestColors::SUCCESS, "  Success Rate: {:.1f}% 🎉\n", percentage ); }
    else if ( percentage >= 80.0 ) { fmt::print( TestColors::WARNING, "  Success Rate: {:.1f}%\n", percentage ); }
    else
    {
      fmt::print( TestColors::ERROR, "  Success Rate: {:.1f}%\n", percentage );
    }

    fmt::print( fmt::emphasis::bold | fg( fmt::color::white ), "{}\n", std::string( 60, '=' ) );
  }

  int getExitCode() const { return failed_tests > 0 ? 1 : 0; }
};

// Test cases
void test_basic_functionality( TestRunner & runner )
{
  runner.startSubSection( "Basic Functionality" );

  runner.runTest(
    "Normal value in middle",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 2.5;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );

      fmt::print( TestColors::DATA, "      Array: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]\n" );
      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );

      assert( last_interval == 2 );
    } );

  runner.runTest(
    "Value near left boundary",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 0.3;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );

      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );
      assert( last_interval == 0 );
    } );
}

void test_boundary_conditions( TestRunner & runner )
{
  runner.startSubSection( "Boundary Conditions" );

  runner.runTest(
    "Exact left boundary",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 0.0;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 0 );
    } );

  runner.runTest(
    "Exact right boundary",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 5.0;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 4 );
    } );

  runner.runTest(
    "At internal node",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 2.0;
      int    last_interval = 1;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::DATA, "      x = {}, found interval = {} (should be 2)\n", x, last_interval );
      assert( last_interval == 2 );
    } );
}

void test_closed_range( TestRunner & runner )
{
  runner.startSubSection( "Closed Range (Periodic)" );

  runner.runTest(
    "Value within range",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 2.5;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, true, false );
      assert( last_interval == 2 );
    } );

  runner.runTest(
    "Value below range (wraps around)",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = -2.5;
      int    last_interval = 2;
      double original_x    = x;

      Utils::search_interval( npts, X, x, last_interval, true, false );

      fmt::print( TestColors::DATA, "      Original x = {}, wrapped x = {}\n", original_x, x );
      fmt::print( TestColors::DATA, "      Found interval = {}\n", last_interval );

      assert( x >= 0.0 && x <= 5.0 );
    } );

  runner.runTest(
    "Value above range (wraps around)",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 7.5;
      int    last_interval = 2;
      double original_x    = x;

      Utils::search_interval( npts, X, x, last_interval, true, false );

      fmt::print( TestColors::DATA, "      Original x = {}, wrapped x = {}\n", original_x, x );
      fmt::print( TestColors::DATA, "      Found interval = {}\n", last_interval );

      assert( x >= 0.0 && x <= 5.0 );
    } );
}

void test_can_extend_parameter( TestRunner & runner )
{
  runner.startSubSection( "Can Extend Parameter" );

  runner.runTest(
    "Out of range with can_extend=false",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 6.5;
      int    last_interval = 2;

      bool exception_thrown = false;
      try
      {
        Utils::search_interval( npts, X, x, last_interval, false, false );
      }
      catch ( const std::exception & e )
      {
        exception_thrown = true;
        fmt::print( TestColors::DATA, "      Expected exception: {}\n", e.what() );
      }

      if ( !exception_thrown )
      {
        fmt::print( TestColors::WARNING, "      WARNING: Exception was expected but not thrown!\n" );
        throw std::runtime_error( "Expected exception for out-of-range value" );
      }
    } );

  runner.runTest(
    "Out of range with can_extend=true",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 6.5;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, true );

      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );
      assert( last_interval == 4 );  // Extends to last interval
    } );
}

void test_different_data_types( TestRunner & runner )
{
  runner.startSubSection( "Different Data Types" );

  runner.runTest(
    "Float type",
    []()
    {
      float X[]           = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
      int   npts          = 6;
      float x             = 2.5f;
      int   last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 2 );
    } );

  runner.runTest(
    "32-bit indices",
    []()
    {
      double  X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int32_t npts          = 6;
      double  x             = 2.5;
      int32_t last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 2 );
    } );

  runner.runTest(
    "64-bit indices",
    []()
    {
      double  X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int64_t npts          = 6;
      double  x             = 2.5;
      int64_t last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 2 );
    } );
}

void test_edge_cases( TestRunner & runner )
{
  runner.startSubSection( "Edge Cases" );

  runner.runTest(
    "Minimum array size (2 points)",
    []()
    {
      double X[]           = { 0.0, 1.0 };
      int    npts          = 2;
      double x             = 0.5;
      int    last_interval = 0;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      assert( last_interval == 0 );
    } );

  runner.runTest(
    "Negative values in array",
    []()
    {
      double X[]           = { -5.0, -3.0, -1.0, 1.0, 3.0, 5.0 };
      int    npts          = 6;
      double x             = -0.5;
      int    last_interval = 2;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::DATA, "      Array with negative values\n" );
      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );
      assert( last_interval == 2 );
    } );

  runner.runTest(
    "Very small intervals",
    []()
    {
      double X[]           = { 0.0, 1e-10, 2e-10, 3e-10 };
      int    npts          = 4;
      double x             = 1.5e-10;
      int    last_interval = 1;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::DATA, "      Very small intervals (1e-10)\n" );
      fmt::print( TestColors::DATA, "      x = {:.2e}, found interval = {}\n", x, last_interval );
      assert( last_interval == 1 );
    } );
}

void test_duplicate_values( TestRunner & runner )
{
  runner.startSubSection( "Arrays with Duplicate Values" );

  runner.runTest(
    "Near duplicate values",
    []()
    {
      double X[]           = { 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0 };
      int    npts          = 9;
      double x             = 1.5;
      int    last_interval = 3;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::DATA, "      Array with duplicates: [0, 1, 1, 2, 3, 3, 3, 4, 5]\n" );
      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );
      // Corretto: 1.5 è nell'intervallo [1.0, 2.0] che ha indice 2
      assert( last_interval == 2 );
    } );
  runner.runTest(
    "At duplicate value",
    []()
    {
      double X[]           = { 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0 };
      int    npts          = 9;
      double x             = 3.0;
      int    last_interval = 4;

      Utils::search_interval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::DATA, "      x = {} (duplicate value), found interval = {}\n", x, last_interval );
      // Could be any of intervals containing 3.0
      assert( last_interval >= 4 && last_interval <= 6 );
    } );
}

void test_legacy_wrapper( TestRunner & runner )
{
  runner.startSubSection( "Legacy Wrapper" );

  runner.runTest(
    "searchInterval (legacy wrapper)",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
      int    npts          = 6;
      double x             = 2.5;
      int    last_interval = 2;

      Utils::searchInterval( npts, X, x, last_interval, false, false );
      fmt::print( TestColors::WARNING, "      Using deprecated searchInterval()\n" );
      fmt::print( TestColors::DATA, "      x = {}, found interval = {}\n", x, last_interval );
      assert( last_interval == 2 );
    } );
}

void test_error_conditions( TestRunner & runner )
{
  runner.startSubSection( "Error Conditions" );

  runner.runTest(
    "Invalid npts (npts < 2)",
    []()
    {
      double X[]           = { 0.0 };
      int    npts          = 1;
      double x             = 0.5;
      int    last_interval = 0;

      bool exception_thrown = false;
      try
      {
        Utils::search_interval( npts, X, x, last_interval, false, false );
      }
      catch ( const std::exception & e )
      {
        exception_thrown = true;
        fmt::print( TestColors::DATA, "      Expected exception: {}\n", e.what() );
      }

      if ( !exception_thrown ) { throw std::runtime_error( "Expected exception for npts < 2" ); }
    } );

  runner.runTest(
    "Invalid last_interval (negative)",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0 };
      int    npts          = 4;
      double x             = 1.5;
      int    last_interval = -1;

      bool exception_thrown = false;
      try
      {
        Utils::search_interval( npts, X, x, last_interval, false, false );
      }
      catch ( const std::exception & e )
      {
        exception_thrown = true;
        fmt::print( TestColors::DATA, "      Expected exception: {}\n", e.what() );
      }

      if ( !exception_thrown ) { throw std::runtime_error( "Expected exception for negative last_interval" ); }
    } );

  runner.runTest(
    "Invalid last_interval (too large)",
    []()
    {
      double X[]           = { 0.0, 1.0, 2.0, 3.0 };
      int    npts          = 4;
      double x             = 1.5;
      int    last_interval = 3;  // Should be in [0, npts-2] = [0, 2]

      bool exception_thrown = false;
      try
      {
        Utils::search_interval( npts, X, x, last_interval, false, false );
      }
      catch ( const std::exception & e )
      {
        exception_thrown = true;
        fmt::print( TestColors::DATA, "      Expected exception: {}\n", e.what() );
      }

      if ( !exception_thrown ) { throw std::runtime_error( "Expected exception for out-of-range last_interval" ); }
    } );
}

void test_performance( TestRunner & runner )
{
  runner.startSubSection( "Performance Test" );

  runner.runTest(
    "Large array performance",
    []()
    {
      const int           npts = 10000;
      std::vector<double> X( npts );

      // Create a sorted array
      for ( int i = 0; i < npts; ++i ) { X[i] = static_cast<double>( i ); }

      // Time the searches
      const int num_searches  = 100;
      int       last_interval = npts / 2;
      auto      start_time    = std::chrono::high_resolution_clock::now();

      for ( int i = 0; i < num_searches; ++i )
      {
        double x                = static_cast<double>( rand() % ( npts * 2 ) ) - npts / 4.0;
        int    current_interval = last_interval;

        Utils::search_interval( npts, X.data(), x, last_interval, false, true );
        last_interval = current_interval;  // Reset for next iteration
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time );

      double avg_time = static_cast<double>( duration.count() ) / num_searches;

      fmt::print( TestColors::DATA, "      {} searches in {} μs\n", num_searches, duration.count() );
      fmt::print( TestColors::DATA, "      Average time per search: {:.2f} μs\n", avg_time );

      if ( avg_time > 1000.0 )
      {
        fmt::print( TestColors::WARNING, "      WARNING: Performance might be suboptimal\n" );
      }
    } );
}

void print_banner()
{
  fmt::print(
    fmt::emphasis::bold | fg( fmt::color::cyan ),
    "\n"
    "╔══════════════════════════════════════════════════════════════════╗\n"
    "║                                                                  ║\n"
    "║      🚀 SEARCH_INTERVAL TEST SUITE WITH FMT COLORING 🚀          ║\n"
    "║                                                                  ║\n"
    "╚══════════════════════════════════════════════════════════════════╝\n"
    "\n" );

  fmt::print( fg( fmt::color::light_gray ), "Test Legend:\n" );
  fmt::print( TestColors::PASS, "  ✓ PASS  " );
  fmt::print( fg( fmt::color::light_gray ), " - Test passed successfully\n" );
  fmt::print( TestColors::FAIL, "  ✗ FAIL  " );
  fmt::print( fg( fmt::color::light_gray ), " - Test failed with an error\n" );
  fmt::print( "\n" );
}

int main()
{
  print_banner();

  TestRunner runner;

  try
  {
    runner.startSection( "BASIC FUNCTIONALITY TESTS" );
    test_basic_functionality( runner );

    runner.startSection( "BOUNDARY CONDITION TESTS" );
    test_boundary_conditions( runner );

    runner.startSection( "CLOSED RANGE TESTS" );
    test_closed_range( runner );

    runner.startSection( "CAN EXTEND PARAMETER TESTS" );
    test_can_extend_parameter( runner );

    runner.startSection( "DATA TYPE TESTS" );
    test_different_data_types( runner );

    runner.startSection( "EDGE CASE TESTS" );
    test_edge_cases( runner );

    runner.startSection( "DUPLICATE VALUE TESTS" );
    test_duplicate_values( runner );

    runner.startSection( "LEGACY WRAPPER TESTS" );
    test_legacy_wrapper( runner );

    runner.startSection( "ERROR CONDITION TESTS" );
    test_error_conditions( runner );

    runner.startSection( "PERFORMANCE TESTS" );
    test_performance( runner );

    runner.printResults();

    return runner.getExitCode();
  }
  catch ( const std::exception & e )
  {
    fmt::print( TestColors::ERROR, "\nUnexpected error in test suite: {}\n", e.what() );
    return 1;
  }
  catch ( ... )
  {
    fmt::print( TestColors::ERROR, "\nUnknown error in test suite\n" );
    return 1;
  }
}
