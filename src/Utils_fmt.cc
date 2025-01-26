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

//
// file: Utils_fmt.cc
//

#include "Utils_fmt.hh"

namespace Utils {

  static
  inline
  string
  fmt_fw( string_view L, unsigned width, string_view fill, string_view R )
  { return fmt::format( "{}{{:{}^{}}}{}", L, fill, width, R ); }

  static
  inline
  string
  fmt_fw_L( string_view L, unsigned width, string_view fill, string_view R )
  { return fmt::format( "{}{{:{}<{}}}{}", L, fill, width, R ); }

  string
  fmt_table_top_row( unsigned width, string_view title, string_view fill )
  { return fmt::format( fmt_fw( "┌", width, fill, "┐\n" ), title ); }
  
  string
  fmt_table_middle_row( unsigned width, string_view title, string_view fill )
  { return fmt::format( fmt_fw( "├", width, fill, "┤\n" ), title ); }
  
  string
  fmt_table_bottom_row( unsigned width, string_view title, string_view fill )
  { return fmt::format( fmt_fw( "└", width, fill, "┘\n" ), title ); }
  
  string
  fmt_table_row( unsigned width, string_view title, string_view fill )
  { return fmt::format( fmt_fw( "│", width, fill, "│\n" ), title ); }

  string
  fmt_table_row_L( unsigned width, string_view title, string_view fill )
  { return fmt::format( fmt_fw_L( "│", width, fill, "│\n" ), title ); }

  string
  fmt_table_row( unsigned width, std::initializer_list<string_view> names ) {
    unsigned N{ unsigned(names.size()) };
    unsigned w{ (width+1-N)/N };
    UTILS_ASSERT( w > 3, "fmt_table_row( width={}, ... ) no space to print\n", width );
    string FMT{ fmt::format("│ {{:<{}}} ", w-2 ) };
    string res{""};
    for ( auto & n : names ) res += fmt::format( FMT, n );
    res += string( width+1-N-w*N,' ');
    res += "│\n";
    return res;
  }

  string
  fmt_table_top_row( unsigned width, unsigned N, string_view fill ) {
    unsigned w{ (width+1-N)/N };
    UTILS_ASSERT( w > 3, "fmt_table_row( width={}, ... ) no space to print\n", width );
    string res{""};
    string str{""};
    for ( unsigned i{0}; i < w; ++i ) str += fill;
    for ( unsigned i{0}; i < N; ++i ) { res += i == 0 ? "┌" : "┬"; res += str; }
    for ( unsigned i{width+1-N-w*N}; i > 0; --i ) res += fill;
    res += "┐\n";
    return res;
  }

  string
  fmt_table_middle_row( unsigned width, unsigned N, string_view fill ) {
    unsigned w{ (width+1-N)/N };
    UTILS_ASSERT( w > 3, "fmt_table_row( width={}, ... ) no space to print\n", width );
    string res{""};
    string str{""};
    for ( unsigned i{0}; i < w; ++i ) str += fill;
    for ( unsigned i{0}; i < N; ++i ) { res += i == 0 ? "├" : "┼"; res += str; }
    for ( unsigned i{width+1-N-w*N}; i > 0; --i ) res += fill;
    res += "┤\n";
    return res;
  }

  string
  fmt_table_bottom_row( unsigned width, unsigned N, string_view fill ) {
    unsigned w{ (width+1-N)/N };
    UTILS_ASSERT( w > 3, "fmt_table_row( width={}, ... ) no space to print\n", width );
    string res{""};
    string str{""};
    for ( unsigned i{0}; i < w; ++i ) str += fill;
    for ( unsigned i{0}; i < N; ++i ) { res += i == 0 ? "└" : "┴"; res += str; }
    for ( unsigned i{width+1-N-w*N}; i > 0; --i ) res += fill;
    res += "┘\n";
    return res;
  }

}

//
// EOF: Utils_fmt.cc
//
