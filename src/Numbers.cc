/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2020                                                      |
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

//
// file: Numbers.cc
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if defined(__llvm__) || defined(__clang__)
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wduplicate-enum"
#pragma clang diagnostic ignored "-Wpoison-system-directories"
#endif

#include "Utils.hh"
#ifndef UTILS_MINIMAL_BUILD
#include "Utils_fmt.hh"
#endif

namespace Utils {

  //============================================================================
  /*    __                       _ _   _       _   _
  //   / _| ___  _   _ _ __   __| | \ | | __ _| \ | |
  //  | |_ / _ \| | | | '_ \ / _` |  \| |/ _` |  \| |
  //  |  _| (_) | |_| | | | | (_| | |\  | (_| | |\  |
  //  |_|  \___/ \__,_|_| |_|\__,_|_| \_|\__,_|_| \_|
  */
  //! check if the vector `pv` os size `DIM` contains only regular floats
  bool
  found_NaN( double const * pv, int const DIM ) {
    for ( int i{0}; i < DIM; ++i )
      if ( !is_finite(pv[i]) )
        return true;
    return false;
  }

  bool
  found_NaN( float const * pv, int const DIM ) {
    for ( int i{0}; i < DIM; ++i )
      if ( !is_finite(pv[i]) )
        return true;
    return false;
  }

  /*       _               _    _   _       _   _
  //   ___| |__   ___  ___| | _| \ | | __ _| \ | |
  //  / __| '_ \ / _ \/ __| |/ /  \| |/ _` |  \| |
  // | (__| | | |  __/ (__|   <| |\  | (_| | |\  |
  //  \___|_| |_|\___|\___|_|\_\_| \_|\__,_|_| \_|
  */

  #define LINE_LINE_LINE_LINE "--------------------------------------------------------------------------------"

#ifndef UTILS_MINIMAL_BUILD
  //! check if the vector `pv` os size `DIM` contains only regular floats. If not an error is issued
  void
  check_NaN(
    double      const pv[],
    string_view const v_name,
    int         const DIM,
    int         const line,
    string_view const file
  ) {
    for ( int i{0}; i < DIM; ++i ) {
      if ( is_infinite(pv[i]) ) {
        UTILS_ERROR(
          "{}\n({}):{}) found Infinity at {}[{}]\n{}\n",
          LINE_LINE_LINE_LINE,
          Utils::get_basename(file), line, v_name, i,
          LINE_LINE_LINE_LINE
        );
      }
      if ( is_NaN(pv[i]) ) {
        UTILS_ERROR(
          "{}\n({}):{}) found NaN at {}[{}]\n{}\n",
          LINE_LINE_LINE_LINE,
          Utils::get_basename(file), line, v_name, i,
          LINE_LINE_LINE_LINE
        );
      }
    }
  }

  void
  check_NaN(
    float       const pv[],
    string_view const v_name,
    int         const DIM,
    int         const line,
    string_view const file
  ) {
    for ( int i = 0; i < DIM; ++i ) {
      if ( is_infinite(pv[i]) ) {
        UTILS_ERROR(
          "{}\n({}):{}) found Infinity at {}[{}]\n{}\n",
          LINE_LINE_LINE_LINE,
          Utils::get_basename(file), line, v_name, i,
          LINE_LINE_LINE_LINE
        );
      }
      if ( is_NaN(pv[i]) ) {
        UTILS_ERROR(
          "{}\n({}):{}) found NaN at {}[{}]\n{}\n",
          LINE_LINE_LINE_LINE,
          Utils::get_basename(file), line, v_name, i,
          LINE_LINE_LINE_LINE
        );
      }
    }
  }
#endif
}

#endif

//
// eof: Numbers.cc
//
