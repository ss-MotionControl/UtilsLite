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

#include <iostream>

#include "Utils.hh"

using std::cout;
using std::string;
using std::vector;

int
main()
{
  string const   str = "pippo,pluto paperino;nonna papera,,,zorro";
  string const   sep = " ,;";
  vector<string> res;

  Utils::split_string( str, sep, res );

  cout << "STR: " << str << "\n";
  for ( auto & e : res ) cout << "TOKEN:" << e << "\n";
  cout << "\nAll Done Folks!\n";
  return 0;
}
