# UTILS

A collection of useful code for C++ applications:

- Terminal coloring use code from [here](https://github.com/agauniyal/rang) by Abhinav Gauniyal ([license](http://unlicense.org))

- Stream compression use code from [here](https://github.com/geromueller/zstream-cpp) by Jonathan de Halleux and Gero Müller

- Stream formatting use code from [here](https://fmt.dev) by Victor Zverovich (MIT license)

- Table formatting use code from [here](https://github.com/Bornageek/terminal-table) by Andreas Wilhelm (Apache License, Version 2.0) partially rewritten.

- [Eigen3](https://eigen.tuxfamily.org) a C++ template library for linear algebra.

in addition a TreadPool class, TicToc class for timing, Malloc
class for easy allocation with traking of allocated memory.

- Online doc [here](https://ebertolazzi.github.io/UtilsLite)

## COMPILE AND TEST

using rake (with cmake and ninja)

```
rake
```

to compile using mingw open MSYS2 shell and compile as in unix evironment.
The commanda run `cmake` with the appropriate parameters.

### Compile without the use of rake

Prepare for compilation

```
mkdir build
cd build
```

compile with ninja (reccomended)

```
cmake -G Ninja ..
ninja
```

or use default (makfile in unix)

```
cmake ..
make # or make -j10 for parallel copilation
```

other engines are avaible in windows or unix.