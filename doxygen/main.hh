/*!
 * \defgroup TwoD Two-Dimensional Objects
 *
 * \brief Classes and functions for 2D geometric manipulation.
 *
 * This module offers a set of tools to manage, manipulate, and analyze
 * two-dimensional geometric objects, such as points, lines, shapes, and
 * grids. It also includes data structures tailored for efficient 2D
 * geometry processing, allowing for operations like transformations,
 * intersection testing, and boundary detection.
 *
 */

/*!
 * \defgroup Malloc Memory Allocation Utilities
 *
 * \brief Custom memory management and optimization utilities.
 *
 * This module offers specialized classes and functions for handling memory
 * allocation, deallocation, and optimization in performance-critical
 * applications. It provides fine-grained control over memory management
 * to ensure efficient use of resources, including custom allocators and
 * techniques for reducing memory fragmentation.
 */

/*!
 * \defgroup Zeros Zero-Finding Algorithms
 *
 * \brief Algorithms for root-finding in mathematical functions.
 *
 * This module implements a variety of efficient algorithms to compute the
 * roots (zeros) of scalar functions, i.e., points where a function equals
 * zero. It supports techniques ranging from simple bisection and secant
 * methods to more advanced techniques like Newton-Raphson and Brent's method
 * for solving nonlinear equations.
 */

/*!
 * \defgroup Minimize Minimization Algorithms
 * \brief Algorithms for finding local and global minima of functions.
 *
 * This module includes classes and functions for minimizing mathematical
 * functions, offering both local and global minimization techniques. These
 * algorithms are useful in various contexts, including optimization
 * problems, curve fitting, and machine learning applications.
 *
 */

/*!
 * \defgroup Mex Mex Support
 * \brief Utilities for interfacing C++ with MATLAB using the MEX API.
 *
 * This module provides functions and utilities for seamless integration
 * between C++ code and MATLAB via the MEX (MATLAB Executable) API. It enables
 * the creation, management, and conversion of MATLAB data types to and from
 * C++ structures, allowing developers to leverage C++ performance within
 * MATLAB environments.
 */

/*!
 * \defgroup OS OS Utility Functions
 *
 * \brief Utilities for interacting with the operating system.
 *
 * This module provides a variety of utility functions designed to facilitate
 * interaction with the operating system. It includes functions for managing
 * environment variables, retrieving network details such as IP and MAC
 * addresses, querying system information, and performing common file system
 * operations such as file path manipulation and directory management.
 *
 */

/*!
 * \defgroup UTILS C++ Utilities
 *
 * \brief A collection of utility functions and classes.
 *
 * This module provides various helper functions and classes designed to facilitate
 * common programming tasks. The utilities included in this group aim to improve
 * code reusability and maintainability by providing generic and widely applicable
 * functionalities.
 */

/*!
 * \defgroup THREAD Thread Utilities
 *
 * \brief A collection of utilities for threading.
 *
 * This module provides various utilities for managing threads and executing tasks concurrently.
 * It includes functionalities such as thread pools, task management, and synchronization mechanisms.
 */

/*!

  \mainpage UtilsLite

  ## Introduction

  `UtilsLite` is a set of `C++` classes collected for simplify
  my software developing.

  - [Github repository](https://github.com/ebertolazzi/UtilsLite)
  - \subpage 3rd
  - \subpage Install
  - \subpage License

  \author Enrico Bertolazzi

  - Dipartimento di Ingegneria Industriale
  - Università degli Studi di Trento
  - [personal homepage](https://e.bertolazzi.dii.unitn.it)

  \copyright [The 2-Clause BSD License](https://opensource.org/license/bsd-2-clause)
*/

/*!
  \page 3rd Third party software

  - [{fmt}](https://fmt.dev/latest/index.html)
  - [Eigen](https://eigen.tuxfamily.org)
  - [rang](https://github.com/agauniyal/rang) (local documentation \subpage Rang)

*/

/*!
  \page Install Installation guide

  To compile the library, you'll need:
  - A C++ compiler (e.g., GCC, Clang, or MSVC)
  - [Rake](https://ruby.github.io/rake/), a build automation tool.

  ### Compile the Library

  Open a terminal in the project's root directory and run the following command:

  \code{.bash}
  rake
  \endcode

  This will compile the source files and generate the library files in the following structure:

  \code{.text}
  lib/
    ├── include/  # Header files
    └── lib/      # Compiled library files
  \endcode

  ### Integrate into Your Project

  - **Include Headers**: Add the `include/` directory to your project's include path.
    \code{.bash}
    g++ -I/path/to/lib/include your_project.cpp -o your_project
    \endcode

  - **Link the Library**: Use the `-L` and `-l` flags to link against the library:
    \code{.bash}
    g++ -L/path/to/lib/lib -lyourlibrary your_project.cpp -o your_project
    \endcode

  ### Verify the Installation

  Compile and run your project to verify that it works with the newly linked library.
*/

/*!
  \page License

  \verbinclude "../license.md"
*/

/*!

  \page Rang

  ## rang - Colors for your Terminal

  \note
  This is an adaptation of original rang documentation
  (see [original github repository](https://github.com/agauniyal/rang) ).
  The code now is not header only as the original one
  to avoid windows header conflict in complex projects.

  **Example usage**

  \code{cpp}
  #include "rang.hpp"

  using namespace std;
  using namespace rang;

  int main()
  {
     cout << "Plain old text"
          << style::bold << "Rang styled text!!"
          << style::reset << endl;
  }
  \endcode

  *Rang* uses iostream objects - `cout`/`clog`/`cerr` to apply
  attributes to output text. Since *rang* aims to support both windows and
  unix like systems, it takes care of the os specific details and tries to
  provide a uniform interface. Due to incompatiblities b/w different OS
  versions, not all kinds of attributes are supported on every system so
  rang will try to skip the ones which might produce garbage(instead of
  pushing random ANSI escape codes on your streams). Detection of tty is
  also handled internally so you don’t need to check if application user
  might redirect output to a file.

  -  Need support for non-ansi terminals? Check
     out **[Termdb](https://github.com/agauniyal/termdb)** which
     supports virtually all terminals and their capablities.

     Apart from setting text attributes, you can also ask rang to override
     its default behaviour through these methods

  \code{cpp}
  void rang::setControlMode(rang::control);
  \endcode

  where `rang::control` takes - `control::Auto` - Automatically
  detects whether terminal supports color or not(**Default**) -
  `control::Off` - Turn off colors completely - `control::Force` -
  Force colors even if terminal doesn’t supports them or output is
  redirected to non-terminal

  \code{cpp}
  void rang::setWinTermMode(rang::winTerm);
  \endcode

  where `rang::winTerm` takes - `winTerm::Auto` - Checks for newer
  windows and picks Ansi otherwise falls back to Native(**Default**) -
  `winTerm::Native` - This method is supported in all versions of
  windows but supports less attributes - `winTerm::Ansi` - This method
  is supported in newer versions of windows and supports rich variety of
  attributes

  Supported attributes with their compatiblity are listed below -

  **Text Styles**:

  Code                     | Linux/Win/Others | Old Win
  -------------------------|------------------|--------
  `rang::style::bold`      | yes              |  yes
  `rang::style::dim`       | yes              |  no
  `rang::style::italic`    | yes              |  no
  `rang::style::underline` | yes              |  no
  `rang::style::blink`     | no               |  no
  `rang::style::rblink`    | no               |  no
  `rang::style::reversed`  | yes              |  yes
  `rang::style::conceal`   | maybe            |  yes
  `rang::style::crossed`   | yes              |  no

  **Text Color**:

  Code                     | Linux/Win/Others | Old Win
  -------------------------|------------------|--------
  `rang::fg::black`        |  yes             | yes
  `rang::fg::red`          |  yes             | yes
  `rang::fg::green`        |  yes             | yes
  `rang::fg::yellow`       |  yes             | yes
  `rang::fg::blue`         |  yes             | yes
  `rang::fg::magenta`      |  yes             | yes
  `rang::fg::cyan`         |  yes             | yes
  `rang::fg::gray`         |  yes             | yes

  **Background Color**:

  Code                     | Linux/Win/Others | Old Win
  -------------------------|------------------|--------
  `rang::bg::black`        | yes              | yes
  `rang::bg::red`          | yes              | yes
  `rang::bg::green`        | yes              | yes
  `rang::bg::yellow`       | yes              | yes
  `rang::bg::blue`         | yes              | yes
  `rang::bg::magenta`      | yes              | yes
  `rang::bg::cyan`         | yes              | yes
  `rang::bg::gray`         | yes              | yes

  **Bright Foreground Color**:

  Code                     | Linux/Win/Others | Old Win
  -------------------------|------------------|--------
  `rang::fgB::black`       | yes              | yes
  `rang::fgB::red`         | yes              | yes
  `rang::fgB::green`       | yes              | yes
  `rang::fgB::yellow`      | yes              | yes
  `rang::fgB::blue`        | yes              | yes
  `rang::fgB::magenta`     | yes              | yes
  `rang::fgB::cyan`        | yes              | yes
  `rang::fgB::gray`        | yes              | yes


  **Bright Background Color**:

  Code                     | Linux/Win/Others | Old Win
  -------------------------|------------------|--------
  `rang::bgB::black`       | yes              | yes
  `rang::bgB::red`         | yes              | yes
  `rang::bgB::green`       | yes              | yes
  `rang::bgB::yellow`      | yes              | yes
  `rang::bgB::blue`        | yes              | yes
  `rang::bgB::magenta`     | yes              | yes
  `rang::bgB::cyan`        | yes              | yes
  `rang::bgB::gray`        | yes              | yes


  **Reset Styles/Colors**:

  Code                    | Linux/Win/Others | Old Win
  ------------------------|------------------|--------
  `rang::style::reset`    | yes              | yes
  `rang::fg::reset`       | yes              | yes
  `rang::bg::reset`       | yes              | yes

  --------------

  **My terminal is not detected/gets garbage output!**

  Check your env variable `TERM`’s value. Then open an issue
  [here](https://github.com/agauniyal/rang/issues/new) and make sure to
  mention `TERM`’s value along with your terminal name.

  **Redirecting `cout`/`cerr`/`clog` rdbuf?**

  Rang doesn’t interfere if you try to redirect `cout`/`cerr`/`clog`
  to somewhere else and leaves the decision to the library user.
  Make sure you’ve read this
  [conversation](https://github.com/agauniyal/rang/pull/77#issuecomment-360991652)
  and check out the example code
  [here](https://gist.github.com/kingseva/a918ec66079a9475f19642ec31276a21)

*/