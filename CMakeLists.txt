############################################################################
#                                                                          #
#  file: CMakeLists.txt                                                    #
#  Copyright (C) 2016                                                      #
#      Enrico Bertolazzi                                                   #
#      Dipartimento di Ingegneria Industriale                              #
#      Università degli Studi di Trento                                    #
#      email: enrico.bertolazzi@ing.unitn.it                               #
#                                                                          #
############################################################################

cmake_minimum_required(VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
  cmake_policy(SET CMP0177 NEW)
endif()

project( UtilsLite )

if (IS_DIRECTORY "${PROJECT_SOURCE_DIR}/cmake_utils")
  set( HOME "${PROJECT_SOURCE_DIR}/cmake_utils/" )
else()
  if (IS_DIRECTORY "${PROJECT_SOURCE_DIR}/../cmake_utils")
    set( HOME "${PROJECT_SOURCE_DIR}/../cmake_utils/" )
  else()
    set( HOME "${PROJECT_SOURCE_DIR}/../../cmake_utils/" )
  endif()
endif()

include( "${HOME}/CMakeLists-common.txt" )

project(
  ${PROJECT_NAME}
  VERSION ${UTILS_PROJECT_VERSION}
  HOMEPAGE_URL "https://ebertolazzi.github.io/UtilsLite/"
)

include( "${HOME}/CMakeLists-cflags.txt" )
include( "${HOME}/CMakeLists-utilities.txt" )

if( CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang" )
  add_compile_options( "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-poison-system-directories>" )
  add_compile_options( "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-switch-default>" )
  add_compile_options( "$<$<COMPILE_LANGUAGE:C,CXX>:-Wno-exit-time-destructors>" )
endif()

# Evaluating the suffix to append to target built by this CMake file
utils_artifacts_suffix(ARTIFACTS_STATIC_SUFFIX TRUE)
if( UTILS_BUILD_SHARED )
  utils_artifacts_suffix(ARTIFACTS_DYNAMIC_SUFFIX FALSE)
endif()

if( EXISTS "${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt" )
  include(${CMAKE_CURRENT_LIST_DIR}/../CMakeLists-customize.txt)
endif()

message( STATUS "Compiler used: ${CMAKE_CXX_COMPILER_ID}" )
message( STATUS "BASE SUFFIXES = ${ARTIFACTS_STATIC_SUFFIX} ${ARTIFACTS_DYNAMIC_SUFFIX}" )

#   ____                                _     _ _
#  |  _ \ _ __ ___ _ __ ___  __ _ _   _(_)___(_) |_ ___
#  | |_) | '__/ _ \ '__/ _ \/ _` | | | | / __| | __/ _ \
#  |  __/| | |  __/ | |  __/ (_| | |_| | \__ \ | ||  __/
#  |_|   |_|  \___|_|  \___|\__, |\__,_|_|___/_|\__\___|
#                              |_|
# MUST be executed before the build of the header list
message( STATUS "Execute rake install for ThirdParties" )
if (UNIX)
  execute_process(
    COMMAND bash "-c" "rake install_3rd"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ECHO STDOUT
  )
else()
  execute_process(
    COMMAND cmd /c "rake install_3rd"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ECHO STDOUT
  )
endif()

#   ___
#  / __| ___ _  _ _ _ __ ___ ___
#  \__ \/ _ \ || | '_/ _/ -_|_-<
#  |___/\___/\_,_|_| \__\___/__/
#
set( SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )

# Usa CONFIGURE_DEPENDS per aggiornare automaticamente i GLOB
set( SOURCES )
file(
  GLOB S CONFIGURE_DEPENDS
  ${SOURCE_DIR}/*.cc
  ${SOURCE_DIR}/**/*.cc
)
foreach(F ${S})
  file( RELATIVE_PATH RF ${SOURCE_DIR} "${F}" )
  list( APPEND SOURCES ${RF} )
endforeach(F ${S})

#   _  _             _
#  | || |___ __ _ __| |___ _ _ ___
#  | __ / -_) _` / _` / -_) '_(_-<
#  |_||_\___\__,_\__,_\___|_| /__/
#
set( PUBLIC_HEADERS )
file(
  GLOB_RECURSE S CONFIGURE_DEPENDS
  ${SOURCE_DIR}/*.h
  ${SOURCE_DIR}/*.hh
  ${SOURCE_DIR}/*.hpp
  ${SOURCE_DIR}/*.hxx
  ${SOURCE_DIR}/*.h++
)
foreach (F ${S})
  file( RELATIVE_PATH RF "${SOURCE_DIR}" "${F}" )
  list( APPEND PUBLIC_HEADERS ${RF} )
endforeach(F ${S})

# Filtra file indesiderati
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "\\\.DS_Store$")
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "\\\.gitkeep$")
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "^.*\\\.swp$")
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "^.*\\\.swo$")
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "cmake-build-.*")
list(FILTER PUBLIC_HEADERS EXCLUDE REGEX "build/.*")

# Debug: stampa gli header trovati
message(STATUS "Public headers found (${PUBLIC_HEADERS}):")
foreach(H ${PUBLIC_HEADERS})
    message(STATUS "  - ${H}")
endforeach()

# Verifica specifica per Utils_MADS.hh
if(NOT "Utils_MADS.hh" IN_LIST PUBLIC_HEADERS)
    message(WARNING "Utils_MADS.hh not found in public headers!")
    # Cerca specificamente Utils_MADS.hh
    file(GLOB_RECURSE MADS_HEADER CONFIGURE_DEPENDS
        ${SOURCE_DIR}/**/Utils_MADS.hh
    )
    if(MADS_HEADER)
        file(RELATIVE_PATH MADS_REL "${SOURCE_DIR}" "${MADS_HEADER}")
        list(APPEND PUBLIC_HEADERS "${MADS_REL}")
        message(STATUS "Added missing Utils_MADS.hh: ${MADS_REL}")
    else()
        message(FATAL_ERROR "Utils_MADS.hh not found in source tree!")
    endif()
else()
    message(STATUS "Utils_MADS.hh found in public headers")
endif()

# Crea lista completa di path assoluti per gli header
set(PUBLIC_HEADERS_FULL)
foreach(H ${PUBLIC_HEADERS})
    list(APPEND PUBLIC_HEADERS_FULL "${SOURCE_DIR}/${H}")
endforeach()

# Crea directory di inclusione principali
set(INCLUDE_DIRS 
    ./src 
    ./src/Utils
    ./src/Utils/3rd
)

# Aggiungi tutte le sottodirectory di Utils
file(GLOB_RECURSE UTILS_SUBDIRS CONFIGURE_DEPENDS
    ${SOURCE_DIR}/Utils/*/
)
foreach(DIR ${UTILS_SUBDIRS})
    if(IS_DIRECTORY ${DIR})
        list(APPEND INCLUDE_DIRS ${DIR})
    endif()
endforeach()

# Rimuovi duplicati
list(REMOVE_DUPLICATES INCLUDE_DIRS)

include_directories( ${INCLUDE_DIRS} )

#   _____                  _
#  |_   _|_ _ _ _ __ _ ___| |_ ___
#    | |/ _` | '_/ _` / -_)  _(_-<
#    |_|\__,_|_| \__, \___|\__/__/
#                |___/
#
# define target - STATIC
utils_setup_target(
  ${PROJECT_NAME}
  TRUE # build static
  "${SOURCE_DIR}"
  "${SOURCES}"
  "${SOURCE_DIR}"
  "${PUBLIC_HEADERS}"
)

# Imposta esplicitamente gli header pubblici per Xcode per la libreria statica
if(TARGET ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static)
    target_sources(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static 
        PUBLIC 
        ${PUBLIC_HEADERS_FULL}
    )
    set_target_properties(${UTILS_NAMESPACE}_${PROJECT_NAME}_Static PROPERTIES
        XCODE_ATTRIBUTE_PUBLIC_HEADERS "${PUBLIC_HEADERS_FULL}"
        XCODE_ATTRIBUTE_HEADER_SEARCH_PATHS "${SOURCE_DIR}"
    )
endif()

# define target - SHARED
if ( UTILS_BUILD_SHARED )
  utils_setup_target(
    ${PROJECT_NAME}
    FALSE # build dynamic
    "${SOURCE_DIR}"
    "${SOURCES}"
    "${SOURCE_DIR}"
    "${PUBLIC_HEADERS}"
  )
  
  # Imposta esplicitamente gli header pubblici per Xcode per la libreria shared
  if(TARGET ${UTILS_NAMESPACE}_${PROJECT_NAME})
      target_sources(${UTILS_NAMESPACE}_${PROJECT_NAME} 
          PUBLIC 
          ${PUBLIC_HEADERS_FULL}
      )
      set_target_properties(${UTILS_NAMESPACE}_${PROJECT_NAME} PROPERTIES
          XCODE_ATTRIBUTE_PUBLIC_HEADERS "${PUBLIC_HEADERS_FULL}"
          XCODE_ATTRIBUTE_HEADER_SEARCH_PATHS "${SOURCE_DIR}"
      )
  endif()
endif()

#   _____       _
#  |_   _|__ __| |_ ___
#    | |/ -_|_-<  _(_-<
#    |_|\___/__/\__/__/
#
if ( UTILS_ENABLE_TESTS )

  enable_testing()

  set( EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin )
  set( LIBS ${UTILS_NAMESPACE}_${PROJECT_NAME}_Static ${CMAKE_DL_LIBS} )

  if( UNIX )
    if ( NOT APPLE )
      set( THREADS_PREFER_PTHREAD_FLAG ON )
      find_package( Threads REQUIRED )
      set( LIBS ${LIBS} Threads::Threads )
    endif()
  else()
    set( LIBS ${LIBS} ws2_32 iphlpapi kernel32 )
    if (UTILS_BUILD_SHARED)
      target_link_libraries( ${UTILS_NAMESPACE}_${PROJECT_NAME} ${LIBS} ${CMAKE_DL_LIBS} )
    endif()
  endif()

  set( EXELISTCPP )
  file(
    GLOB S CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src_tests/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src_tests/**/*.cc
  )
  foreach(F ${S})
    file( RELATIVE_PATH RF "${CMAKE_CURRENT_SOURCE_DIR}/src_tests/" "${F}" )
    get_filename_component( RFF ${RF} NAME_WLE )
    list( APPEND EXELISTCPP ${RFF} )
  endforeach(F ${S})

  add_custom_target( "${PROJECT_NAME}_all_tests" ALL )

  foreach( S ${EXELISTCPP} )
    add_executable( ${S} ${CMAKE_CURRENT_SOURCE_DIR}/src_tests/${S}.cc )
    set_target_properties( ${S} PROPERTIES SUFFIX ".exe" )
    target_link_libraries( ${S} ${LIBS} )
    target_include_directories( ${S} PRIVATE ${INCLUDE_DIRS} )
    add_test( NAME "${S}" COMMAND ./bin/${S}.exe WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
    add_dependencies( "${PROJECT_NAME}_all_tests" ${S} )
    
    # Imposta proprietà per Xcode per gli eseguibili di test
    set_target_properties( ${S} PROPERTIES
      XCODE_ATTRIBUTE_HEADER_SEARCH_PATHS "${SOURCE_DIR}"
    )
  endforeach()

endif()

#   ___         _        _ _
#  |_ _|_ _  __| |_ __ _| | |
#   | || ' \(_-<  _/ _` | | |
#  |___|_||_/__/\__\__,_|_|_|
#
install(
  TARGETS ${TARGETS_TO_BE_INSTALLED}
  LIBRARY DESTINATION ${INSTALL_PREFIX}/dll
  ARCHIVE DESTINATION ${INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${INSTALL_PREFIX}/bin
  PUBLIC_HEADER DESTINATION ${PROJECT_PUBLIC_HEADER_DESTINATION}
  EXCLUDE_FROM_ALL
)

# Installazione aggiuntiva degli header per sicurezza
install(
  FILES ${PUBLIC_HEADERS_FULL}
  DESTINATION ${PROJECT_PUBLIC_HEADER_DESTINATION}
  EXCLUDE_FROM_ALL
)

#   ___         _
#  | _ \__ _ __| |____ _ __ _ ___
#  |  _/ _` / _| / / _` / _` / -_)
#  |_| \__,_\__|_\_\__,_\__, \___|
#                       |___/
#

include( "${HOME}/CMakeLists-cpack.txt" )

utils_final_messages()

#
# Remove warning for 3rd
#
get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)

foreach(target IN LISTS all_targets)
  if (TARGET ${target})
    get_target_property(type ${target} TYPE)
    if(type STREQUAL "EXECUTABLE" OR
       type STREQUAL "STATIC_LIBRARY" OR
       type STREQUAL "SHARED_LIBRARY" OR
       type STREQUAL "MODULE_LIBRARY" OR
       type STREQUAL "OBJECT_LIBRARY")
      target_include_directories(${target} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/src/Utils/3rd)
      target_include_directories(${target} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/lib/include/Utils/3rd)
    endif()
  endif()
endforeach()
