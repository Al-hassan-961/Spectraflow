cmake_minimum_required(VERSION 3.16)
project(spectraflow_core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Spectraflow native core (C++17 + pybind11)
#
# Builds:
#   spectraflow_core_lib  - static library with the packet codec, ring buffer
#                           and DSP primitives (link this from C++ tooling)
#   spectraflow_core      - the importable Python extension module
#
# Quick start:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   # or simply:  pip install -e .   (drives this same build via setup.py)
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 ships its CMake package inside the Python site-packages tree, which
# is not on CMake's default search path on every platform (notably Termux).
# Ask the interpreter itself where it lives before giving up.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE SPECTRAFLOW_PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(SPECTRAFLOW_PYBIND11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${SPECTRAFLOW_PYBIND11_DIR}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)
endif()

# The native core is deliberately dependency-free so it cross-compiles cleanly.
add_library(spectraflow_core_lib STATIC
  core/src/csi_packet.cpp
  core/src/ring_buffer.cpp
  core/src/dsp_filters.cpp
)
target_include_directories(spectraflow_core_lib PUBLIC core/include)
target_compile_features(spectraflow_core_lib PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(spectraflow_core_lib PRIVATE /W4)
else()
  target_compile_options(spectraflow_core_lib PRIVATE -Wall -Wextra -Wpedantic)
endif()

pybind11_add_module(spectraflow_core core/bindings.cpp)
target_link_libraries(spectraflow_core PRIVATE spectraflow_core_lib)

# ---------------------------------------------------------------------------
# Android / Termux
#
# Termux's Python executable does not export its symbols to dlopen()ed
# extension modules, so a normally-linked .so fails with:
#   dlopen failed: cannot locate symbol "PyExc_ImportError"
# Linking the extension against libpython explicitly resolves this. Without the
# flag the module builds but cannot be imported.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_library(SPECTRAFLOW_LIBPYTHON
    NAMES python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
    PATHS ${Python3_LIBRARY_DIRS} ${Python3_PREFIX}/lib
  )
  if(SPECTRAFLOW_LIBPYTHON)
    target_link_libraries(spectraflow_core PRIVATE ${SPECTRAFLOW_LIBPYTHON})
    message(STATUS "Spectraflow: linked native core against ${SPECTRAFLOW_LIBPYTHON}")
  else()
    message(WARNING
      "Spectraflow: could not locate libpython${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR} "
      "on Android; the extension may build but fail to import. "
      "The pure-NumPy fallback will be used instead.")
  endif()
endif()

install(TARGETS spectraflow_core_lib ARCHIVE DESTINATION lib)
