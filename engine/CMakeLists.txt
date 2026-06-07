# CMake build for the chess engine.
#
# Produces these binaries:
#   engine      — the UCI engine that plays games
#   perft       — the move-generation correctness test
#   parity      — the C++-vs-Python eval parity test (shallow judge)
#   parity_deep — the C++-vs-Python parity test for the DEEP residual judge
#
# We compile with -O3 -march=native so the integer brain and search run at full
# speed and use the box's AVX2 instructions.

cmake_minimum_required(VERSION 3.10)
project(OurChessEngine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Shared engine sources (everything except the per-binary main()).
# eval_deep.cpp adds the deep residual judge alongside the shallow eval; the
# engine picks which to use at load time by the weight file's magic header.
set(CORE src/board.cpp src/eval.cpp src/eval_deep.cpp src/eval_king.cpp src/search.cpp)

add_executable(engine ${CORE} src/uci.cpp)
add_executable(perft ${CORE} tests/perft.cpp)
add_executable(parity ${CORE} tests/parity.cpp)
add_executable(parity_deep ${CORE} tests/parity_deep.cpp)
add_executable(parity_king ${CORE} tests/parity_king.cpp)
add_executable(acc_difftest ${CORE} tests/acc_difftest.cpp)
add_executable(evalspeed ${CORE} tests/evalspeed.cpp)
add_executable(searchbench ${CORE} tests/searchbench.cpp)
