# Builds the AxisForge -> GPD adapter against an unmodified GPD build.
#   cmake -S tools/research/phase7_extension/gpd_adapter -B BUILD -DGPD_SOURCE=/path/to/gpd -DGPD_BUILD=/path/to/gpd-build
cmake_minimum_required(VERSION 3.8)
project(gpd_adapter CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(PCL 1.9 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
add_definitions(${PCL_DEFINITIONS})
include_directories(${GPD_SOURCE}/include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
add_executable(gpd_score_candidates gpd_score_candidates.cpp)
# Same flags as GPD itself: Eigen fixed-size types must have the same alignment on both sides.
target_compile_options(gpd_score_candidates PRIVATE -O3 -march=native -mtune=intel -msse4.2 -mavx2 -mfma -fopenmp -fPIC)
target_link_libraries(gpd_score_candidates ${GPD_BUILD}/libgpd.so ${PCL_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
