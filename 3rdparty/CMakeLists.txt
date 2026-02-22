include(FetchContent)

add_subdirectory(cutlass)

# gtest
FetchContent_Declare(
    3rdparty_googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest
)
set(BUILD_GMOCK ON CACHE BOOL "Set to OFF to do not build googlemock subproject" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Set to ON to generate the install target" FORCE)
set(gtest_build_samples OFF CACHE BOOL "Set to ON to build examples" FORCE)
set(gtest_build_tests OFF CACHE BOOL "Set to ON to build tests" FORCE)
FetchContent_MakeAvailable(3rdparty_googletest)


