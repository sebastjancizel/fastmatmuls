add_executable(
  matrix_test
  matrix_tests.cpp
)
target_link_libraries(
  matrix_test
  matrix_lib
  GTest::gtest
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(matrix_test)

