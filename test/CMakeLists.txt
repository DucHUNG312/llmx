include(cc_test)

add_subdirectory(cuda)

cc_test(
  NAME
    tensor_test
  SRCS
    tensor_test.cpp
  DEPS
    llmx
)

cc_test(
  NAME
    ops_test
  SRCS
    ops_test.cpp
  DEPS
    llmx
)



