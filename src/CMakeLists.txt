include(cc_library)

add_subdirectory(cuda)

cc_library(
  NAME
    llmx
  SRCS
    utility.cpp
    tensor.cpp
    scalar.cu
  DEPS
    llmx_cu_attention
)


