include(cc_library)

# copy the file to binary dir in order to force re-configuration when the file changes
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/gen_instantiations.py
  ${CMAKE_CURRENT_BINARY_DIR}/gen_instantiations.py
)

# execute the script to generate the instantiation of the kernels
execute_process(
  COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/gen_instantiations.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
  COMMAND_ERROR_IS_FATAL ANY
)
# globbing all generated files in sub directory "gensrc"
file(GLOB GENERATED_SRC_FILES "${CMAKE_CURRENT_BINARY_DIR}/gensrc/*.cu")

cc_library(
  NAME
    llmx_cu_attention
  SRCS
    attention_api.cpp
    ${GENERATED_SRC_FILES}
  DEPS
    cutlass
)