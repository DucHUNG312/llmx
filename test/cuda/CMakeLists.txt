include(cc_test)

cc_test(
  NAME
    single_mha_test
  SRCS
    mha_attention_test.cpp
  DEPS
    llmx
)



