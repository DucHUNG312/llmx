#pragma once

#include "llmx/cuda/attention/attention_api.h"
#include "llmx/tensor.h"
#include "llmx/utility.h"
#include <any>
#include <cuda_runtime_api.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llmx {

using TensorDict = std::unordered_map<std::string, Tensor *>;
using Parameters = std::unordered_map<std::string, std::any>;

struct OperatorBase {
public:
  virtual void pre_run(TensorDict &data, const Parameters &params) {}
  virtual void post_run(TensorDict &data, const Parameters &params) {}
  virtual void run(TensorDict &data, const Parameters &params) = 0;
};

struct CudaToFloat32 : public OperatorBase {
  void pre_run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    output->dtype(ScalarType::kFP32);
    output->resize(input->dims());
    output->allocate();
  }

  void run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    llmx_copy(input->dtype(), output->dtype(), input->data().ptr(),
              output->data().ptr(), input->numel());
  }
};

struct CudaToFloat16 : public OperatorBase {
  void pre_run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    output->dtype(ScalarType::kFP16);
    output->resize(input->dims());
    output->allocate();
  }

  void run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    llmx_copy(input->dtype(), output->dtype(), input->data().ptr(),
              output->data().ptr(), input->numel());
  }
};

struct CudaToBFloat16 : public OperatorBase {
  void pre_run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    output->dtype(ScalarType::kBF16);
    output->resize(input->dims());
    output->allocate();
  }

  void run(TensorDict &datas, const Parameters &params) override {
    auto *input = datas["input"];
    auto *output = datas["output"];
    llmx_copy(input->dtype(), output->dtype(), input->data().ptr(),
              output->data().ptr(), input->numel());
  }
};

struct CudaAttentionHalf : public OperatorBase {
  void pre_run(TensorDict &datas, const Parameters &params) override {
    auto *q = datas["query"];
    auto *k = datas["key"];
    auto *v = datas["value"];
    auto *output = datas["output"];

    LLMX_CHECK(q->dtype() == ScalarType::kFP16 ||
               q->dtype() == ScalarType::kBF16);
    LLMX_CHECK(k->dtype() == ScalarType::kFP16 ||
               k->dtype() == ScalarType::kBF16);
    LLMX_CHECK(v->dtype() == ScalarType::kFP16 ||
               v->dtype() == ScalarType::kBF16);
    LLMX_CHECK(q->ndim() == 3); // [q_len, n_heads, head_dim]
    LLMX_CHECK(k->ndim() == 3); // [kv_len, n_kv_heads, head_dim]
    LLMX_CHECK(v->ndim() == 3); // [kv_len, n_kv_heads, head_dim]
    LLMX_CHECK(q->dim(2) == k->dim(2));
    LLMX_CHECK(q->dim(2) == v->dim(2));

    output->dtype(q->dtype());
    output->resize(q->dims());
    output->allocate();
  }

  void run(TensorDict &datas, const Parameters &params) override {
    auto *q = datas["query"];
    auto *k = datas["key"];
    auto *v = datas["value"];
    auto *output = datas["output"];
    std::optional<Tensor> alibi_slopes =
        std::any_cast<std::optional<Tensor>>(params.at("alibi_slopes"));
    float sm_scale = std::any_cast<float>(params.at("sm_scale"));
    int sliding_window = std::any_cast<int>(params.at("sliding_window"));
    float logits_soft_cap = std::any_cast<float>(params.at("logits_soft_cap"));
    llmx_single_mha_attention(*output, *q, *k, *v, alibi_slopes, sm_scale,
                              sliding_window, logits_soft_cap);
  }
};

struct ExecutorImpl {
  ExecutorImpl(Device device) : device(device) {}

  virtual ~ExecutorImpl() {
    for (auto it : ops) {
      delete it.second;
      it.second = nullptr;
    }
  }

  virtual void pre_run(const std::string &op_name, TensorDict &data,
                       const Parameters &params) {
    ops[op_name]->pre_run(data, params);
  }
  virtual void post_run(const std::string &op_name, TensorDict &data,
                        const Parameters &params) {
    ops[op_name]->post_run(data, params);
  }
  virtual void run(const std::string &op_name, TensorDict &data,
                   const Parameters &params) {
    ops[op_name]->run(data, params);
  }

  bool can_run(const std::string &op_name) { return ops.contains(op_name); }

  std::unordered_map<std::string, OperatorBase *> ops;
  Device device;
};

struct CpuExecutorImpl : public ExecutorImpl {
  CpuExecutorImpl(int id) : ExecutorImpl(Device(DeviceType::kCPU)) {}
};

struct CudaExecutorImpl : public ExecutorImpl {
  CudaExecutorImpl(int id) : ExecutorImpl(Device(DeviceType::kCUDA, id)) {
    ops["to_fp32_op"] = new CudaToFloat32();
    ops["to_fp16_op"] = new CudaToFloat16();
    ops["to_bf16_op"] = new CudaToBFloat16();
    // single batch, multi-head attention
    ops["attention_haft_op"] = new CudaAttentionHalf();
  }
};

class Executor {
public:
  Executor() {
    // device_executors_.push_back(new ExecutorImpl(Device(DeviceType::kCPU)));
    int n_cuda_device = cuda_device_count();
    for (int i = 0; i < n_cuda_device; i++) {
      device_executors_.push_back(new CudaExecutorImpl(i));
    }
  }

  ~Executor() {
    for (auto *executor : device_executors_) {
      delete executor;
    }
  }

  void run(const std::string &op_name, TensorDict data,
           const Parameters &params) {
    // find suitable device
    ExecutorImpl *executor = nullptr;
    for (auto *exe : device_executors_) {
      if (exe->can_run(op_name)) {
        for (auto &it : data) {
          if (it.second != nullptr) {
            if (it.first == "output") {
              it.second->to_device(exe->device, false);
            } else {
              if (it.second->is_valid()) {
                // copy to device
                it.second->to_device(exe->device);
              } else {
                LLMX_CHECK(false,
                           "Require  input tensor have to be valid to execute");
              }
            }
          }
        }
        executor = exe;
        break;
      }
    }
    if (executor == nullptr) {
      LLMX_CHECK(false, "Could not find suitable device to execute operator "
                            << op_name);
    }
    executor->pre_run(op_name, data, params);
    executor->run(op_name, data, params);
    executor->post_run(op_name, data, params);
  }

private:
  std::vector<ExecutorImpl *> device_executors_;
};

static Executor *get_op_executor() {
  static Executor executor{};
  return &executor;
}

inline void to_fp32(Tensor &output, const Tensor &input) {
  get_op_executor()->run(
      "to_fp32_op",
      {{"input", (Tensor *)&input}, {"output", (Tensor *)&output}}, {});
}
inline void to_fp16(Tensor &output, const Tensor &input) {
  get_op_executor()->run(
      "tto_fp16_op",
      {{"input", (Tensor *)&input}, {"output", (Tensor *)&output}}, {});
}
inline void to_bf16(Tensor &output, const Tensor &input) {
  get_op_executor()->run(
      "to_bf16_op",
      {{"input", (Tensor *)&input}, {"output", (Tensor *)&output}}, {});
}
inline void embedding(Tensor &output, const Tensor &input,
                      const Tensor &weight) {
  get_op_executor()->run("embedding_op",
                         {{"input", (Tensor *)&input},
                          {"weight", (Tensor *)&weight},
                          {"output", (Tensor *)&output}},
                         {});
}

} // namespace llmx