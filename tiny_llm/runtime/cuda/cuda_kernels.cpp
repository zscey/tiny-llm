#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/add.hpp"
#include <numeric>
#include <stdexcept>

namespace tiny_llm::cuda {
void AddKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                  TensorDesc *const *output_descs) {
  TINY_LLM_CHECK(input_descs[0]->dtype == input_descs[1]->dtype);
  TINY_LLM_CHECK(input_descs[0]->cur_shape == input_descs[1]->cur_shape);

  output_descs[0]->dtype = input_descs[0]->dtype;
  output_descs[0]->cur_shape = input_descs[0]->cur_shape;
  element_size = std::accumulate(input_descs[0]->cur_shape.begin(),
                                 input_descs[0]->cur_shape.end(), 1,
                                 [](auto a, auto b) { return a * b; });
}

void AddKernel::execute(const void *const *inputs, void *const *outputs,
                        ExecuteContext &ctx) {
  (void)(this);
  add(static_cast<const float *>(inputs[0]),
      static_cast<const float *>(inputs[1]), static_cast<float *>(outputs[0]),
      element_size, ctx.stream);
}
} // namespace tiny_llm::cuda
