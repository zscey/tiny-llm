#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/silu.hpp"
#include <numeric>

namespace tiny_llm::cuda {
void SiLUKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                   TensorDesc *const *output_descs) {
  output_descs[0]->dtype = input_descs[0]->dtype;
  output_descs[0]->cur_shape = input_descs[0]->cur_shape;
  element_size = std::accumulate(input_descs[0]->cur_shape.begin(),
                                 input_descs[0]->cur_shape.end(), 1,
                                 [](auto a, auto b) { return a * b; });
}

void SiLUKernel::execute(const void *const *inputs, void *const *outputs,
                         ExecuteContext &ctx) {
  (void)(this);
  silu(static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]),
       element_size, ctx.stream);
}
} // namespace tiny_llm::cuda
