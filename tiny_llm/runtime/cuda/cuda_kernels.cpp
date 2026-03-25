#include "tiny_llm/runtime/cuda/cuda_kernels.hpp"
#include "tiny_llm/common/log_and_excepts.hpp"
#include "tiny_llm/ops/cuda/silu.hpp"
#include <numeric>

namespace tiny_llm::cuda {
void SiLUKernel::dtype_shape_infer(const TensorDesc *const *input_descs,
                                   TensorDesc *const *output_descs) {
  const auto *input_desc = input_descs[0];
  TINY_LLM_CHECK(input_desc->dtype == DataType::kFloat32);

  auto *output_desc = output_descs[0];
  output_desc->dtype = input_desc->dtype;
  output_desc->cur_shape = input_desc->cur_shape;
  element_size =
      input_desc->cur_shape.empty()
          ? 0
          : std::accumulate(input_desc->cur_shape.begin(),
                            input_desc->cur_shape.end(), 1,
                            [](auto a, auto b) -> auto { return a * b; });
}

void SiLUKernel::execute(const void *const *inputs,
                         void *const *outputs) const {
  silu(static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]),
       element_size);
}
} // namespace tiny_llm::cuda
