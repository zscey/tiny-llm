#pragma once

#include "tiny_llm/device_managers/buffer.hpp"

namespace tiny_llm {
/**
 * @brief Copy `size` bytes from `src_ptr` on `src_device` to `dst_ptr` on
 * `dst_device`.
 *
 * This function handles cross-device copies (CPU <-> CUDA device,
 * device-to-device, etc.).
 *
 * @param src_ptr Source memory address.
 * @param src_device Source device information.
 * @param dst_ptr Destination memory address.
 * @param dst_device Destination device information.
 * @param size Number of bytes to copy.
 */
void copy_data(const void *src_ptr, Device src_device, void *dst_ptr,
               Device dst_device, std::size_t size);
} // namespace tiny_llm
