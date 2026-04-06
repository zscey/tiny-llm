#include "gflags/gflags.h"
#include "tiny_llm/device_managers/cuda/cuda_device_guard.hpp"
#include "tiny_llm/pipeline/pipeline.hpp"
#include <iostream>

// NOLINTBEGIN
DEFINE_string(model_path, "", "Path to the model directory (required)");
DEFINE_int32(device, 0, "CUDA device ID");
// NOLINTEND

auto main(int argc, char **argv) -> int32_t {
  std::string usage =
      "Usage: " + std::string(argv[0]) + " --model_path <path> [options]";
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    std::cerr << "Error: --path is mandatory.\n\n";
    gflags::ShowUsageWithFlags(argv[0]);
    return 1;
  }

  try {
    tiny_llm::CudaDeviceSwitchGuard guard(FLAGS_device);

    tiny_llm::Pipeline pipeline(
        FLAGS_model_path,
        {.model_type = tiny_llm::ModelType::kTinyLlama,
         .dtype = tiny_llm::DataType::kFloat32,
         .device = {.type = tiny_llm::DeviceType::kCuda,
                    .id = static_cast<tiny_llm::DeviceId>(FLAGS_device)},
         .batch = 1});

    std::cout << "\n>>> Interactive Mode (Multi-line enabled) <<<\n";
    std::cout << ">>> Enter your prompt. Enter a blank line to end the current "
                 "input. Enter EOF (Ctrl+D) to exit. <<<\n";

    while (true) {
      std::string input_buffer;
      std::string line;
      std::cout << "\n[Input] >\n";

      while (std::getline(std::cin, line)) {
        if (line.empty()) {
          break;
        }
        input_buffer += line + "\n";
      }

      if (std::cin.eof()) {
        break;
      }

      if (input_buffer.empty()) {
        std::cout << "The prompt is empty. Please re-enter.\n";
        continue;
      }

      auto result = pipeline.apply({input_buffer});
      std::cout << "[Output Begin]\n"
                << result.at(0).substr(input_buffer.size())
                << "\n[Output End]\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "[Fatal]: " << e.what() << "\n";
    return 1;
  }

  std::cout << "\n";
  gflags::ShutDownCommandLineFlags();
  return 0;
}
