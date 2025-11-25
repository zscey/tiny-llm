#pragma once

#define TINY_LLM_DELETE_COPY_MOVE(ClassName)                                   \
  ClassName(const ClassName &) = delete;                                       \
  ClassName(ClassName &&) = delete;                                            \
  auto operator=(const ClassName &)->ClassName & = delete;                     \
  auto operator=(ClassName &&)->ClassName & = delete;
