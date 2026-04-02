#include "tiny_llm/tokenizers/huggingface/tokenizers.hpp"
#include "tiny_llm/utils/runfile.hpp"
#include "gtest/gtest.h"

namespace tiny_llm {
TEST(Tokenizers, Huggingface) {
  TokenizerWrapper wrapper(HuggingfaceTokenizer(utils::BazelRunfile::RLocation(
      "tiny_llm/tests/datas/bert_base_uncased_tokenizer.json")));
  EXPECT_EQ(wrapper.get_vocab_size(), 30522);
  std::string text("replace me by any text you'd like.");

  {
    auto encode_res = wrapper.encode(text, true);
    std::vector<uint32_t> encode_target{101,  5672, 2033, 2011, 2151, 3793,
                                        2017, 1005, 1040, 2066, 1012, 102};
    EXPECT_EQ(encode_res.size(), encode_target.size());
    for (size_t i = 0, i_end = encode_res.size(); i < i_end; ++i) {
      EXPECT_EQ(encode_res[i], encode_target[i]);
    }

    {
      auto decode_res = wrapper.decode(encode_res, true);
      EXPECT_TRUE(decode_res == "replace me by any text you ' d like.");
    }
    {
      auto decode_res = wrapper.decode(encode_res, false);
      EXPECT_TRUE(decode_res ==
                  "[CLS] replace me by any text you ' d like. [SEP]");
    }
  }
  {
    auto encode_res = wrapper.encode(text, false);
    std::vector<uint32_t> encode_target{5672, 2033, 2011, 2151, 3793,
                                        2017, 1005, 1040, 2066, 1012};
    EXPECT_EQ(encode_res.size(), encode_target.size());
    for (size_t i = 0, i_end = encode_res.size(); i < i_end; ++i) {
      EXPECT_EQ(encode_res[i], encode_target[i]);
    }

    {
      auto decode_res = wrapper.decode(encode_res, true);
      EXPECT_TRUE(decode_res == "replace me by any text you ' d like.");
    }
    {
      auto decode_res = wrapper.decode(encode_res, false);
      EXPECT_TRUE(decode_res == "replace me by any text you ' d like.");
    }
  }
}
} // namespace tiny_llm
