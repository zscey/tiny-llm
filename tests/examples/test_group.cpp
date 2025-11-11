#include "fmt/format.h"
#include "gtest/gtest.h"

TEST(ExampleGroup, T0) { EXPECT_EQ(fmt::format("Test0"), "Test0"); }
TEST(ExampleGroup, T1) { EXPECT_EQ(fmt::format("Test1"), "Test1"); }
