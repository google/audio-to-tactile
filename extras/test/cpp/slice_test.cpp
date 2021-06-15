// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/cpp/slice.h"

#include "src/dsp/logging.h"

// NOLINTBEGIN(readability/check)

namespace audio_tactile {

void TestDynamicSize() {
  puts("TestDynamicSize");
  // Default-construct Slice has null data pointer and zero size.
  Slice<int> slice1;
  CHECK(slice1.data() == nullptr);
  CHECK(slice1.empty());

  int buffer[6] = {0, 1, 2, 3, 4, 5};
  // Make slice1 view the buffer.
  slice1 = Slice<int>(buffer, 6);
  CHECK(slice1.data() == buffer);
  CHECK(!slice1.empty());
  CHECK(slice1.size() == 6);
  CHECK(slice1.size_bytes() == 6 * sizeof(int));
  CHECK(slice1[2] == 2);

  // Changing size should work.
  slice1 = Slice<int, 4>(buffer);
  CHECK(slice1.size() == 4);
  slice1 = Slice<int>(buffer, 0);
  CHECK(slice1.empty());
  slice1 = Slice<int>(buffer, 6);
  CHECK(slice1.size() == 6);

  // Test read-only slice construction from mutable slice.
  Slice<const int> slice2(slice1);
  CHECK(slice2.data() == buffer);
  CHECK(slice2.size() == 6);

  int expected = 0;
  for (int value : slice2) {  // Test that ranged-for works.
    CHECK(value == expected);
    ++expected;
  }
  CHECK(expected == 6);

  Slice<const int> slice3;
  // Test read-only slice assignment from mutable slice.
  slice3 = slice1;
  CHECK(slice3.data() == buffer);
  CHECK(slice3.size() == 6);

  // Changes through slice1 should be visible to slice2.
  slice1[2] = 42;
  CHECK(slice2[2] == 42);
}

void TestFixedSize() {
  puts("TestFixedSize");
  constexpr int kSize = 6;
  // Default-construct Slice has null data pointer and size kSize.
  Slice<int, kSize> slice1;
  CHECK(slice1.data() == nullptr);
  CHECK(slice1.size() == kSize);
  CHECK(!slice1.empty());

  int buffer[6] = {0, 1, 2, 3, 4, 5};
  // Make slice1 view the buffer.
  slice1 = Slice<int, kSize>(buffer);
  CHECK(slice1.data() == buffer);
  CHECK(slice1.size() == kSize);
  CHECK(slice1.size_bytes() == kSize * sizeof(int));
  CHECK(slice1[2] == 2);

  // Attempting to change size should not work.
  slice1 = Slice<int>(buffer, 3);
  CHECK(slice1.size() == kSize);
  slice1 = Slice<int>(buffer, 0);
  CHECK(slice1.size() == kSize);

  int expected = 0;
  for (int value : slice1) {  // Test that ranged-for works.
    CHECK(value == expected);
    ++expected;
  }
  CHECK(expected == kSize);

  // Test dynamically-sized slice construction from fixed-sized slice.
  Slice<int> slice2(slice1);
  CHECK(slice2.data() == buffer);
  CHECK(slice2.size() == kSize);
  // Fixed-sized slices don't store the size, so sizeof() should be smaller.
  CHECK(sizeof(slice1) < sizeof(slice2));

  Slice<const int> slice3;
  // Test dynamically-sized slice assignment from fixed-sized slice.
  slice3 = slice1;
  CHECK(slice3.data() == buffer);
  CHECK(slice3.size() == kSize);
}

void TestSubslices() {
  puts("TestSubslices");
  int buffer[6] = {0, 1, 2, 3, 4, 5};
  Slice<int> slice(buffer, 6);

  auto head1 = slice.head(3);
  CHECK(head1.data() == buffer);
  CHECK(head1.size() == 3);
  CHECK(decltype(head1)::kSizeAtCompileTime == kDynamic);
  auto head2 = slice.head<3>();
  CHECK(head2.data() == buffer);
  CHECK(head2.size() == 3);
  CHECK(decltype(head2)::kSizeAtCompileTime == 3);

  auto tail1 = slice.tail(2);
  CHECK(tail1.data() == buffer + 4);
  CHECK(tail1.size() == 2);
  CHECK(decltype(tail1)::kSizeAtCompileTime == kDynamic);
  auto tail2 = slice.tail<2>();
  CHECK(tail2.data() == buffer + 4);
  CHECK(tail2.size() == 2);
  CHECK(decltype(tail2)::kSizeAtCompileTime == 2);

  auto segment1 = slice.segment(1, 4);
  CHECK(segment1.data() == buffer + 1);
  CHECK(segment1.size() == 4);
  CHECK(decltype(segment1)::kSizeAtCompileTime == kDynamic);
  auto segment2 = slice.segment<4>(1);
  CHECK(segment2.data() == buffer + 1);
  CHECK(segment2.size() == 4);
  CHECK(decltype(segment2)::kSizeAtCompileTime == 4);
}

void TestCopyFrom() {
  puts("TestCopyFrom");
  float a[3] = {1.2f, 3.4f, 5.6f};
  float b[3] = {0.0f, 0.0f, 0.0f};

  Slice<const float, 3> source(a);
  Slice<float> target(b, 2);
  CHECK(!target.CopyFrom(source));  // Copy isn't done because sizes mismatch.
  CHECK(b[0] == 0.0f);
  CHECK(b[1] == 0.0f);
  CHECK(b[2] == 0.0f);

  target = Slice<float>(b, 3);
  CHECK(target.CopyFrom(source));  // Copy is successful.
  CHECK(b[0] == 1.2f);
  CHECK(b[1] == 3.4f);
  CHECK(b[2] == 5.6f);
}

}  // namespace audio_tactile

// NOLINTEND

int main(int argc, char** argv) {
  audio_tactile::TestDynamicSize();
  audio_tactile::TestFixedSize();
  audio_tactile::TestSubslices();
  audio_tactile::TestCopyFrom();

  puts("PASS");
  return EXIT_SUCCESS;
}
