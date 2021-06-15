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

#include "src/cpp/std_shim.h"

#include "src/dsp/logging.h"

// NOLINTBEGIN(readability/check)

namespace audio_tactile {

// Checks that T1 and T2 are the same type.
template <typename T1, typename T2>
void CheckSame() {
  const bool is_same = std_shim::is_same<T1, T2>::value;
  CHECK(is_same);
}

// Test std_shim::min.
void TestMin() {
  puts("TestMin");
  CHECK(std_shim::min<int>(5, 5) == 5);
  CHECK(std_shim::min<int>(-15, 8) == -15);
  CHECK(std_shim::min<float>(6.4f, 2.1f) == 2.1f);
}

// Test std_shim::max.
void TestMax() {
  puts("TestMax");
  CHECK(std_shim::max<int>(5, 5) == 5);
  CHECK(std_shim::max<int>(-15, 8) == 8);
  CHECK(std_shim::max<float>(6.4f, 2.1f) == 6.4f);
}

// Test std_shim::is_same.
void TestIsSame() {
  puts("TestIsSame");
  bool is_same = std_shim::is_same<int, int>::value;
  CHECK(is_same);
  is_same = std_shim::is_same<float&, float&>::value;
  CHECK(is_same);

  is_same = std_shim::is_same<int, float>::value;
  CHECK(!is_same);
  is_same = std_shim::is_same<int, const int>::value;
  CHECK(!is_same);
  is_same = std_shim::is_same<int, int&>::value;
  CHECK(!is_same);
  is_same = std_shim::is_same<int&, int&&>::value;
  CHECK(!is_same);
  is_same = std_shim::is_same<int*, int[]>::value;
  CHECK(!is_same);
}

// Test std_shim::is_const.
void TestIsConst() {
  puts("TestIsConst");
  bool is_const = std_shim::is_const<const int>::value;
  CHECK(is_const);
  is_const = std_shim::is_const<int* const>::value;
  CHECK(is_const);
  is_const = std_shim::is_const<const int[]>::value;
  CHECK(is_const);
  is_const = std_shim::is_const<const int[3]>::value;
  CHECK(is_const);

  is_const = std_shim::is_const<int>::value;
  CHECK(!is_const);
  is_const = std_shim::is_const<int*>::value;
  CHECK(!is_const);
  // The pointer itself in `const int*` is non-const, so is_const returns false.
  // This agrees with the standard library's std::is_const.
  is_const = std_shim::is_const<const int*>::value;
  CHECK(!is_const);
  is_const = std_shim::is_const<int[]>::value;
  CHECK(!is_const);
  is_const = std_shim::is_const<int[3]>::value;
  CHECK(!is_const);
}

// Test std_shim::remove_reference.
void TestRemoveReference() {
  puts("TestRemoveReference");
  CheckSame<int, std_shim::remove_reference<int>::type>();
  CheckSame<int, std_shim::remove_reference<int&>::type>();
  CheckSame<int, std_shim::remove_reference<int&&>::type>();

  CheckSame<const int, std_shim::remove_reference<const int>::type>();
  CheckSame<const int, std_shim::remove_reference<const int&>::type>();
  CheckSame<const int, std_shim::remove_reference<const int&&>::type>();
}

// Test std_shim::remove_const.
void TestRemoveConst() {
  puts("TestRemoveConst");
  CheckSame<int, std_shim::remove_const<int>::type>();
  CheckSame<int, std_shim::remove_const<const int>::type>();
  CheckSame<int*, std_shim::remove_const<int* const>::type>();
  CheckSame<int[], std_shim::remove_const<const int[]>::type>();
  CheckSame<int[3], std_shim::remove_const<const int[3]>::type>();
}

// Test std_shim::conditional.
void TestConditional() {
  puts("TestConditional");
  CheckSame<int, std_shim::conditional<true, int, float>::type>();
  CheckSame<float, std_shim::conditional<false, int, float>::type>();
}

// Test std_shim::enable_if. There are quite a few idiomatic ways to use
// enable_if. We test just a couple on a simple function EvenOdd<X>() that
// returns 'E' if X is even or 'O' if X is odd.

// Version A: enable_if through the return type.
template <int X>
typename std_shim::enable_if<X % 2 == 0, char>::type EvenOddVA() { return 'E'; }
template <int X>
typename std_shim::enable_if<X % 2 != 0, char>::type EvenOddVA() { return 'O'; }

// Version B: enable_if through a non-type template parameter.
template <int X, typename std_shim::enable_if<X % 2 == 0, int>::type = 0>
char EvenOddVB() { return 'E'; }
template <int X, typename std_shim::enable_if<X % 2 != 0, int>::type = 0>
char EvenOddVB() { return 'O'; }

void TestEnableIf() {
  puts("TestEnableIf");
  CHECK(EvenOddVA<2>() == 'E');
  CHECK(EvenOddVA<3>() == 'O');

  CHECK(EvenOddVB<2>() == 'E');
  CHECK(EvenOddVB<3>() == 'O');
}

}  // namespace audio_tactile

// NOLINTEND

int main(int argc, char** argv) {
  audio_tactile::TestMin();
  audio_tactile::TestMax();
  audio_tactile::TestIsSame();
  audio_tactile::TestIsConst();
  audio_tactile::TestRemoveReference();
  audio_tactile::TestRemoveConst();
  audio_tactile::TestConditional();
  audio_tactile::TestEnableIf();

  puts("PASS");
  return EXIT_SUCCESS;
}
