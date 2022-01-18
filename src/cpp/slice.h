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
//
//
// Slice, a templated type for making non-owning views of existing data.
//
// Slice is a data pointer and a size to describe a contiguous sequence of
// elements. Slice is similar to C++20 std::span, but works in C++11.
//
//   int buffer[6] = {0, 1, 2, 3, 4, 5};
//   Slice<int> slice1(buffer, 6);  // A mutable, dynamically-sized slice.
//   assert(slice1.data() == buffer);
//   assert(slice1.size() == 6);
//   slice1[3] = 42;
//
// A slice of size 0 is allowed (and can be useful) to represent an empty slice.
// For a read-only slice, add `const` in the type template argument:
//
//   Slice<const int> slice2(buffer, 6);  // Read-only, dynamically-sized slice.
//
// For a slice with a compile-time fixed size, add the size as a template arg.
// This is useful since the compiler can optimize for the hardcoded size, and
// certain errors can be caught at compile time:
//
//   Slice<int, 6> slice3(buffer);        // Slice with size hardcoded to 6.
//
// Similar to the Eigen library's block API, sub-slices can be extracted with
// `head`, `tail`, and `segment` methods:
//
//   slice.head(n)            // Extracts first n elements.
//   slice.tail(n)            // Extracts last n elements.
//   slice.segment(start, n)  // Extracts n elements beginning at `start`.
//
//   // Same as above, but for `n` a compile-time fixed size.
//   slice.head<n>()
//   slice.tail<n>()
//   slice.segment<n>(start)

#ifndef AUDIO_TO_TACTILE_SRC_CPP_SLICE_H_
#define AUDIO_TO_TACTILE_SRC_CPP_SLICE_H_

#include <stdint.h>
#include <string.h>

#include "cpp/fixed_or_dynamic.h"  // NOLINT(build/include)
#include "cpp/std_shim.h"

namespace audio_tactile {

namespace slice_internal {
// We only need to store the slice size if it is dynamic. The `Representation`
// struct is used to conditionally include an `int dynamic_size` member.
template <typename T, int kFixedSize>
struct Representation {  // Case for fixed size.
  T* data;

  constexpr Representation(T* data_in, int unused_size) : data(data_in) {}
  constexpr int size() const noexcept { return kFixedSize; }
};
template <typename T>
struct Representation<T, kDynamic> {  // Case for dynamic size.
  T* data;
  int dynamic_size;  // Store the dynamic size.

  constexpr Representation(T* data_in, int size_in)
      : data(data_in), dynamic_size(size_in) {}
  constexpr int size() const noexcept { return dynamic_size; }
};
}  // namespace slice_internal

template <typename T, int kSize = kDynamic>
class Slice {
 public:
  enum { kSizeAtCompileTime = kSize };
  using value_type = typename std_shim::remove_const<T>::type;
  using iterator = T*;
  using const_iterator = const T*;
  template <typename U>
  using AddConstIfTIsConst = typename
      std_shim::conditional<std_shim::is_const<T>::value, const U, U>::type;
  static_assert(kSize == kDynamic || kSize >= 0,
                "Slice size must be nonnegative");

  // Default constructor sets null data pointer. The size is set to 0 for a
  // dynamic Slice, or kSize for a fixed-sized Slice.
  constexpr Slice() noexcept : representation_(nullptr, 0) {}
  // Construct a Slice with given `data` pointer and `size`. For a fixed-sized
  // Slice, `size` is ignore.
  constexpr Slice(T* data, int size) noexcept : representation_(data, size) {}
  // Only for fixed-sized Slice: construct with a given `data` pointer.
  template <int kLazySize = kSize, typename = typename std_shim::enable_if<
                                       kLazySize != kDynamic>::type>
  constexpr explicit Slice(T* data) noexcept : representation_(data, kSize) {}
  // Implicit conversion from another Slice.
  template <typename RhsT, int kRhsSize>
  constexpr Slice(  // NOLINT(runtime/explicit)
      Slice<RhsT, kRhsSize> rhs) noexcept
      : representation_(rhs.data(), rhs.size()) {
    static_assert(
        std_shim::is_const<T>::value || !std_shim::is_const<RhsT>::value,
        "Cannot convert Slice<const T> to Slice<T>");
    static_assert(
        std_shim::is_same<typename std_shim::remove_const<T>::type,
                          typename std_shim::remove_const<RhsT>::type>::value,
        "Type mismatch constructing Slice");
    static_assert(
        kSize == kRhsSize || kSize == kDynamic || kRhsSize == kDynamic,
        "Size mismatch constructing Slice");
  }

  // Basic accessors.

  // Data pointer to the first element of the Slice.
  constexpr T* data() const noexcept { return representation_.data; }
  // Size in units of elements.
  constexpr int size() const noexcept { return representation_.size(); }
  // Size in units of bytes.
  constexpr int size_bytes() const noexcept { return size() * sizeof(T); }
  // Accesses the ith object in the Slice.
  constexpr T& operator[](int i) const noexcept { return *(data() + i); }
  // Returns true if the Slice is empty.
  constexpr bool empty() const noexcept { return size() == 0; }

  // begin/end iterators.
  constexpr iterator begin() const noexcept { return data(); }
  constexpr iterator end() const noexcept { return data() + size(); }
  constexpr const_iterator cbegin() const noexcept { return data(); }
  constexpr const_iterator cend() const noexcept { return data() + size(); }

  // Methods for extracting sub-slices.

  // `head(n)` extracts a Slice of the first `n` elements.
  constexpr Slice<T> head(int n) const noexcept { return segment(0, n); }
  // `head<n>()` is the same as above, but for compile-time fixed size n.
  template <int kSegmentSize>
  constexpr Slice<T, kSegmentSize> head() const noexcept {
    return segment<kSegmentSize>(0);
  }
  // `tail(n)` extracts a Slice of the last `n` elements.
  constexpr Slice<T> tail(int n) const noexcept {
    return segment(size() - n, n);
  }
  // `tail<n>()` is the same as above, but for compile-time fixed size n.
  template <int kSegmentSize>
  constexpr Slice<T, kSegmentSize> tail() const noexcept {
    return segment<kSegmentSize>(size() - kSegmentSize);
  }
  // `segment(start, n)` extracts a Slice of `n` elements beginning at `start`.
  constexpr Slice<T> segment(int start, int n) const noexcept {
    return Slice<T>(data() + start, n);
  }
  // `segment<n>(start)` is the same as above, but for fixed size n.
  template <int kSegmentSize>
  constexpr Slice<T, kSegmentSize> segment(int start) const noexcept {
    static_assert(kSegmentSize >= 0, "Segment size must be nonnegative");
    static_assert(kSize == kDynamic || kSegmentSize <= kSize,
                  "Segment size must be <= Slice size");
    return Slice<T, kSegmentSize>(data() + start);
  }

  // Reinterprets Slice data as a Slice of bytes. This is useful together with
  // CopyFrom for serialization of trivially-copyable data:
  //
  //   Slice<const float> original = ...
  //   uint8_t buffer[n];
  //   Slice<uint8_t> serialized(buffer, n);
  //   serialized.CopyFrom(original.bytes());
  //
  //   Slice<float> deserialized = ...
  //   deserialized.bytes().CopyFrom(serialized);
  constexpr Slice<AddConstIfTIsConst<uint8_t>,
                  (kSize == kDynamic) ? kDynamic
                                      : (kSize * static_cast<int>(sizeof(T)))>
  bytes() const noexcept {
    return {reinterpret_cast<AddConstIfTIsConst<uint8_t>*>(data()),
            size_bytes()};
  }

  // memcpy from another Slice of same type and size. The destination Slice must
  // be non-const, non-overlapping, and T must be a trivially-copyable type.
  // Returns true on success.
  template <typename RhsT, int kRhsSize>
  bool CopyFrom(Slice<RhsT, kRhsSize> rhs) {
    static_assert(!std_shim::is_const<T>::value,
                  "Cannot call CopyFrom() on a Slice<const T>");
    static_assert(
        std_shim::is_same<typename std_shim::remove_const<T>::type,
                          typename std_shim::remove_const<RhsT>::type>::value,
        "Type mismatch copying Slice");
    static_assert(
        kSize == kRhsSize || kSize == kDynamic || kRhsSize == kDynamic,
        "Size mismatch copying Slice");
    if /*constexpr*/ (kSize != kRhsSize) {
      if (size() != rhs.size()) { return false; }
    }
    if (!empty()) {
      memcpy(data(), rhs.data(), size_bytes());
    }
    return true;
  }

 private:
  slice_internal::Representation<T, kSize> representation_;
};

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_SLICE_H_
