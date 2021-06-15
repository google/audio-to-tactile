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
// std_shim, substitutes for a few C++ standard library definitions.
//
// Segger Embedded Studio seems to be missing most of the C++ standard
// libraries, so this file provides substitutes for a few necessities.

#ifndef AUDIO_TO_TACTILE_SRC_CPP_STD_SHIM_H_
#define AUDIO_TO_TACTILE_SRC_CPP_STD_SHIM_H_

namespace audio_tactile {

// NOTE: Adding user code under namespace std has undefined behavior; only the
// standard library should do that. So we put these defs under `std_shim`.
namespace std_shim {

// Substitutes for std::min and std::max. A full implementation would also
// support passing an initializer list and passing a comparator. We just do the
// simple two-arg form with a ternary.
template <typename T>
constexpr inline const T& min(const T& a, const T& b) {
  return (b < a) ? b : a;
}
template <typename T>
constexpr inline const T& max(const T& a, const T& b) {
  return (a < b) ? b : a;
}

// Limited substitutes for std::true_type and std::false_type.
struct true_type { static constexpr bool value = true; };
struct false_type { static constexpr bool value = false; };

// Substitute for std::is_same. Like the other metafunctions below, the
// implementation is through partial specialization: The first line defines a
// base case `is_same<T, U>` as false and the next line add a "specialization"
// that `is_same<T, T>` is true.
template<typename T, typename U> struct is_same: public false_type {};
template<typename T> struct is_same<T, T>: public true_type {};

// Substitute for std::is_const.
template <typename T> struct is_const: public false_type {};
template <typename T> struct is_const<const T>: public true_type {};

// Substitute for std::remove_reference.
template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T&> { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };

// Substitute for std::remove_const.
template <class T> struct remove_const { using type = T; };
template <class T> struct remove_const<const T> { using type = T; };
template <class T> struct remove_const<const T[]> { using type = T[]; };
template <class T, unsigned int Size>
struct remove_const<const T[Size]> { using type = T[Size]; };

// Substitute for std::conditional.
template <bool Condition, typename Then, typename Else>
struct conditional { using type = Then; };
template <typename Then, typename Else>
struct conditional <false, Then, Else> { using type = Else; };

// Substitute for std::enable_if.
template <bool Condition, typename T = void> struct enable_if;
template <typename T> struct enable_if<true, T> { using type = T; };

}  // namespace std_shim
}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_STD_SHIM_H_
