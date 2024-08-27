// Copyright 2021, 2023 Google LLC
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

#include "src/cpp/object_pool.h"

#include <string>

#include "src/dsp/logging.h"

// NOLINTBEGIN(readability/check)

namespace audio_tactile {

// A toy object for testing the pool. This object has a `label` member so that
// the test can distinguish different instances, and `constructor_count` and
// `destructor_count` static members to track the number of constructor and
// destructor calls.
struct Object {
  explicit Object(int label_in) : label(label_in) { ++constructor_count; }
  ~Object() { ++destructor_count; }

  char label;

  static void ResetCounters() { constructor_count = destructor_count = 0; }
  static int constructor_count;
  static int destructor_count;
};
int Object::constructor_count = 0;
int Object::destructor_count = 0;

// Gets the state of the pool. Returns a string of length kCapacity, in which
// the ith char is '*' if the ith object is live or '.' if it is free.
template <typename PoolType>
std::string PoolState(const PoolType& pool) {
  std::string state(PoolType::kCapacity, ' ');
  char is_live[PoolType::kCapacity];
  PoolType::TestAccess::MarkLiveObjects(pool, is_live);
  for (int i = 0; i < PoolType::kCapacity; ++i) {
    state[i] = is_live[i] ? '*' : '.';
  }
  return state;
}

// Test the initial state of the pool.
void TestPoolInitialState() {
  puts("TestPoolInitialState");
  ObjectPool<Object, 5> pool;
  CHECK(decltype(pool)::kCapacity == 5);
  CHECK(pool.num_live() == 0);  // All pool objects are initially free.
  CHECK(pool.num_free() == 5);
  CHECK(PoolState(pool) == ".....");
}

// Test allocating and freeing a few objects.
void TestPoolAllocation() {
  puts("TestPoolAllocation");
  Object::ResetCounters();

  ObjectPool<Object, 5> pool;
  CHECK(Object::constructor_count == 0);

  Object* object_a = pool.Allocate('a');  // Allocate 3 objects.
  Object* object_b = pool.Allocate('b');
  Object* object_c = pool.Allocate('c');

  CHECK(object_a && object_a->label == 'a');
  CHECK(object_b && object_b->label == 'b');
  CHECK(object_c && object_c->label == 'c');
  CHECK(Object::constructor_count == 3);  // Constructor called 3 times.
  CHECK(Object::destructor_count == 0);
  CHECK(pool.num_live() == 3);
  CHECK(pool.num_free() == 2);
  CHECK(PoolState(pool) == "***..");

  pool.Free(object_b);                    // Free the objects.
  pool.Free(object_c);
  pool.Free(object_a);

  CHECK(Object::destructor_count == 3);   // Destructor called 3 times.
  CHECK(PoolState(pool) == ".....");
}

// Test that pool destructor destroys remaining live objects.
void TestPoolDestructor() {
  puts("TestPoolDestructor");
  Object::ResetCounters();

  {
    ObjectPool<Object, 5> pool;
    pool.Allocate('a');  // Allocate 3 objects.
    pool.Allocate('b');
    pool.Allocate('c');
    CHECK(Object::destructor_count == 0);
  }

  // Pool destructor should have destroyed the 3 objects.
  CHECK(Object::destructor_count == 3);
}

// Test that pool reuses objects.
void TestPoolObjectReuse() {
  puts("TestPoolObjectReuse");
  Object::ResetCounters();

  ObjectPool<Object, 5> pool;
  using PoolType = decltype(pool);
  Object* object_a = pool.Allocate('a');  // Allocate 4 objects.
  Object* object_b = pool.Allocate('b');
  Object* object_c = pool.Allocate('c');
  Object* object_d = pool.Allocate('d');

  CHECK(Object::constructor_count == 4);
  CHECK(object_a && object_a->label == 'a');
  CHECK(object_b && object_b->label == 'b');
  CHECK(object_c && object_c->label == 'c');
  CHECK(object_d && object_d->label == 'd');
  CHECK(PoolState(pool) == "****.");

  pool.Free(object_b);                    // Free a couple objects.
  pool.Free(object_c);

  CHECK(Object::destructor_count == 2);
  CHECK(pool.num_live() == 2);
  CHECK(PoolState(pool) == "*..*.");

  Object* object_e = pool.Allocate('e');  // Allocate an object, reusing c.
  CHECK(object_e == object_c);
  CHECK(object_e->label == 'e');
  CHECK(PoolState(pool) == "*.**.");

  Object* object_f = pool.Allocate('f');  // Allocate an object, reusing b.
  CHECK(object_f == object_b);
  CHECK(object_f->label == 'f');
  CHECK(PoolState(pool) == "****.");

  Object* object_g = pool.Allocate('g');  // Allocate one more, but not a reuse.
  CHECK(object_g && object_g->label == 'g');
  CHECK(Object::constructor_count == 7);
  CHECK(PoolState(pool) == "*****");

  // Check that pool nodes correspond to what they should.
  CHECK(PoolType::TestAccess::GetObject(pool, 0) == object_a);
  CHECK(PoolType::TestAccess::GetObject(pool, 1) == object_f);
  CHECK(PoolType::TestAccess::GetObject(pool, 2) == object_e);
  CHECK(PoolType::TestAccess::GetObject(pool, 3) == object_d);
  CHECK(PoolType::TestAccess::GetObject(pool, 4) == object_g);
}

// Test pool behavior when out of memory.
void TestPoolOutOfMemory() {
  puts("TestPoolOutOfMemory");
  Object::ResetCounters();

  ObjectPool<Object, 5> pool;
  Object* object_a = pool.Allocate('a');  // Try to allocate 6 objects.
  Object* object_b = pool.Allocate('b');
  Object* object_c = pool.Allocate('c');
  Object* object_d = pool.Allocate('d');
  Object* object_e = pool.Allocate('e');
  Object* object_f = pool.Allocate('f');

  CHECK(object_a && object_a->label == 'a');
  CHECK(object_b && object_b->label == 'b');
  CHECK(object_c && object_c->label == 'c');
  CHECK(object_d && object_d->label == 'd');
  CHECK(object_e && object_e->label == 'e');
  CHECK(object_f == nullptr);             // Failed to allocate f.
  CHECK(Object::constructor_count == 5);

  // But if we free d, allocation works if we try again.
  pool.Free(object_d);
  object_f = pool.Allocate('f');
  CHECK(object_f == object_d);
  CHECK(object_f->label == 'f');
  CHECK(Object::constructor_count == 6);
  CHECK(PoolState(pool) == "*****");
}

}  // namespace audio_tactile

// NOLINTEND

int main(int argc, char** argv) {
  audio_tactile::TestPoolInitialState();
  audio_tactile::TestPoolAllocation();
  audio_tactile::TestPoolDestructor();
  audio_tactile::TestPoolObjectReuse();
  audio_tactile::TestPoolOutOfMemory();

  puts("PASS");
  return EXIT_SUCCESS;
}
