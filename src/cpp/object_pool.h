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
// ObjectPool, a pool of reusable objects.
//
// This is a simple generic implementation of the object pool design pattern
// [https://en.wikipedia.org/wiki/Object_pool_pattern]. ObjectPool<T, kCapacity>
// is a pool of `T` objects of capacity `kCapacity`. Objects are allocated and
// freed from the pool with Allocate() and Free():
//
//   ObjectPool<T, 10> pool;
//   T* obj1 = pool.Allocate();
//   T* obj2 = pool.Allocate();
//   T* obj3 = pool.Allocate();
//   assert(pool.num_live() == 3);
//
//   pool.Free(obj2);  // Objects may be freed in any order.
//   pool.Free(obj3);
//   pool.Free(obj1);

#ifndef AUDIO_TO_TACTILE_SRC_CPP_OBJECT_POOL_H_
#define AUDIO_TO_TACTILE_SRC_CPP_OBJECT_POOL_H_

#include <algorithm>
#include <new>
#include <type_traits>
#include <utility>

namespace audio_tactile {

template <typename T, int kCapacity_>
class ObjectPool {
 public:
  enum { kCapacity = kCapacity_ };
  struct TestAccess;

  ObjectPool() noexcept: num_free_(kCapacity) {
    first_free_ = &nodes_[0];
    for (int i = 0; i < kCapacity - 1; ++i) {
      nodes_[i].next_free = &nodes_[i + 1];
    }
  }
  ObjectPool(const ObjectPool&) = delete;  // No copying.
  ObjectPool& operator=(const ObjectPool&) = delete;

  ~ObjectPool() {
    // Find and free any remaining live objects. We only need to do this if T
    // has a nontrivial destructor.
    if (!std::is_trivially_destructible<T>::value && num_live() > 0) {
      char is_live[kCapacity];
      MarkLiveObjects(is_live);
      for (int i = 0; i < kCapacity; ++i) {
        if (is_live[i]) {
          Free(GetObject(i));
        }
      }
    }
  }

  // Number of live objects.
  int num_live() const noexcept { return kCapacity - num_free_; }
  // Number of free objects, available for allocation.
  int num_free() const noexcept { return num_free_; }

  // Allocates an object from the pool and invokes T's constructor, creating the
  // new object on a preallocated block of memory. Constructor arguments may be
  // passed as:
  //
  //   pool.Allocate(x, y, z);  // Invokes constructor `T(x, y, z)`.
  template <typename... Args>
  T* Allocate(Args&&... args) {
    if (num_free_ <= 0) { return nullptr; }
    --num_free_;
    Node* node = first_free_;
    first_free_ = node->next_free;  // Pop head off the free list.
    void* memory = node->storage;
    return new(memory) T(std::forward<Args>(args)...);  // Construct the object.
  }

  // Frees an object from the pool.
  //
  // WARNING: `object` must be a pointer for a live object in this pool
  // previously obtained from Allocate(), otherwise behavior is undefined.
  void Free(T* object) {
    if (num_free_ == kCapacity || object == nullptr) { return; }
    Node* node = reinterpret_cast<Node*>(object);
    object->~T();  // Destruct the object.
    node->next_free = first_free_;  // Push onto head of the free list.
    first_free_ = node;
    ++num_free_;
  }

 private:
  // Returns T pointer to the ith node, assuming it holds a live object.
  T* GetObject(int i) noexcept {
    return reinterpret_cast<T*>(nodes_[i].storage);
  }

  // Fills array `is_live` such that `is_live[i]` is 1 if the ith object is
  // live or 0 if it is free.
  void MarkLiveObjects(char is_live[kCapacity]) const noexcept {
    std::fill_n(is_live, kCapacity, 1);    // Initialize all objects as live.
    Node* p = first_free_;
    for (int i = 0; i < num_free_; ++i) {  // Iterate the free list.
      is_live[p - nodes_] = 0;             // Mark object as free.
      p = p->next_free;
    }
  }

  // Each `Node` contains either a T, if that object is in use, or a pointer
  // to the next free node. The latter is used to create a singly-linked list to
  // keep track of which nodes are free.
  union Node {
    alignas(T) char storage[sizeof(T)];
    Node* next_free;
  };

  Node nodes_[kCapacity];
  Node* first_free_;  // Head of the free list.
  int num_free_;
};

// Test-only access to ObjectPool.
template <typename T, int kCapacity>
struct ObjectPool<T, kCapacity>::TestAccess {
  static T* GetObject(ObjectPool<T, kCapacity>& pool, int i) {
    return pool.GetObject(i);
  }
  static void MarkLiveObjects(const ObjectPool<T, kCapacity>& pool,
                              char is_live[kCapacity]) {
    return pool.MarkLiveObjects(is_live);
  }
};

}  // namespace audio_tactile

#endif  // AUDIO_TO_TACTILE_SRC_CPP_OBJECT_POOL_H_
