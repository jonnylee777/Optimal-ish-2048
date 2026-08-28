#pragma once

#include <atomic>
#include <version>

namespace adversarial_2048::learning {

// Relaxed atomic access to a single float, for Hogwild-style parallel training.
//
// Several threads read and write the same weights with no locking. Lost updates
// are fine and are the point: 40 weights out of 83.9M are touched per update, so
// collisions are rare and the noise they add is far smaller than the noise TD
// already carries. What is NOT fine is a plain concurrent read/write, which is a
// data race and therefore undefined behaviour however benign it looks in
// practice -- and this project keeps a sanitized build that would rightly flag
// it.
//
// On ARM64 and x86-64 these lower to the same load/store instruction a plain
// access emits, so the concurrent path is free. Callers still gate on a flag so
// the single-threaded path is provably the code it was, not merely as fast.

// Apple Clang's libc++ does not ship std::atomic_ref yet; the builtins are the
// same operation spelled differently and exist on both Clang and GCC.
#if defined(__cpp_lib_atomic_ref)

[[nodiscard]] inline float load_relaxed(const float& slot) noexcept {
    return std::atomic_ref<const float>(slot).load(std::memory_order_relaxed);
}

inline void add_relaxed(float& slot, float delta) noexcept {
    std::atomic_ref<float> ref(slot);
    ref.store(ref.load(std::memory_order_relaxed) + delta, std::memory_order_relaxed);
}

#else

// The generic __atomic_load/__atomic_store forms, not the _n suffixed ones:
// those only accept integer and pointer types, and these weights are float.
[[nodiscard]] inline float load_relaxed(const float& slot) noexcept {
    float value{};
    // const_cast because the builtin's first parameter is non-const even for a
    // pure load; the slot is genuinely only read here.
    __atomic_load(const_cast<float*>(&slot), &value, __ATOMIC_RELAXED);
    return value;
}

inline void add_relaxed(float& slot, float delta) noexcept {
    // Deliberately load-add-store rather than a compare-exchange loop. Under
    // Hogwild a lost update is acceptable and expected -- retrying until the
    // read-modify-write is atomic would serialise exactly the contended weights
    // to buy an accuracy TD does not need.
    float current{};
    __atomic_load(&slot, &current, __ATOMIC_RELAXED);
    // Not const: the builtin takes a mutable pointer to the value to store.
    float next = current + delta;
    __atomic_store(&slot, &next, __ATOMIC_RELAXED);
}

#endif

}  // namespace adversarial_2048::learning
