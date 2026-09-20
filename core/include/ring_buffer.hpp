// Spectraflow — single-producer / single-consumer lock-free ring buffer.
//
// The UDP ingestion path is the hot path of the whole system: the native
// receiver pushes frames from the socket thread while the DSP/pose worker
// drains them. A mutex there would couple the two rates and add jitter to the
// very phase measurements we are trying to preserve, so the queue is
// wait-free for both ends and degrades by dropping the newest frame (never by
// blocking the producer).
//
// Memory ordering: `head_` is published with release semantics and read with
// acquire semantics, which is sufficient for a single-producer /
// single-consumer queue because each index is written by exactly one thread.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "csi_packet.hpp"

namespace spectraflow {

/// Cumulative counters describing how the buffer has behaved.
struct RingBufferStats {
  std::uint64_t pushed = 0;
  std::uint64_t popped = 0;
  std::uint64_t dropped = 0;  ///< pushes rejected because the buffer was full
  std::size_t capacity = 0;
  std::size_t size = 0;
};

/// Lock-free SPSC queue of CSI frames.
///
/// Capacity is rounded up to the next power of two so the wrap-around is a
/// mask rather than a modulo (a modulo per frame is measurable at 100+ Hz on
/// an ARM core).
class CsiRingBuffer {
 public:
  /// @param capacity Minimum number of frames to hold; rounded up to a power
  ///                 of two, with a floor of 2.
  explicit CsiRingBuffer(std::size_t capacity);

  CsiRingBuffer(const CsiRingBuffer&) = delete;
  CsiRingBuffer& operator=(const CsiRingBuffer&) = delete;

  /// Producer side. Returns false when the buffer is full (frame is dropped
  /// and counted); never blocks and never reallocates.
  bool push(CsiFrame&& frame);
  bool push(const CsiFrame& frame);

  /// Consumer side. Returns std::nullopt when empty.
  std::optional<CsiFrame> pop();

  /// Peek at the most recently pushed frame without removing anything.
  /// Returns std::nullopt when empty.
  std::optional<CsiFrame> peek_latest() const;

  /// Consumer-side helper: pop every currently-available frame into `out`.
  /// Returns the number of frames moved.
  std::size_t drain(std::vector<CsiFrame>& out, std::size_t max_frames = 0);

  void clear();

  std::size_t capacity() const { return capacity_; }
  /// Approximate size; exact only when no producer is concurrently pushing.
  std::size_t size() const;
  bool empty() const;
  bool full() const;

  RingBufferStats stats() const;

 private:
  const std::size_t capacity_;
  const std::size_t mask_;
  std::vector<CsiFrame> slots_;

  // Separate cache lines stop the producer and consumer from false-sharing.
  alignas(64) std::atomic<std::uint64_t> head_{0};
  alignas(64) std::atomic<std::uint64_t> tail_{0};
  alignas(64) std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace spectraflow
