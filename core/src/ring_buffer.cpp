// Spectraflow — lock-free SPSC CSI ring buffer.
#include "ring_buffer.hpp"

#include <algorithm>
#include <utility>

namespace spectraflow {
namespace {

/// Round `n` up to the next power of two, with a floor of 2.
constexpr std::size_t round_up_pow2(std::size_t n) {
  std::size_t p = 2;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

}  // namespace

CsiRingBuffer::CsiRingBuffer(std::size_t capacity)
    : capacity_(round_up_pow2(std::max<std::size_t>(capacity, 2))),
      mask_(capacity_ - 1),
      slots_(capacity_) {}

bool CsiRingBuffer::push(const CsiFrame& frame) {
  const std::uint64_t head = head_.load(std::memory_order_relaxed);
  const std::uint64_t tail = tail_.load(std::memory_order_acquire);
  if (head - tail >= capacity_) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  slots_[head & mask_] = frame;
  // Release: the slot write above must be visible before the consumer sees the
  // new head.
  head_.store(head + 1, std::memory_order_release);
  return true;
}

bool CsiRingBuffer::push(CsiFrame&& frame) {
  const std::uint64_t head = head_.load(std::memory_order_relaxed);
  const std::uint64_t tail = tail_.load(std::memory_order_acquire);
  if (head - tail >= capacity_) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  slots_[head & mask_] = std::move(frame);
  head_.store(head + 1, std::memory_order_release);
  return true;
}

std::optional<CsiFrame> CsiRingBuffer::pop() {
  const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
  // Acquire: pairs with the producer's release store, guaranteeing the slot
  // contents are visible.
  const std::uint64_t head = head_.load(std::memory_order_acquire);
  if (tail >= head) {
    return std::nullopt;
  }
  CsiFrame frame = std::move(slots_[tail & mask_]);
  tail_.store(tail + 1, std::memory_order_release);
  return frame;
}

std::optional<CsiFrame> CsiRingBuffer::peek_latest() const {
  const std::uint64_t tail = tail_.load(std::memory_order_relaxed);
  const std::uint64_t head = head_.load(std::memory_order_acquire);
  if (tail >= head) {
    return std::nullopt;
  }
  return slots_[(head - 1) & mask_];
}

std::size_t CsiRingBuffer::drain(std::vector<CsiFrame>& out,
                                 std::size_t max_frames) {
  std::size_t moved = 0;
  while (max_frames == 0 || moved < max_frames) {
    auto frame = pop();
    if (!frame) {
      break;
    }
    out.push_back(std::move(*frame));
    ++moved;
  }
  return moved;
}

void CsiRingBuffer::clear() {
  const std::uint64_t head = head_.load(std::memory_order_acquire);
  tail_.store(head, std::memory_order_release);
}

std::size_t CsiRingBuffer::size() const {
  const std::uint64_t head = head_.load(std::memory_order_acquire);
  const std::uint64_t tail = tail_.load(std::memory_order_acquire);
  return static_cast<std::size_t>(head >= tail ? head - tail : 0);
}

bool CsiRingBuffer::empty() const { return size() == 0; }
bool CsiRingBuffer::full() const { return size() >= capacity_; }

RingBufferStats CsiRingBuffer::stats() const {
  RingBufferStats s;
  s.pushed = head_.load(std::memory_order_acquire);
  s.popped = tail_.load(std::memory_order_acquire);
  s.dropped = dropped_.load(std::memory_order_acquire);
  s.capacity = capacity_;
  s.size = size();
  return s;
}

}  // namespace spectraflow
