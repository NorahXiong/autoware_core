// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/trajectory/interpolator/lane_ids_interpolator.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace autoware::experimental::trajectory::interpolator
{

// Helper function to work around GCC 13 false positive warning
// This prevents the compiler from inlining vector copy operations that trigger -Warray-bounds
[[gnu::noinline]] static std::vector<int64_t> safe_vector_copy(
  const std::vector<int64_t> & vec)
{
  return vec;
}

std::vector<int64_t> LaneIdsInterpolator::compute_impl(const double s) const
{
  const int32_t idx = this->get_index(s);

  // Check for exact matches at base points
  if (s == this->bases_[idx]) {
    return values_.at(idx);
  }

  // Ensure we have a valid right boundary for interpolation
  // get_index with end_inclusive=true returns size-2 for s==end(), but we add explicit check
  const size_t next_idx = static_cast<size_t>(idx) + 1;
  if (next_idx >= values_.size()) {
    // Should not happen due to get_index logic, but guard against it for GCC optimization
    return values_.at(idx);
  }

  if (s == this->bases_[next_idx]) {
    return values_.at(next_idx);
  }

  // Domain knowledge: prefer boundaries with single lane IDs over multiple lane IDs
  // This handles the case where lane boundaries should contain more than two elements
  const size_t left_size = values_.at(idx).size();
  const size_t right_size = values_.at(next_idx).size();

  if (left_size == 1 && right_size > 1) {
    return safe_vector_copy(values_.at(idx));
  }
  if (left_size > 1 && right_size == 1) {
    return safe_vector_copy(values_.at(next_idx));
  }

  // If both are single or both are multiple, choose the closest one
  const double left_distance = s - this->bases_[idx];
  const double right_distance = this->bases_[next_idx] - s;
  return (left_distance <= right_distance) ? safe_vector_copy(values_.at(idx))
                                           : safe_vector_copy(values_.at(next_idx));
}

bool LaneIdsInterpolator::build_impl(
  const std::vector<double> & bases, const std::vector<std::vector<int64_t>> & values)
{
  this->bases_ = bases;
  this->values_ = values;
  return true;
}

bool LaneIdsInterpolator::build_impl(
  const std::vector<double> & bases, std::vector<std::vector<int64_t>> && values)
{
  this->bases_ = bases;
  this->values_ = std::move(values);
  return true;
}

size_t LaneIdsInterpolator::minimum_required_points() const
{
  return 2;
}

}  // namespace autoware::experimental::trajectory::interpolator
