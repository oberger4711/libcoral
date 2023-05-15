#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace coral {
namespace coraldetector {

// This is like BBox but I want the interface to be one header file.
struct CoralDetection {
  // The box x-minimum (left-most) point.
  float xmin;
  // The box y-minimum (top-most) point.
  float ymin;
  // The box x-maximum (right-most) point.
  float xmax;
  // The box y-maximum (bottom-most) point.
  float ymax;
  // Detection score between 0 and 1.
  float score;

  CoralDetection() = default;

  CoralDetection(float xmin, float ymin, float xmax, float ymax, float score)
      : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax), score(score) {}

  // Gets the box width.
  float width() const { return xmax - xmin; }
  // Gets the box height.
  float height() const { return ymax - ymin; }
  // Gets the box area.
  float area() const { return width() * height(); }
  // Checks whether the box is a valid rectangle (width >= 0 and height >= 0).
  bool valid() const { return xmin <= xmax && ymin <= ymax; }
};

class CoralDetector {
 private:
  class CoralDetectorImpl;
  std::unique_ptr<CoralDetectorImpl> p_impl;

 public:
  CoralDetector(const std::string& model_path, const float det_thresh = 0.7);
  ~CoralDetector();
  std::vector<CoralDetection> detect(const std::vector<uint8_t>& image_rgb);
};

}  // namespace coraldetector
}  // namespace coral