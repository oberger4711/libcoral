// Face detection
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, you may resize and convert an existing image to pixel array like:
//   convert cat.bmp -resize 224x224! cat.rgb
#include "coral/coraldetector/coraldetector.h"

#include <cmath>
#include <iostream>
#include <string>

#include "coral/detection/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

// #include "absl/flags/flag.h"
// #include "absl/flags/parse.h"
// #include "coral/classification/adapter.h"

namespace coral {
namespace coraldetector {

constexpr float NORM_MEAN = 127.5;
constexpr float NORM_STD = 127.5;

void preprocess_img(const std::vector<uint8_t>& in,
                    const TfLiteTensor& tensor) {
  if (tensor.type == kTfLiteUInt8) {
    auto data = MutableTensorData<uint8_t>(tensor);
    CHECK_EQ(in.size(), data.size())
        << "Image size must match input size of " << data.size() << " bytes";
    std::memcpy(&data.data()[0], &in[0], in.size());
  } else if (tensor.type == kTfLiteFloat32) {
    auto data = MutableTensorData<float>(tensor);
    CHECK_EQ(in.size(), data.size())
        << "Image size must match input size of " << data.size() << " bytes";
    // Normalize
    for (size_t i = 0; i < data.size(); ++i)
      data[i] = (in[i] - NORM_MEAN) / NORM_STD;
  } else {
    LOG(FATAL) << "Unsupported tensor type " << tensor.type;
  }
}

class CoralDetector::CoralDetectorImpl {
 public:
  CoralDetectorImpl(float det_thresh) : det_thresh(det_thresh) {}

  void load_model(const std::string& model_path) {
    model = LoadModelOrDie(model_path);
    context = GetEdgeTpuContextOrDie("usb");
    interpreter = MakeEdgeTpuInterpreterOrDie(*model, context.get());
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  }

  std::vector<CoralDetection> detect(const std::vector<uint8_t>& image_rgb) {
    CHECK_EQ(interpreter->inputs().size(), 1);
    const auto* input_tensor = interpreter->input_tensor(0);
    preprocess_img(image_rgb, *input_tensor);
    CHECK_EQ(interpreter->Invoke(), kTfLiteOk);

    // Read the label file.
    // auto labels = ReadLabelFile(absl::GetFlag(FLAGS_labels_path));

    const auto dets = GetDetectionResults(*interpreter, det_thresh);
    std::vector<CoralDetection> out;
    out.reserve(dets.size());
    for (const auto& det : dets) {
      out.emplace_back(det.bbox.xmin, det.bbox.ymin, det.bbox.xmax, det.bbox.ymax, det.score);
    }
    return out;
  }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::shared_ptr<edgetpu::EdgeTpuContext> context; // Needs to stay alive or USB connection will be cleaned up.
  float det_thresh;
};

CoralDetector::CoralDetector(const std::string& model_path,
                             const float det_thresh)
    : p_impl(std::make_unique<CoralDetectorImpl>(det_thresh)) {
  p_impl->load_model(model_path);
}

CoralDetector::~CoralDetector() = default;

std::vector<CoralDetection> CoralDetector::detect(
    const std::vector<uint8_t>& image_rgb) {
  return p_impl->detect(image_rgb);
}

}  // namespace coraldetector
}  // namespace coral