# Hacked this on top of the libcoral to get a library artefact for face detection purpose.

package(
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "libcoraldetector.so",
    linkshared = True,
    srcs = [
        "coraldetector.cc",
        "coraldetector.h"
    ],
    deps = [
        "//coral/examples:file_utils",
        "//coral:tflite_utils",
        "//coral/detection:adapter",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@glog",
        "@libedgetpu//tflite/public:edgetpu",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common"


        # From detection_models_tests
        #":common",
        #":pipelined_model_runner",
        #":test_utils",
        #"//coral:error_reporter",
        #"//coral:test_utils",
        #"//coral:tflite_utils",
        #"//coral/detection:adapter",
        #"//coral/pipeline/internal:default_allocator",
        #"@com_google_absl//absl/flags:flag",
        #"@com_google_absl//absl/types:span",
        #"@com_google_googletest//:gtest",
        #"@glog",
        #"@libedgetpu//tflite/public:edgetpu",
        #"@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        #"@org_tensorflow//tensorflow/lite:framework",
    ],
)