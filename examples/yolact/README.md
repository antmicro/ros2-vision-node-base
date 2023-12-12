# Instance segmentation inference YOLACT

This demo runs an instance segmentation model YOLACT on sequences from [LindenthalCameraTraps](https://lila.science/datasets/lindenthal-camera-traps/) dataset.
The demo consists of four parts:

* `CVNodeManager` - manages testing scenario and data flow between dataprovider and tested CVNode.
* `CVNodeManagerGUI` - visualizes the input data and results of the inference testing.
* `Kenning` - provides sequences from LindenthalCameraTraps dataset and collects inference results.
* `CVNode` - runs inference on the input images and returns the results.

## Dependencies

This demo requires the following dependencies:
* [CVNodeManager](https://github.com/antmicro/ros2-vision-node-manager)
* [KenningComputerVisionMessages](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [Kenning](https://github.com/antnmicro/kenning)

## Building the demo

First of all, load the `setup.sh` script for ROS 2 tools, e.g.:

```bash
source /opt/ros/setup.sh
```

Then, build the GUI node and the Camera node with:

```bash
colcon build --base-path=src/ --packages-select \
    kenning_computer_vision_msgs \
    cvnode_base \
    cvnode_manager \
    --cmake-args ' -DBUILD_GUI=ON' ' -DBUILD_YOLACT=ON'
```

Where the `--cmake-args` are:

* `-DBUILD_GUI=ON` - builds the GUI for CVNodeManager
* `-DBUILD_YOLACT=ON` - builds the YOLACT CVNodes

Build targets then can be sourced with:

```bash
source install/setup.sh
```

## Running the demo

This example provides a single launch scripts for running the demo:

* `yolact_launch.py` - starts provided executable as CVNode along with other nodes

A sample launch with the TFLite backend can be run with:

```bash
ros2 launch cvnode_base yolact_launch.py \
    backend:=tflite \
    model_path:=/data/model.tflite \
    scenario:=synthetic \
    measurements:=/data/build/yolact_tflite_measurements.json \
    report_path:=/data/build/reports/yolact_synthetic_tflite/report.md \
    log_level:=INFO
```

Where the parameters are:

* `tflite` - backend to use, one of:
    * `tflite` - TFLite backend
    * `tvm` - TVM backend
    * `onnxruntime` - ONNXRuntime backend
* `model_path` - path to the model file.
Make sure to have IO specification placed alongside the model file with the same name and `.json` extension.
* `scenario` - scenario to run the demo in, one of:
    * `real_world_last` - tries to process last received frame within timeout
    * `real_world_first` - tries to process first received frame
    * `synthetic` - ignores timeout and processes frames as fast as possible
* `measurements` - path to the file where inference measurements will be stored
* `report_path` - path to the file where the rendered report will be stored
* `log_level` - log level for running the demo

Later, produced reports can be found under `/data/build/reports` directory.

This demo supports TFLite, TVM and ONNX backends.
For more information on how to export model for these backends, see [Kenning documentation](https://antmicro.github.io/kenning/json-scenarios.html).
