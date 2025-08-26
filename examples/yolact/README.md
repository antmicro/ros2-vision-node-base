# Instance segmentation inference YOLACT

This demo runs the YOLACT instance segmentation model on sequences from the [LindenthalCameraTraps](https://lila.science/datasets/lindenthal-camera-traps/) dataset.
The demo consists of four parts:

* `CVNodeManager` - manages testing scenario and data flow between dataprovider and tested CVNode.
* `CVNodeManagerGUI` - visualizes input data and results of inference testing.
* `Kenning` - provides sequences from the LindenthalCameraTraps dataset and collects inference results.
* `CVNode` - runs inference on input images and returns results.

## Dependencies

This demo requires the following dependencies:
* [CVNodeManager](https://github.com/antmicro/ros2-vision-node-manager)
* [KenningComputerVisionMessages](https://github.com/antmicro/ros2-kenning-computer-vision-msgs)
* [Kenning](https://github.com/antnmicro/kenning)

## Quickstart

Since setting up a whole environment for ROS 2 and Kenning can be time-consuming, we will use the ready Docker image with ROS 2 and minimal Kenning dependencies.


First of, create an empty workspace directory (source files for the project, virtual environment and `repo` tool configuration will be placed here), download the [`demo-helper` script](https://raw.githubusercontent.com/antmicro/ros2-vision-node-base/main/examples/demo-helper) and run it:

```bash
mkdir workspace && cd workspace
wget https://raw.githubusercontent.com/antmicro/ros2-vision-node-base/main/examples/demo-helper
chmod +x ./demo-helper
./demo-helper prepare-workspace
```

This will:

* Pull a Docker image with ROS 2 and basic Kenning dependencies - the image is provided in [ros2-gui-node repository Dockerfile](https://github.com/antmicro/ros2-gui-node/blob/main/examples/kenning-instance-segmentation/Dockerfile) and available under `ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo`.
* In Docker container, it will:
    * Load ROS 2 workspace
    * Clone necessary repositories for demo purposes
    * Install Kenning with its dependencies for optimization, inference and reports
    * Build the ROS 2 Vision node project for YOLACT instance segmentation model, as well as its dependencies

Secondly, enter the Docker container with:

```bash
./demo-helper enter-docker
```

Command checks for CUDA-capable GPU.
If no such device is present, it will use CPU for inference.
You can also type:

```bash
export USE_CPU="1"
```

to force running demo with CPU inference.

and (in the Docker container):

```bash
source ./demo-helper source-workspace
```

From this point, we can run ROS 2 node running YOLACT edge detection in various implementations.
The easiest option is to just run the model using ONNX Runtime on CPU with:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=onnxruntime \
    model_path:=./models/yolact-lindenthal.onnx \
    measurements:=onnxruntime.json \
    report_path:=onnxruntime/report.md
```

What we can do next is we can specify the scenario to be based on real time:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=onnxruntime \
    model_path:=./models/yolact-lindenthal.onnx \
    measurements:=onnxruntime-rt.json \
    report_path:=onnxruntime-rt/report.md \
    scenario:=real_world_first
```

The framerate can be controlled with `inference_timeout_ms:=<val>`.
For more options, check:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py --show-args
```

To use another runtime, e.g. TVM, first compile the model:

```bash
kenning optimize --json-cfg ./src/vision_node_base/examples/config/yolact-tvm-lindenthal.json
```

The compiled model is available under `build/yolact.so`.

After compiling, run the model with:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tvm \
    model_path:=./build/yolact.so \
    measurements:=tvm.json \
    report_path:=tvm/report.md
```

In the end, we can build the comparison report for various runtimes and time constraints, e.g.:

```bash
kenning report --measurements \
	onnxruntime.json \
	tvm.json \
	--report-name "YOLACT comparison report" \
	--report-path comparison-result/yolact/report.md \
	--report-types performance detection \
	--to-html comparison-result-html
```

## Building the demo

First, load the `setup.sh` script for ROS 2 tools, e.g.:

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

Here, the `--cmake-args` are:

* `-DBUILD_GUI=ON` - builds the GUI for CVNodeManager
* `-DBUILD_YOLACT=ON` - builds the YOLACT CVNodes

Build targets then can be sourced with:

```bash
source install/setup.sh
```

## Running the demo

This example provides a single launch script for running the demo:

* `yolact_kenning_launch.py` - starts the provided executable as CVNode along with other nodes.

Run a sample launch with the TFLite backend using:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tflite \
    model_path:=/data/model.tflite \
    scenario:=synthetic \
    measurements:=/data/build/yolact_tflite_measurements.json \
    report_path:=/data/build/reports/yolact_synthetic_tflite/report.md \
    log_level:=INFO
```

Here, the parameters are:

* `tflite` - backend to use, one of:
    * `tflite` - TFLite backend
    * `tvm` - TVM backend
    * `onnxruntime` - ONNXRuntime backend
* `model_path` - path to the model file.
Make sure to have IO specification placed alongside the model file with the same name and `.json` extension.
* `scenario` - scenario for running the demo, one of:
    * `real_world_last` - tries to process last received frame within timeout
    * `real_world_first` - tries to process first received frame
    * `synthetic` - ignores timeout and processes frames as fast as possible
* `measurements` - path to file where inference measurements will be stored
* `report_path` - path to file where the rendered report will be stored
* `log_level` - log level for running the demo

The produced reports can later be found in the `/data/build/reports` directory.

This demo supports TFLite, TVM and ONNX backends.
For more information on how to export model for these backends, see [Kenning documentation](https://antmicro.github.io/kenning/json-scenarios.html).
