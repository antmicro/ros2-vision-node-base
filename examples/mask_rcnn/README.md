# Instance segmentation inference testing with MaskRCNN

This demo runs an instance segmentation algorithm on frames from the COCO dataset.
The demo consists of four parts:

* `CVNodeManager` - manages testing scenario and data flow between dataprovider and tested MaskRCNN node.
* `CVNodeManagerGUI` - visualizes input data and results of inference testing.
* `Kenning` - provides images to the MaskRCNN node and collects inference results.
* `MaskRCNN` - runs inference on input images and returns results.

## Necessary dependencies

This demo requires:

* A CUDA-enabled NVIDIA GPU for inference acceleration
* [repo tool](https://gerrit.googlesource.com/git-repo/+/refs/heads/main/README.md) to clone all necessary repositories
* [Docker](https://www.docker.com/) to use a prepared environment
* [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to provide access to the GPU in the Docker container.

All the necessary build, runtime and development dependencies are provided in the [Dockerfile](./Dockerfile).
The image contains:

* [ROS 2 Humble](https://docs.ros.org/en/humble/index.html) environment
* [OpenCV](https://github.com/opencv/opencv) for image processing
* Dependencies for the [Kenning framework](https://github.com/antmicro/kenning)
* [TorchLib](https://pytorch.org/cppdocs/) and [Detectron2](https://github.com/facebookresearch/detectron2/) targets for `MaskRCNN` node execution
* CUDNN and CUDA libraries for faster acceleration on GPUs
* Additional development tools

To build the Docker image containing all necessary dependencies, run:

```bash
sudo ./build-docker.sh
```

For more details regarding base image refer to the [ROS2 GuiNode](https://github.com/antmicro/ros2-gui-node/blob/main/examples/kenning-instance-segmentation/README.md).

## Preparing the environment

First off, create a workspace directory to store downloaded repositories:

```bash
mkdir cvnode && cd cvnode
```

Download all dependencies using the `repo` tool:

```bash
repo init -u https://github.com/antmicro/ros2-vision-node-base.git -m examples/mask_rcnn/manifest.xml -b main

repo sync -j`nproc`
```

It downloads the following repositories:

* [Kenning](https://github.com/antmicro/kenning) for providing input data and inference testing reports rendering.
* [Detectron2](htpps://github.com/facebookresearch/detectron2) for exporting MaskRCNN model to TorchScript.
* [ROS2 CVNodeManager](https://github.com/antmicro/ros2-vision-node-manager) for managing the inference testing flow.
* [Kenning's ROS2 messages and services](https://github.com/antmicro/ros2-kenning-computer-vision-msgs) for computer vision.
* This repository, in the `src/cvnode_base` directory.

## Starting the Docker environment

If you are using the Docker container, allow non-network local connections to X11 so that the GUI can be started from the Docker container:

```bash
xhost +local:
```

Then, run a Docker container under the previously created `cvnode` workspace directory:

```bash
./src/cvnode_base/examples/mask_rcnn/run-docker.sh
```

**NOTE:** In case you have built the image manually, e.g. with name `ros2-humble-cuda-torch`, run:

```bash
DOCKER_IMAGE=ros2-humble-cuda-torch ./src/cvnode_base/examples/mask_rcnn/run-docker.sh
```

This script starts the image with:

* `-v $(pwd):/data` - mounts current (`cvnode`) directory in the `/data` directory in the container's context
* `-v /tmp/.X11-unix/:/tmp/.X11-unix/` - passes the X11 socket directory to the container's context (to allow running GUI application)
* `-e DISPLAY=$DISPLAY`, `-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR` - adds X11-related environment variables
* `--gpus='all,"capabilities=compute,utility,graphics,display"'` - adds GPUs to the container's context for compute and display purposes

Then, in the Docker container, install graphics libraries for NVIDIA that match your host's drivers.
To check NVIDIA driver versions, run:

```bash
nvidia-smi
```

And check `Driver version`.

For example, for 530.41.03, install the following in the container:

```bash
apt-get update && apt-get install libnvidia-gl-530
```

Then, go to the workspace directory in the container:

```bash
cd /data
```

Finally, install `Kenning`:

```bash
pip install kenning/
```

## Exporting MaskRCNN to TorchScript

MaskRCNN model can be exported to TorchScript with the `export-model.py` script.
The script takes the following arguments:

* `--image` - path to the image to run inference on
* `--output` - path to the directory where the exported model will be stored
* `--method` - method for model export. Should be one of: onnx, torchscript
* `--num-classes` - optional argument indicating amount of classes to use in model architecture
* `--weights` - optional argument indicating path to the file storage weights.
    By default, fetches COCO pre-trained model weights from model zoo.

For example, to export the model to `TorchScript` and locate it in the `config` directory, run:

```bash
curl http://images.cocodataset.org/val2017/000000000632.jpg --output image.jpg

/data/src/cvnode_base/examples/mask_rcnn/export-model.py \
    --image image.jpg \
    --output /data/src/cvnode_base/examples/config \
    --method torchscript
```

This will download an image from the COCO dataset and export the model to the `config` directory.
Later, the model can be loaded with the `mask_rcnn_torchscript_launch.py` launch file.

## Building the MaskRCNN demo

First, source the ROS2 environment:

```bash
source /opt/ros/setup.sh
```

Then, build the GUI node and the Camera node:

```bash
colcon build --base-path=src/ --packages-select \
    kenning_computer_vision_msgs \
    cvnode_base \
    cvnode_manager \
    --cmake-args ' -DBUILD_GUI=ON' ' -DBUILD_MASK_RCNN=ON ' ' -DBUILD_MASK_RCNN_TORCHSCRIPT=ON' ' -DBUILD_TORCHVISION=ON'
```

Here, the `--cmake-args` are:

* `-DBUILD_GUI=ON` - builds the GUI for CVNodeManager
* `-D BUILD_MASK_RCNN=ON and ' -DBUILD_MASK_RCNN_TORCHSCRIPT=ON` - builds the MaskRCNN demos
* `-DBUILD_TORCHVISION=ON` - builds the TorchVision library needed for MaskRCNN

Source the build targets with:

```bash
source install/setup.sh
```

## Running the MaskRCNN demo

`CVNode` provides two launch scripts for running the demo:

* `mask_rcnn_detectron_launch.py` - runs the MaskRCNN node with Python Detectron2 backend
* `mask_rcnn_torchscript_launch.py` - runs the MaskRCNN node with C++ TorchScript backend

You can run a sample launch with a Python backend with:

```bash
ros2 launch cvnode_base mask_rcnn_detectron_launch.py \
    class_names_path:=/data/src/cvnode_base/examples/config/coco_classes.csv \
    inference_configuration:=/data/src/cvnode_base/examples/config/coco_inference.json \
    publish_visualizations:=True \
    preserve_output:=False \
    scenario:=real_world_last \
    inference_timeout_ms:=100 \
    measurements:=/data/build/ros2_detectron_measurements.json \
    report_path:=/data/build/reports/detectron_real_world_last/report.md \
    log_level:=INFO
```

For a C++ backend, run:

```bash
ros2 launch cvnode_base mask_rcnn_torchscript_launch.py \
    model_path:=/data/src/cvnode_base/examples/config/model.ts \
    class_names_path:=/data/src/cvnode_base/examples/config/coco_classes.csv \
    inference_configuration:=/data/src/cvnode_base/examples/config/coco_inference.json \
    publish_visualizations:=True \
    preserve_output:=False \
    scenario:=real_world_last \
    inference_timeout_ms:=100 \
    measurements:=/data/build/ros2_torchscript_measurements.json \
    report_path:=/data/build/reports/torchscript_real_world_last/report.md \
    log_level:=INFO
```

Here, the parameters are:

* `model_path` - path to a TorchScript model
* `class_names_path` - path to a CSV file with class names
* `inference_configuration` - path to a JSON file with Kenning's inference configuration
* `publish_visualizations` - whether to publish visualizations for the GUI
* `preserve_output` - whether to preserve the output of the last inference if timeout is reached
* `scenario` - scenario for running the demo, one of:
    * `real_world_last` - tries to process last received frame within timeout
    * `real_world_first` - tries to process first received frame
    * `synthetic` - ignores timeout and processes frames as fast as possible
* `inference_timeout_ms` - timeout for inference in milliseconds. Used only by `real_world` scenarios
* `measurements` - path to file where inference measurements will be stored
* `report_path` - path to file where the rendered report will be stored
* `log_level` - log level for running the demo.

The produced reports can later be found in the `/data/build/reports` directory.
