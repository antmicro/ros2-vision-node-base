#!/bin/bash

# Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# This script runs the Docker container for ROS 2 and CVNode integration

DOCKER_IMAGE=${DOCKER_IMAGE:-ghcr.io/antmicro/ros2-cvnode-base:ros2-humble-cuda-torch}

docker run -it \
    -v "$(pwd)":/data \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    --gpus='all,"capabilities=compute,utility,graphics,display"' \
    -e DISPLAY="$DISPLAY" \
    -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
    "$DOCKER_IMAGE" \
    /bin/bash
