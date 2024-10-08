#!/bin/bash

# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# This script runs the Docker container for ROS 2 and CVNode integration

set -e

DOCKER_IMAGE=${DOCKER_IMAGE:-ros2-vision-node-base}

docker run -it \
    -v "$(pwd)":"$(pwd)" \
    -w "$(pwd)" \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    --gpus='all,"capabilities=compute,utility,graphics,display"' \
    -e DISPLAY="$DISPLAY" \
    -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
    "$DOCKER_IMAGE" \
    /bin/bash
