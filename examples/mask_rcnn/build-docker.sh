#!/bin/bash

# Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# This script builds the Docker container for ROS 2 and Kenning integration

DOCKER_TAG=${DOCKER_TAG:-ghcr.io/antmicro/ros2-cvnode-base:ros2-humble-cuda-torch}

SCRIPTDIR=$(dirname "$(realpath "$0")")

pushd "${SCRIPTDIR}" || exit

mkdir -p third-party/

if [ ! -d "third-party/vision_opencv" ]
then
    git clone --recursive https://github.com/ros-perception/vision_opencv --branch humble third-party/vision_opencv
fi

docker build . --tag "${DOCKER_TAG}"

popd || exit
