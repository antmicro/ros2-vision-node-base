#!/bin/bash

# Copyright (c) 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# This script builds the Docker container for ROS 2 based vision nodes development.

DOCKER_TAG=${DOCKER_TAG:-ghcr.io/antmicro/ros2-vision-node-base:vision-node-base}

SCRIPTDIR=$(dirname "$(realpath "$0")")

pushd "${SCRIPTDIR}" || exit

docker build . --tag "${DOCKER_TAG}"

popd || exit
