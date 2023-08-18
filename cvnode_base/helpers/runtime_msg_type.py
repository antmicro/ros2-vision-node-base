# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides enum representing message type in the communication with the
CVNodeManager ROS2 node.
"""

from enum import Enum


class RuntimeMsgType(Enum):
    """
    Enum representing message type in the communication with the CVNodeManager.

    OK - message indicating successful execution of the command.
    ERROR - message indicating failed execution of the command.
    DATA - message indicating input data.
    MODEL - message indicating request for model preparation.
    PROCESS - message indicating request for processing.
    OUTPUT - message indicating request for output data.
    STATS - message indicating request for statistics.
    IOSPEC - message indicating request for input/output specification.
    """
    OK = 0
    ERROR = 1
    DATA = 2
    MODEL = 3
    PROCESS = 4
    OUTPUT = 5
    STATS = 6
    IOSPEC = 7
