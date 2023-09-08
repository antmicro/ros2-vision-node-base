#!/usr/bin/env python3
# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import rclpy

from cvnode_base.nodes.mask_rcnn_detectron import MaskRCNNDetectronNode


def main(args=None):
    rclpy.init(args=args)
    node = MaskRCNNDetectronNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
