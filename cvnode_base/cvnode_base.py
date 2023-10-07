# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for computer vision nodes."""

from typing import List

from kenning_computer_vision_msgs.msg import SegmentationMsg
from kenning_computer_vision_msgs.srv import ManageCVNode, SegmentCVNodeSrv
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class BaseCVNode(Node):
    """
    Base class for tested computer vision nodes.

    Responsible for process with the CVNodeManager
    and running inference.
    """

    def __init__(self, node_name: str):
        """
        Initialize the node.

        Parameters
        ----------
        node_name : str
            Name of the node.
        """
        # Service client for node management
        self._manage_node_client = None

        # Service responsible for communication with the CVNodeManager
        self._prepare_service = None
        self._process_service = None
        self._cleanup_service = None

        super().__init__(node_name)

        self.registerNode('cvnode_register')

    def registerNode(self, manage_service_name: str):
        """
        Register node with the manage service.

        Parameters
        ----------
        manage_service_name : str
            Name of the service for nodes management.
        """
        self.get_logger().debug('Registering node')
        if self._manage_node_client:
            self._unregisterNode()

        self._manage_node_client = self.create_client(ManageCVNode,
                                                      manage_service_name)
        if not self._manage_node_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                'Node manage service not available')
            return

        prepare_service_name = f'{self.get_name()}/prepare'
        process_service_name = f'{self.get_name()}/process'
        cleanup_service_name = f'{self.get_name()}/cleanup'

        # Initialize services
        self._prepare_service = self.create_service(
            Trigger,
            prepare_service_name,
            self._prepareCallback
        )
        qos_profile = QoSProfile(depth=1)
        self._process_service = self.create_service(
            SegmentCVNodeSrv,
            process_service_name,
            self._processCallback,
            qos_profile=qos_profile)
        self._cleanup_service = self.create_service(
            Trigger,
            cleanup_service_name,
            self._cleanupCallback
        )

        # Create request for manage service
        request = ManageCVNode.Request()
        request.type = request.REGISTER
        request.node_name = self.get_name()
        request.prepare_srv_name = prepare_service_name
        request.process_srv_name = process_service_name
        request.cleanup_srv_name = cleanup_service_name

        def register_callback(future):
            response = future.result()
            self.get_logger().debug('Received register response')
            if not response:
                self.get_logger().error('Service call failed.')
                return

            if not response.status:
                error_msg = 'Register service call failed: ' + \
                    response.message
                self.get_logger().error(error_msg)
                return

            self.get_logger().debug('Register service call succeeded')
            return

        future = self._manage_node_client.call_async(request)
        future.add_done_callback(register_callback)

    def destroy_node(self):
        """
        Destroy node.

        Unregisters node if was registered.
        """
        self.get_logger().debug('Destroying node')
        if self._manage_node_client:
            self._unregisterNode()
        super().destroy_node()

    def prepare(self) -> bool:
        """
        Prepare node and model for inference.

        Returns
        -------
        bool :
            True if preparation was successful, False otherwise.
        """
        raise NotImplementedError

    def run_inference(self, X: List[Image]) -> List[SegmentationMsg]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : List[Image]
            List of input image messages.

        Returns
        -------
        List[SegmentationMsg] :
            List of postprocessed segmentation messages.
        """
        raise NotImplementedError

    def cleanup(self):
        """Cleanup allocated resources used by the node."""
        raise NotImplementedError

    def _unregisterNode(self):
        """Unregister node with service."""
        self.get_logger().debug('Unregistering node')
        if self._manage_node_client is None:
            self.get_logger().warn('Node was not registered')
            return
        request = ManageCVNode.Request()
        request.type = request.UNREGISTER
        request.node_name = self.get_name()

        self._manage_node_client.call_async(request)
        self._manage_node_client = None
        self.get_logger().debug('Node unregistered')

    def _prepareCallback(self, request: Trigger.Request,
                         response: Trigger.Response) -> Trigger.Response:
        """
        Callback for the prepare service.

        Responsible for preparing node for inference.

        Parameters
        ----------
        request : Trigger.Request
            Request for the prepare service.
        response : Trigger.Response
            Processed response for the prepare service client.

        Returns
        -------
        Trigger.Response :
            Processed response for the prepare service client.
        """
        self.get_logger().debug('Preparing node')
        if not self.prepare():
            self.get_logger().error('Node preparation failed')
            response.success = False
            return response
        response.success = True
        self.get_logger().debug('Node prepared')
        return response

    def _processCallback(self, request: SegmentCVNodeSrv.Request,
                         response: SegmentCVNodeSrv.Response
                         ) -> SegmentCVNodeSrv.Response:
        """
        Callback for the process service.

        Responsible for running inference on the input data.

        Parameters
        ----------
        request : SegmentCVNodeSrv.Request
            Request for the process service.
        response : SegmentCVNodeSrv.Response
            Processed response for the process service client.

        Returns
        -------
        SegmentCVNodeSrv.Response :
            Processed response for the process service client.
        """
        self.get_logger().debug('Executing inference on input data')
        response.output = self.run_inference(request.input)
        response.success = True
        self.get_logger().debug('Inference executed')
        return response

    def _cleanupCallback(self, request: Trigger.Request,
                         response: Trigger.Response) -> Trigger.Response:
        """
        Callback for the cleanup service.

        Responsible for cleaning up node's resources.

        Parameters
        ----------
        request : Trigger.Request
            Request for the cleanup service.
        response : Trigger.Response
            Processed response for the cleanup service client.

        Returns
        -------
        Trigger.Response :
            Processed response for the cleanup service client.
        """
        self.get_logger().debug('Cleaning up node')
        self.cleanup()
        response.success = True
        return response
