# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for computer vision nodes."""

from threading import Lock
from typing import List

from kenning_computer_vision_msgs.msg import RuntimeMsgType, SegmentationMsg
from kenning_computer_vision_msgs.srv import ManageCVNode, SegmentCVNodeSrv
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image


class BaseCVNode(Node):
    """
    Base class for tested computer vision nodes.

    Responsible for communication with the CVNodeManager
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
        self._communication_service = None

        # Stores input data
        self._input_data = None

        # Stores model output data
        self._output_data = []

        # Output data lock
        self._output_data_lock = Lock()

        # Input data lock
        self._input_data_lock = Lock()

        # Request id increment lock
        self._request_id_lock = Lock()

        # Process access lock
        self._process_lock = Lock()

        # Id of request
        self._request_id = 0

        super().__init__(node_name)

        self.registerNode("node_manager/register")

    def registerNode(self, manage_service_name: str):
        """
        Register node with the manage service.

        Parameters
        ----------
        manage_service_name : str
            Name of the service for nodes management.
        """
        if self._manage_node_client:
            self._unregisterNode()

        self._manage_node_client = self.create_client(ManageCVNode,
                                                      manage_service_name)
        if not self._manage_node_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                    '[REGISTER] Node manage service not available')
            return

        communication_service_name = f'{self.get_name()}/communication'

        # Initialize services
        self._communication_service = self.create_service(
            SegmentCVNodeSrv,
            communication_service_name,
            self._communicationCallback,
            callback_group=ReentrantCallbackGroup())

        # Create request for manage service
        request = ManageCVNode.Request()
        request.type = request.REGISTER
        request.node_name = self.get_name()
        request.srv_name = communication_service_name

        def register_callback(future):
            response = future.result()
            if not response:
                self.get_logger().error('[REGISTER] Service call failed.')
                return

            if not response.status:
                error_msg = '[REGISTER] Register service call failed: ' + \
                        response.message
                self.get_logger().error(error_msg)
                return

            self.get_logger().debug(
                    '[REGISTER] Register service call succeeded')
            return

        future = self._manage_node_client.call_async(request)
        future.add_done_callback(register_callback)

    def destroy_node(self):
        """
        Destroy node.

        Unregisters node if was registered.
        """
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

    def _run_inference(self) -> SegmentCVNodeSrv.Response:
        """
        Executes inference stages and returns response.

        Returns
        -------
        SegmentCVNodeSrv.Response :
            Response to the manager node.
        """
        response = SegmentCVNodeSrv.Response()
        response.message_type = RuntimeMsgType.OK

        with self._request_id_lock:
            self._request_id += 1
            tmp_request_id = self._request_id
        with self._input_data_lock:
            tmp_input_data = self._input_data
            self._input_data = []

        if not tmp_input_data:
            return self.report_error(
                    SegmentCVNodeSrv.Response(),
                    '[PREDICT] Received empty input data')

        with self._process_lock:
            with self._request_id_lock:
                if tmp_request_id != self._request_id:
                    return response
            output = self.run_inference(tmp_input_data)

        with self._output_data_lock:
            with self._request_id_lock:
                if tmp_request_id != self._request_id:
                    return response
            self._output_data = output
            return response

    def cleanup(self):
        """
        Cleanup allocated resources used by the node.
        """
        raise NotImplementedError

    def _unregisterNode(self):
        """
        Unregister node with service.
        """
        request = ManageCVNode.Request()
        request.type = request.UNREGISTER
        request.node_name = self.get_name()

        self._manage_node_client.call_async(request)
        self._manage_node_client = None

    def report_error(self, response: SegmentCVNodeSrv.Response,
                     message: str) -> SegmentCVNodeSrv.Response:
        """
        Report error to the client.

        Parameters
        ----------
        response : SegmentCVNodeSrv.Response
            Response to the client.
        message : str
            Error message to be logged.

        Returns
        -------
        SegmentCVNodeSrv.Response :
            Response to the client with ERROR message type set.
        """
        response.message_type = RuntimeMsgType.ERROR
        self.get_logger().error(message)
        return response

    def _communicationCallback(self, request: SegmentCVNodeSrv.Request,
                               response: SegmentCVNodeSrv.Response
                               ) -> SegmentCVNodeSrv.Response:
        """
        Callback for the communication service.

        Responsible for handling CVNodeManager's request messages by invoking
        appropriate methods.

        Parameters
        ----------
        request : SegmentCVNodeSrv.Request
            Request for the communication service.
        response : SegmentCVNodeSrv.Response
            Processed response for the communication service client.

        Returns
        -------
        SegmentCVNodeSrv.Response :
            Processed response for the communication service client.
        """

        if request.message_type == RuntimeMsgType.MODEL:
            if not self.prepare():
                return self.report_error(
                        response, '[MODEL] Failed to prepare node.')
        elif request.message_type == RuntimeMsgType.DATA:
            if not request.input:
                return self.report_error(
                        response, '[DATA] Received empty data')
            with self._input_data_lock:
                self._input_data = request.input
        elif request.message_type == RuntimeMsgType.PROCESS:
            return self._run_inference()
        elif request.message_type == RuntimeMsgType.OUTPUT:
            with self._output_data_lock:
                with self._request_id_lock:
                    self._request_id += 1
                if not self._output_data:
                    self.get_logger().debug('[OUTPUT] No output data to send')
                response._output = self._output_data
                self._output_data = []
        elif request.message_type == RuntimeMsgType.ERROR:
            response = self.report_error(
                    response, '[ERROR] Received ERROR message. Cleaning up.')
            self.cleanup()
            self._unregisterNode()
            return response
        else:
            return self.report_error(
                    response,
                    '[UNKNOWN] Not supported message type: ' +
                    request.message_type)
        response.message_type = RuntimeMsgType.OK
        return response
