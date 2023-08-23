# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Base class for computer vision nodes."""

from threading import Lock
from typing import Any, List

from kenning_computer_vision_msgs.msg import SegmentationMsg
from kenning_computer_vision_msgs.srv import ManageCVNode, SegmentCVNodeSrv
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image

from cvnode_base.helpers.runtime_msg_type import RuntimeMsgType


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

        # Data access lock
        self._data_lock = Lock()

        # Process access lock
        self._process_lock = Lock()

        # Id of request
        self._request_id = 0

        super().__init__(node_name)

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
            self.get_logger().error('Node manage service not available')
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
                self.get_logger().error('Service call failed.')
                return

            if not response.status:
                self.get_logger().error(f'Register service call failed: \
                        {response.message}')
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

    def preprocess(self, X: List[Image]) -> List[Any]:
        """
        Preprocess input data.

        Parameters
        ----------
        X : List[Image]
            List of input image messages.

        Returns
        -------
        List[Any] :
            Preprocessed data compatible with model inputs.
        """
        raise NotImplementedError

    def predict(self, X: List[Any]) -> List[Any]:
        """
        Run inference on the input data.

        Parameters
        ----------
        X : List[Any]
            Preprocessed input data.

        Returns
        -------
        List[Any] :
            Model predictions compatible with post processing stage.
        """
        raise NotImplementedError

    def postprocess(self, X: List[Any]) -> List[SegmentationMsg]:
        """
        Postprocess model predictions.

        Parameters
        ----------
        X : List[Any]
            Model predictions.

        Returns
        -------
        List[SegmentationMsg] :
            List of postprocessed segmentation messages.
        """
        raise NotImplementedError

    def cleanup(self):
        """
        Cleanup allocated resources used by the node.
        """
        raise NotImplementedError

    def _unregisterNode(self):
        """Unregister node with service."""
        request = ManageCVNode.Request()
        request.type = request.UNREGISTER
        request.node_name = self.get_name()

        self._manage_node_client.call_async(request)
        self._manage_node_client = None

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

        def report_error(response: SegmentCVNodeSrv.Response,
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
                Response to the client with ERROR code.
            """
            nonlocal self
            response.message_type = RuntimeMsgType.ERROR.value
            self.get_logger().error(message)
            return response

        request_type = RuntimeMsgType(request.message_type)

        if request_type == RuntimeMsgType.OK:
            if not self.prepare():
                response = report_error(response, 'Failed to prepare node.')
                self.cleanup()
                self._unregisterNode()
                return response

        elif request_type == RuntimeMsgType.ERROR:
            response = report_error(response, 'Received ERROR message. ' +
                                    'Cleaning up.')
            self.cleanup()
            self._unregisterNode()
            return response

        elif request_type == RuntimeMsgType.MODEL:
            if not self.prepare():
                return report_error(response, 'Failed to prepare node.')

        elif request_type == RuntimeMsgType.DATA:
            if not request.input:
                return report_error(response, 'Received empty data')
            self._data_lock.acquire()
            self._input_data = request.input
            self._data_lock.release()

        elif request_type == RuntimeMsgType.PROCESS:
            self._data_lock.acquire()
            self._request_id += 1
            this_task_id = self._request_id
            preprocessed = self.preprocess(self._input_data)
            self._data_lock.release()
            if not preprocessed:
                return report_error(response, 'Preprocessing failed')
            self._data_lock.acquire()
            if this_task_id != self._request_id:
                self._data_lock.release()
                self.get_logger().debug('[PREPROCESS] Aborting processing')
                return response
            self._data_lock.release()
            self._process_lock.acquire()
            predictions = self.predict(preprocessed)
            self._process_lock.release()
            if not predictions:
                return report_error(response, 'Inference failed')
            self._data_lock.acquire()
            if this_task_id != self._request_id:
                self._data_lock.release()
                self.get_logger().debug('[PREDICT] Aborting processing')
                return response
            self._output_data = self.postprocess(predictions)
            self._input_data = None
            self._data_lock.release()

        elif request_type == RuntimeMsgType.OUTPUT:
            self._data_lock.acquire()
            self._request_id += 1
            if not self._output_data:
                self.get_logger().warn('No output data to send')
            response._output = self._output_data
            self._output_data = []
            self._data_lock.release()

        else:
            return report_error(response,
                                f'Unknown message type: {request_type}')
        response.message_type = RuntimeMsgType.OK.value
        return response
