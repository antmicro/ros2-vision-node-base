from cvnode_msgs.srv import ManageCVNode, RuntimeProtocolSrv

from rclpy.node import Node


class BaseCVNode(Node):
    """Base class for tested computer vision node."""

    def __init__(self, node_name: str):
        """
        Initializes the node.

        Parameters
        ----------
        node_name : str
            Name of the node.

        """
        # Service client for node management
        self._manage_node_client = None

        # Service responsible for communication with the CVNodeManager
        self._communication_service = None

        super().__init__(node_name)

    def registerNode(self, manage_service_name: str):
        """
        Registers the node with the manage service.

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
            RuntimeProtocolSrv,
            communication_service_name,
            self._communicationCallback
        )

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

            self.get_logger().info('Register service call succeeded')
            return

        future = self._manage_node_client.call_async(request)
        future.add_done_callback(register_callback)

    def destroy_node(self):
        """Unregisters the node with unregister service and destroys it."""
        if self._manage_node_client:
            self._unregisterNode()
        super().destroy_node()

    def _unregisterNode(self):
        """Unregisters the node with the unregister service."""
        request = ManageCVNode.Request()
        request.type = request.UNREGISTER
        request.node_name = self.get_name()

        self._manage_node_client.call_async(request)
        self._manage_node_client = None

    def _communicationCallback(self, request: RuntimeProtocolSrv.Request,
                               response: RuntimeProtocolSrv.Response
                               ) -> RuntimeProtocolSrv.Response:
        """
        Callback for the communication service.

        Responsible for handling the communication message type and invoking
        the appropriate method.

        Parameters
        ----------
        request : RuntimeProtocolSrv.Request
            Request for the communication service.
        response : RuntimeProtocolSrv.Response
            Processed response for the communication service client.

        Returns
        -------
        RuntimeProtocolSrv.Response
            Processed response for the communication service client.

        """
        raise NotImplementedError
