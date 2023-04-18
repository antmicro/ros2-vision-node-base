from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from cvnode_msgs.srv import RegisterCVNode, UnregisterCVNode

from rclpy.node import Node

from std_srvs.srv import Trigger


T = TypeVar('T')
T.Request = TypeVar('T.Request')
T.Response = TypeVar('T.Response')


class BaseCVNode(Node, ABC, Generic[T]):
    """
    Base class for tested computer vision node.

    Notes
    -----
    Generic[T] is used to specify the type of the process service.

    """

    def __init__(self, node_name: str, srv_type: Generic[T]):
        """
        Initializes the node.

        Parameters
        ----------
        node_name : str
            Name of the node.
        srv_type : Generic[T]
            Type of the process service.

        """
        self._T = srv_type

        # Service clients for node registration and unregistration
        self._register_client, self._unregister_client = None, None

        # Services responsible for node preparation and data processing
        self._prepare_service, self._process_service = None, None

        # Service responsible for node cleanup
        self._cleanup_service = None

        super().__init__(node_name)

    def registerNode(self, register_service_name: str,
                     unregister_service: str):
        """
        Registers the node with the register service.

        Parameters
        ----------
        register_service_name : str
            Name of the register service.
        unregister_service : str
            Name of the unregister service.

        """
        _register_client = self.create_client(RegisterCVNode,
                                              register_service_name)

        if not _register_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Register service not available')
            return

        if self._unregister_client:
            self._unregisterNode()

        prepare_service_name = self.get_name() + '_prepare'
        process_service_name = self.get_name() + '_process'
        cleanup_service_name = self.get_name() + '_cleanup'

        # Initialize services
        self._prepare_service = self.create_service(Trigger,
                                                    prepare_service_name,
                                                    self._prepareCallback)

        self._process_service = self.create_service(self._T,
                                                    process_service_name,
                                                    self._processCallback)

        self._cleanup_service = self.create_service(Trigger,
                                                    cleanup_service_name,
                                                    self._cleanupCallback)

        # Create request for register service
        request = RegisterCVNode.Request()
        request.node_name = self.get_name()
        request.prepare_service_name = prepare_service_name
        request.process_service_name = process_service_name
        request.cleanup_service_name = cleanup_service_name

        def register_callback(future):
            response = future.result()
            if not response:
                self.get_logger().error('Service call failed.')
                return

            if not response.status:
                self.get_logger().error(f'Register service call failed: \
                        {response.message}')
                return

            self.get_logger().info(f'Register service call succeeded: \
                    {response.message}')
            self._unregister_client = self.create_client(UnregisterCVNode,
                                                         unregister_service)
            return

        future = _register_client.call_async(request)
        future.add_done_callback(register_callback)

    def destroy_node(self):
        """Unregisters the node with unregister service and destroys it."""
        if self._unregister_client:
            self._unregisterNode()
        super().destroy_node()

    def _unregisterNode(self):
        """Unregisters the node with the unregister service."""
        request = UnregisterCVNode.Request()
        request.node_name = self.get_name()
        self._unregister_client.call_async(request)
        self._unregister_client = None
        self._register_client = None

    @abstractmethod
    def _prepareCallback(self, request: Trigger.Request,
                         response: Trigger.Response) -> Trigger.Response:
        """
        Prepares the node for computer vision processing.

        Allocates resources of the node and prepares for processing.

        Parameters
        ----------
        request : std_srvs.srv.Trigger.Request
            Request for the prepare service.
        response : std_srvs.srv.Trigger.Response
            Response for the prepare service.

        Returns
        -------
        std_srvs.srv.Trigger.Response:
            Response from the prepare service.

        """
        raise NotImplementedError

    @abstractmethod
    def _processCallback(self, request: T.Request,
                         response: T.Response) -> T.Response:
        """
        Processes the data with computer vision algorithms.

        Parameters
        ----------
        request : T.Request
            Request for the process service.
        response : T.Response
            Response for the process service.

        Returns
        -------
        T.Response:
            Response from the process service.

        """
        raise NotImplementedError

    @abstractmethod
    def _cleanupCallback(self, request: Trigger.Request,
                         response: Trigger.Response) -> Trigger.Response:
        """
        Deallocates resources of the node.

        Parameters
        ----------
        request : std_srvs.srv.Trigger.Request
            Request for the cleanup service.
        response : std_srvs.srv.Trigger.Response
            Response for the cleanup service.

        Returns
        -------
        std_srvs.srv.Trigger.Response:
            Response from the cleanup service.

        """
        raise NotImplementedError
