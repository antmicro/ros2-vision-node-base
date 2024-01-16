# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Contains utility functions to work with TensorRT."""

import ctypes
from typing import Any, Callable, Union

import numpy as np
from cuda import cuda, cudart


def check_cuda_err(err: Union[cuda.CUresult, cudart.cudaError_t]):
    """
    Checks the return value of a CUDA call and raises an exception if it
    failed.

    Parameters
    ----------
    err : Union[cuda.CUresult, cudart.cudaError_t]
        The return value of a CUDA call.

    Raises
    ------
    RuntimeError
        If the CUDA call failed.
    """
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call: Callable) -> Any:
    """
    Calls a CUDA function and checks the return value.

    Parameters
    ----------
    call : Callable
        The CUDA function to call.

    Returns
    -------
    Any
        The return value of the CUDA function.
    """
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMemory:
    """
    Class representing a pair of host and device memory with the same size and
    data type.

    Parameters
    ----------
    size : int
        The size of the memory in elements.
    dtype : np.dtype
        The data type of the memory.
    """

    def __init__(self, size: int, dtype: np.dtype):
        self.nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(self.nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(
            ctypes.cast(host_mem, pointer_type), (size,)
        )
        self.device = cuda_call(cudart.cudaMalloc(self.nbytes))

    @property
    def host(self) -> np.ndarray:
        """
        The host memory.

        Returns
        -------
        np.ndarray
            The host memory.
        """
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        """
        Sets the host memory.

        Parameters
        ----------
        arr : np.ndarray
            The host memory.

        Raises
        ------
        ValueError
            If the array size is larger than the memory size.
        """
        if arr.size > self.host.size:
            raise ValueError(
                f"Array size ({arr.size}) is larger than "
                f"the memory size ({self.host.size})"
            )
        np.copyto(self.host[: arr.size], arr.flat, casting="safe")

    def free(self):
        """
        Frees the host and device memory.
        """
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))
