# Copyright (c) 2022-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""Contains utility functions to work with TensorRT."""

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


def memcpy_host_to_device(dst_ptr: int, src_arr: np.ndarray):
    """
    Copies data from host to device.

    Parameters
    ----------
    dst_ptr : int
        The destination memory pointer.
    src_arr : np.ndarray
        The source memory.
    """
    nbytes = src_arr.size * src_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            dst_ptr,
            src_arr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
    )


def memcpy_device_to_host(dst_arr: np.ndarray, src_ptr: int):
    """
    Copies data from device to host.

    Parameters
    ----------
    dst_arr : np.ndarray
        The destination memory.
    src_ptr : int
        The source memory pointer.
    """
    nbytes = dst_arr.size * dst_arr.itemsize
    cuda_call(
        cudart.cudaMemcpy(
            dst_arr,
            src_ptr,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
    )
