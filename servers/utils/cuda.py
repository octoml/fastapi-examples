import ctypes

from cuda import cuda, cudart, nvrtc


def _cuda_parse_error(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def check_cuda_result(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cuda_parse_error(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class CudaMem:
    _initialized = False

    def __init__(self):
        if not CudaMem._initialized:
            check_cuda_result(cuda.cuInit(0))
            CudaMem._initialized = True
        self.cuda_ctx = check_cuda_result(cuda.cuCtxCreate(0, 0))

    def measure(self):
        mem_pool = check_cuda_result(cuda.cuDeviceGetDefaultMemPool(0))
        mem_pool_curr = check_cuda_result(
            cuda.cuMemPoolGetAttribute(
                mem_pool, cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_CURRENT
            )
        )
        mem_pool_high = check_cuda_result(
            cuda.cuMemPoolGetAttribute(
                mem_pool, cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH
            )
        )
        check_cuda_result(
            cuda.cuMemPoolSetAttribute(
                mem_pool,
                cuda.CUmemPool_attribute.CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                cuda.cuuint64_t(ctypes.c_uint64(0).value),
            )
        )

        graphmem_curr = check_cuda_result(
            cuda.cuDeviceGetGraphMemAttribute(
                0, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT
            )
        )
        graphmem_high = check_cuda_result(
            cuda.cuDeviceGetGraphMemAttribute(
                0, cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH
            )
        )
        check_cuda_result(
            cuda.cuDeviceSetGraphMemAttribute(
                0,
                cuda.CUgraphMem_attribute.CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                cuda.cuuint64_t(ctypes.c_uint64(0).value),
            )
        )

        mem_result = check_cuda_result(cuda.cuMemGetInfo())
        mem_free = mem_result[0]
        mem_total = mem_result[1]
        return (mem_free, mem_total)
