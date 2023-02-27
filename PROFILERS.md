We'd like to offer peak GPU mem usage as a metric reported by benchmarking.
While this will be valuable for all runtimes, the initial focus at this time is on ONNX Runtime.

## 1. CUDA API
The easiest and cheapest option is to use the Nvidia [Cuda-Python](https://nvidia.github.io/cuda-python/overview.html) library to query memory usage via **cudaMemGetInfo** before and after model load and benchmarking to determine usage.

The key caveat is that this is a **sampling** approach and it is theoretically possible that peak memory usage is higher than the memory usage we measure at the end of forward pass(es).

The ideal approach would be to hook into memory allocation hooks instead of sampling. The CUDA APIs offer high watermark tracking for [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) and [CUDA Memory Pools](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html). However, these are fairly specialized and do not work for allocations via cuMemAlloc etc.

## 2. Nvidia Profilers
Nvidia offers multiple profiling tools.

### nvprof
This is legacy and has been retired and replaced by the NSight tools below.

###  ncu - NSight Compute Command Line Profiler
This support metrics for GPU memory utilization per node - however tracking usage is challenging.
For example, we can see the following for a Concat Op in Yolo
```
   export MODEL_EXECUTION_PROVIDER=CUDAExecutionProvider
   /usr/local/NVIDIA-Nsight-Compute/ncu --metrics group:memory__dram_table /home/giqbal/Source/fastapi-examples/.venv/bin/python -m servers.yolov5
   
   void onnxruntime::cuda::_ConcatKernel<int>(onnxruntime::cuda::fast_divmod, onnxruntime::cuda::fast_divmod, const long *, const long *,    const long *, T1 *, const void **, int) (2092, 1, 1)x(256, 1, 1), Context 1, Stream 13, Device 0, CC 8.6
    Section: Command line profiler metrics
    --------------------------------------------------- ------------ ------------
    Metric Name                                          Metric Unit Metric Value
    --------------------------------------------------- ------------ ------------
    dram__bytes_read.sum                                       Mbyte         8.77
    dram__bytes_read.sum.pct_of_peak_sustained_elapsed             %        49.65
    dram__bytes_read.sum.per_second                     Gbyte/second       215.62
    dram__bytes_write.sum                                      Mbyte         6.50
    dram__bytes_write.sum.pct_of_peak_sustained_elapsed            %        36.82
    dram__bytes_write.sum.per_second                    Gbyte/second       159.91
    dram__sectors_read.sum                                    sector      274,056
    dram__sectors_write.sum                                   sector      203,240

   /usr/local/NVIDIA-Nsight-Compute/ncu --metrics group:memory__chart /home/giqbal/Source/fastapi-examples/.venv/bin/python -m servers.yolov5
```

### nsys - NSight Systems Profiler
The NSight Systems profiler offers more promising options. It collects metrics in a proprietory format which can then be exported to a SQLite database.

Example command lines

   export MODEL_EXECUTION_PROVIDER=CUDAExecutionProvider
   /usr/local/bin/nsys profile --cuda-graph-trace graph --cuda-memory-usage true /home/giqbal/Source/fastapi-examples/.venv/bin/python -m servers.yolov5
   /usr/local/bin/nsys profile --cuda-graph-trace node --cuda-memory-usage true --trace cuda,opengl,nvtx,osrt,cublas,cudnn,openmp /home/giqbal/Source/fastapi-examples/.venv/bin/python -m servers.yolov5

Export to SQLite via

   /usr/local/bin/nsys export --type sqlite report1.nsys-rep 

Query DB via
   sqlite3 report1.sqlite

   .tables
   .schema CUDA_GPU_MEMORY_USAGE_EVENTS

   ENUM_CUDA_MEM_KIND
   0|CUDA_MEMOPR_MEMORY_KIND_PAGEABLE|Pageable
   1|CUDA_MEMOPR_MEMORY_KIND_PINNED|Pinned
   2|CUDA_MEMOPR_MEMORY_KIND_DEVICE|Device
   3|CUDA_MEMOPR_MEMORY_KIND_ARRAY|Array
   4|CUDA_MEMOPR_MEMORY_KIND_MANAGED|Managed
   5|CUDA_MEMOPR_MEMORY_KIND_DEVICE_STATIC|Device Static
   6|CUDA_MEMOPR_MEMORY_KIND_MANAGED_STATIC|Managed Static
   7|CUDA_MEMOPR_MEMORY_KIND_UNKNOWN|Unknown

   ENUM_CUDA_DEV_MEM_EVENT_OPER;
   0|CUDA_DEV_MEM_EVENT_OPR_ALLOCATION|Allocation
   1|CUDA_DEV_MEM_EVENT_OPR_DEALLOCATION|Deallocation

However, the numbers here do not match the numbers observed via **NVidia SMI** or via **cudaMemGetInfo**
   CUDA Memory: Used: 1350565888 bytes, 1288.00 MB, Free: 6501.50 MB

This query shows a peak of 371487443 bytes
   SELECT e.start, e.memKind, e.memoryOperationType, e.bytes, a.allocated, SUM(a.allocated) OVER (ORDER BY e.start) FROM CUDA_GPU_MEMORY_USAGE_EVENTS e INNER JOIN (SELECT start, CASE WHEN memoryOperationType = 0 THEN bytes ELSE -bytes END as allocated FROM CUDA_GPU_MEMORY_USAGE_EVENTS) as a ON a.start = e.start ORDER BY a.start

Total allocation are notable but the peak values do not correspond.
   select kinds.label as kind, ops.label as event, SUM(events.bytes) as bytes from CUDA_GPU_MEMORY_USAGE_EVENTS events INNER JOIN ENUM_CUDA_DEV_MEM_EVENT_OPER ops ON events.memoryOperationType = ops.id INNER JOIN ENUM_CUDA_MEM_KIND kinds ON kinds.id = events.memKind GROUP BY events.memKind, events.memoryOperationType

   Pinned|Allocation|2097160
   Pinned|Deallocation|2097160
   Device|Allocation|4327023616
   Device|Deallocation|4327023616
   Device Static|Allocation|1789643

My assumption is that the way ORT is dealing with memory (pinned unified memory?) doesn't directly translate to the CUDA memory events captured by the profiler. In any case, this doesn't seem to be a viable path.

### 3. 3rd party profiler
The last option was to look into 3rd party profilers. [Scalene](https://github.com/plasma-umass/scalene) is a great example of this.
