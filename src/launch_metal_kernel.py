import Metal
import ctypes
import numpy as np
from typing import Tuple


def launch_metal_kernel(
    kernel_file: str,
    grid_dim: tuple,
    block_dim: tuple,
    arg_arrays: list[np.ndarray],
    arg_ints: list[int],
) -> Tuple[list[np.ndarray], int]:
    with open(kernel_file, 'r') as f:
        prg = f.read()

    device = Metal.MTLCreateSystemDefaultDevice()

    options = Metal.MTLCompileOptions.new()
    library, err = device.newLibraryWithSource_options_error_(prg, options, None)
    assert err is None, str(err)

    fxn = library.newFunctionWithName_('fxn')
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    assert err is None, str(err)

    # Verify max threads per block isn't exceeded.
    max_threads_per_threadgroup = pipeline_state.maxTotalThreadsPerThreadgroup()
    block_size = 1
    for d in block_dim:
        block_size *= d
    assert block_size <= max_threads_per_threadgroup, f"Block size ({block_size}) exceeds maximum allowed threads per threadgroup ({max_threads_per_threadgroup})"

    command_queue = device.newCommandQueue()
    command_buffer = command_queue.commandBuffer()

    encoder = command_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline_state)

    get_buf_memory = lambda buf: buf.contents().as_buffer(buf.length())

    # Allocate buffers.
    m_bufs = []
    for key, m in enumerate(arg_arrays):
        assert m.dtype == np.float32
        # fp32 => 4 bytes
        m_buf, = device.newBufferWithLength_options_(4 * m.size, Metal.MTLResourceStorageModeShared),
        get_buf_memory(m_buf)[:] = m.tobytes()
        encoder.setBuffer_offset_atIndex_(m_buf, 0, key)
        m_bufs.append(m_buf)

    for key, v in enumerate(arg_ints, start=len(arg_arrays)):
        encoder.setBytes_length_atIndex_(ctypes.c_int32(v), 4, key)

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid_dim),
        Metal.MTLSizeMake(*block_dim)
    )
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    elapsed_secs = command_buffer.GPUEndTime() - command_buffer.GPUStartTime()
    flattened_results = [np.frombuffer(get_buf_memory(m_buf), dtype=np.float32) for m_buf in m_bufs]

    return flattened_results, elapsed_secs
