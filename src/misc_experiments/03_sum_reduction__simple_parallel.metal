#define BLOCK_SIZE 128

kernel void fxn(
    device float* RESULT,
    const device float* A,
    constant uint& N,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 blockdim [[ threads_per_threadgroup ]]
) {
    uint baseIdx = (gid.x * blockdim.x) + lid.x;
    if (baseIdx >= N) {
        return;
    }

    threadgroup float partialSum[BLOCK_SIZE];

    partialSum[lid.x] = A[baseIdx];

    for (uint stride = 1; stride < blockdim.x; stride <<= 1) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (baseIdx % (2*stride) == 0 && (baseIdx + stride) < N) {
            partialSum[lid.x] += partialSum[lid.x + stride];
        }
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (lid.x == 0) {
        RESULT[gid.x] = partialSum[0];
    }
}
