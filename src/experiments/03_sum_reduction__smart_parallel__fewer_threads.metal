#define BLOCK_SIZE 128

kernel void fxn(
    device float* RESULT,
    const device float* A,
    constant uint& N,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 blockdim [[ threads_per_threadgroup ]]
) {
    uint leftIdx = (2 * gid.x * blockdim.x) + lid.x;
    if ((leftIdx + BLOCK_SIZE) >= N) {
        return;
    }

    threadgroup float partialSum[BLOCK_SIZE*2];

    partialSum[lid.x] = A[leftIdx];
    partialSum[lid.x + BLOCK_SIZE] = A[leftIdx + BLOCK_SIZE];

    for (uint stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        if (lid.x < stride) {
            partialSum[lid.x] += partialSum[lid.x + stride];
        }
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (lid.x == 0) {
        RESULT[gid.x] = partialSum[0];
    }
}
