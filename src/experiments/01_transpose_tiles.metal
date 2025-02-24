kernel void fxn(
    device float* A,
    constant uint& A_d0,
    constant uint& A_d1,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 blockdim [[ threads_per_threadgroup ]]
) {
    threadgroup float blockA[4];

    uint xOffset = gid.x * blockdim.x + lid.x;
    uint yOffset = gid.y * blockdim.y + lid.y;
    uint baseIdx = yOffset * A_d1 + xOffset;

    blockA[lid.y * blockdim.y + lid.x] = A[baseIdx];
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    A[baseIdx] = blockA[lid.x * blockdim.x + lid.y];
}
