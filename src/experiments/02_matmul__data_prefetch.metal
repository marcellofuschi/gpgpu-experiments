#define TILE_SIDE 16

kernel void fxn(
    device float* RESULT, // NxN
    const device float* A, // NxN
    const device float* B, // NxN
    constant uint& N,
    uint3 gid [[ threadgroup_position_in_grid ]],
    uint3 lid [[ thread_position_in_threadgroup ]],
    uint3 blockdim [[ threads_per_threadgroup ]]
) {
    uint xOffset = gid.x * blockdim.x + lid.x;
    uint yOffset = gid.y * blockdim.y + lid.y;
    uint baseIdx = (yOffset * N) + xOffset;

    if (xOffset >= N || yOffset >= N) {
        return;
    }

    threadgroup float A_shared_tile[TILE_SIDE * TILE_SIDE];
    threadgroup float B_shared_tile[TILE_SIDE * TILE_SIDE];

    uint nextTileIdx = 0;
    float A_prefetch = A[(nextTileIdx * TILE_SIDE) + lid.x + (yOffset * N)];
    float B_prefetch = B[N*((nextTileIdx * TILE_SIDE) + lid.y) + xOffset];

    float acc = 0;
    for (nextTileIdx = 1; nextTileIdx < metal::ceil(float(N)/TILE_SIDE) + 1; nextTileIdx++) {
        A_shared_tile[lid.x + lid.y * TILE_SIDE] = A_prefetch;
        B_shared_tile[lid.x + lid.y * TILE_SIDE] = B_prefetch;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Prefetch next tile so that we can hide the cost of load from global memory
        // with the computations on the current tile.
        if (nextTileIdx < metal::ceil(float(N)/TILE_SIDE)) { // if it's the last iteration, there's no need to prefetch.
            A_prefetch = A[(nextTileIdx * TILE_SIDE) + lid.x + (yOffset * N)];
            B_prefetch = B[N*((nextTileIdx * TILE_SIDE) + lid.y) + xOffset];
        }

        for (uint k = 0; k < TILE_SIDE; k++) {
            acc += A_shared_tile[k + (lid.y * TILE_SIDE)] * B_shared_tile[(k * TILE_SIDE) + lid.x];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    RESULT[baseIdx] = acc;
}
