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
    uint xOffset = 2 * gid.x * blockdim.x + lid.x; // 2*gid.x because the odd tiles are already calculated by the one on their left.
    uint yOffset = gid.y * blockdim.y + lid.y;
    uint baseIdx = (yOffset * N) + xOffset;

    if (xOffset >= N || yOffset >= N) {
        return;
    }

    threadgroup float A_shared_tile[TILE_SIDE * TILE_SIDE];
    threadgroup float B_shared_tile_left[TILE_SIDE * TILE_SIDE];
    threadgroup float B_shared_tile_right[TILE_SIDE * TILE_SIDE];

    float A_prefetch = A[lid.x + (yOffset * N)];
    float B_left_prefetch = B[N * lid.y + xOffset];
    float B_right_prefetch = B[N * lid.y + xOffset + TILE_SIDE];

    float acc_left = 0;
    float acc_right = 0;
    const uint numberOfTiles = metal::ceil(float(N)/TILE_SIDE);
    // We skip half the tiles because they're calculated by another block (the one on their left).
    for (uint nextTileIdx = 1; nextTileIdx <= numberOfTiles; nextTileIdx++) {
        A_shared_tile[lid.x + lid.y * TILE_SIDE] = A_prefetch;
        B_shared_tile_left[lid.x + lid.y * TILE_SIDE] = B_left_prefetch;
        B_shared_tile_right[lid.x + lid.y * TILE_SIDE] = B_right_prefetch;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        // Prefetch next tile so that we can hide the cost of load from global memory
        // with the computations on the current tile.
        if (nextTileIdx < numberOfTiles) { // if it's the last iteration, we don't prefetch anything
            A_prefetch = A[(nextTileIdx * TILE_SIDE) + lid.x + (yOffset * N)];
            B_left_prefetch = B[N*((nextTileIdx * TILE_SIDE) + lid.y) + xOffset];
            B_right_prefetch = B[N*((nextTileIdx * TILE_SIDE) + lid.y) + xOffset + TILE_SIDE];
        }

        for (uint k = 0; k < TILE_SIDE; k++) {
            acc_left += A_shared_tile[k + (lid.y * TILE_SIDE)] * B_shared_tile_left[(k * TILE_SIDE) + lid.x];
            acc_right += A_shared_tile[k + (lid.y * TILE_SIDE)] * B_shared_tile_right[(k * TILE_SIDE) + lid.x];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    RESULT[baseIdx] = acc_left;
    RESULT[baseIdx+TILE_SIDE] = acc_right;
}
