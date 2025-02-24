#!/usr/bin/env python3
import numpy as np
from math import ceil
import sys, os
from pathlib import Path
CUR_DIR = Path(os.path.dirname(__file__))
sys.path.insert(0, str(CUR_DIR/'..'))
from launch_metal_kernel import launch_metal_kernel


"""
Performance (for N=4096):
- Naive (before tiling smem) -> 0.52s
- With tiles fetched into SMEM, TILE_SIDE=32 -> 0.33s
- With tiles fetched into SMEM, TILE_SIDE=16 -> 0.2s
- With next-tile prefetch, TILE_SIDE=16 -> 0.18s
- With prefetch + increased granularity, TILE_SIDE=16 -> 0.14s
"""
def main():
    N = 4096
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    INCREASED_GRANULARITY = True
    config = _get_launch_config(N, regular_granularity=not INCREASED_GRANULARITY)
    (flattened_result, _, _), secs = launch_metal_kernel(
        kernel_file=CUR_DIR/('02_matmul__data_prefetch'+('__increased_granularity' if INCREASED_GRANULARITY else '')+'.metal'),
        grid_dim=config['grid_dim'],
        block_dim=config['block_dim'],
        arg_arrays=[np.empty((N, N)).astype(np.float32), A, B],
        arg_ints=[N],
    )
    result = flattened_result.reshape((N, N))

    assert np.allclose(result, expected_result := A@B), f'{result=}\n{expected_result=}'
    print(f'Correct result. Seconds elapsed: {round(secs, 4)}')


def _get_launch_config(N, regular_granularity: bool):
    # If you change the block size, you *might* also need to change the TILE_SIDE in the kernel code
    BLOCK_SIDE = 16

    # Regular case
    if regular_granularity:
        return {
            'grid_dim': (ceil(N/BLOCK_SIDE), ceil(N/BLOCK_SIDE), 1),  # we need enough blocks to cover all the product matrix
            'block_dim': (BLOCK_SIDE, BLOCK_SIDE, 1),
        }

    # Increased granularity
    return {
        'grid_dim': (ceil(N/(2*BLOCK_SIDE)), ceil(N/BLOCK_SIDE), 1),  # we launch half the blocks cause each thread achieves double the results
        'block_dim': (BLOCK_SIDE, BLOCK_SIDE, 1),
    }


if __name__ == '__main__':
    main()
