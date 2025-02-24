#!/usr/bin/env python3
import numpy as np
import sys, os
from pathlib import Path
CUR_DIR = Path(os.path.dirname(__file__))
sys.path.insert(0, str(CUR_DIR/'..'))
from launch_metal_kernel import launch_metal_kernel


def main():
    d = np.arange(64).reshape(8, 8).astype(np.float32)

    BLOCK_SIZE = 4
    grid_dim = (d.shape[0]//BLOCK_SIZE, d.shape[1]//BLOCK_SIZE, 1)
    block_dim = (BLOCK_SIZE, BLOCK_SIZE, 1)
    print(f'{grid_dim=}, {block_dim=}')
    (flattened_result,), secs = launch_metal_kernel(
        kernel_file=CUR_DIR/'01_transpose_tiles.metal',
        grid_dim=grid_dim,
        block_dim=block_dim,
        arg_arrays=[d],
        arg_ints=[*d.shape],
    )
    result = flattened_result.reshape(d.shape)
    print(result)

    print(f'\nTransposed all {block_dim[0]}x{block_dim[1]} tiles.')
    print(f'Seconds elapsed: {round(secs, 4)}')


if __name__ == '__main__':
    main()
