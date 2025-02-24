#!/usr/bin/env python3
import numpy as np
import math
import sys, os
from pathlib import Path
CUR_DIR = Path(os.path.dirname(__file__))
sys.path.insert(0, str(CUR_DIR/'..'))
from launch_metal_kernel import launch_metal_kernel


# Perf with N = 2**23
# Single thread (not parallel): 0.8s
# Simple parallel: 0.003s
# Smart parallel: 0.001s
# Smart parallel, fewer threads: 0.0005s
def main():
    N = 2**23
    A = np.random.rand(N).astype(np.float32)

    FEWER_THREADS = True
    BLOCK_SIZE = 128
    grid_dim = (math.ceil(N / (2*BLOCK_SIZE if FEWER_THREADS else BLOCK_SIZE)), 1, 1)
    (result, _), secs = launch_metal_kernel(
        kernel_file=CUR_DIR/('03_sum_reduction__smart_parallel'+('__fewer_threads' if FEWER_THREADS else '')+'.metal'),
        grid_dim=grid_dim,
        block_dim=(BLOCK_SIZE, 1, 1),
        arg_arrays=[np.zeros((grid_dim[0],)).astype(np.float32), A],
        arg_ints=[N],
    )
    assert result.shape == (grid_dim[0],)
    result = result.sum()

    assert math.isclose(result, A.sum(), rel_tol=1e-4), f'Result {result} differs from expected {A.sum()}'
    print(f'Correct result. Seconds elapsed: {round(secs, 4)}')


if __name__ == '__main__':
    main()
