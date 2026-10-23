import os
import time
import subprocess
import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103316
SIZE = 8000

def physical_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())

@njit(nogil=True)
def row_major(matrix, start, end):
    total = 0.0
    for i in range(start, end):
        for j in range(matrix.shape[1]):
            total += matrix[i, j]
    return total

@njit(nogil=True)
def column_major(matrix, start, end):
    total = 0.0
    for j in range(start, end):
        for i in range(matrix.shape[0]):
            total += matrix[i, j]
    return total

def benchmark(matrix, workers, mode):
    function = row_major if mode == "row" else column_major
    ranges = [
        (i * SIZE // workers, (i + 1) * SIZE // workers)
        for i in range(workers)
    ]

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(function, matrix, a, b)
            for a, b in ranges
        ]
        results = [future.result() for future in futures]

    elapsed = (time.perf_counter() - start) * 1000

    return sum(results), elapsed

def main():
    cores = physical_cores()
    logical = os.cpu_count()

    rng = np.random.default_rng(STUDENT_ID)
    matrix = rng.random((SIZE, SIZE), dtype=np.float64)

    row_major(matrix, 0, 1)
    column_major(matrix, 0, 1)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {logical}")
    print(f"Physical cores: {cores}")
    print(f"Matrix size: {SIZE} x {SIZE}")
    print(f"Data type: {matrix.dtype}")
    print()

    row_sum, row_time = benchmark(matrix, cores, "row")
    col_sum, col_time = benchmark(matrix, cores, "column")

    penalty = col_time / row_time

    print(f"Mode A (Row-Major):    {row_time:.2f} ms")
    print(f"Mode B (Column-Major): {col_time:.2f} ms")
    print(f"Cache Penalty Factor:  {penalty:.2f}x")
    print()
    print(f"Row Sum: {row_sum:.6f}")
    print(f"Column Sum: {col_sum:.6f}")
    print(f"Results Match: {np.isclose(row_sum, col_sum)}")

if __name__ == "__main__":
    main()
