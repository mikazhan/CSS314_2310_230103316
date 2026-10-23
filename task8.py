
import os
import time
import subprocess
import numpy as np
from numba import njit, prange, set_num_threads

STUDENT_ID = 230103316
N = 1024
REPEATS = 3

def get_physical_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())

@njit(parallel=True)
def non_tiled(A, B, C):
    n = A.shape[0]

    for i in prange(n):
        for k in range(n):
            a = A[i, k]
            for j in range(n):
                C[i, j] += a * B[k, j]

@njit(parallel=True)
def tiled(A, B, C, block_size):
    n = A.shape[0]
    num_blocks = (n + block_size - 1) // block_size

    for bi in prange(num_blocks):
        ii = bi * block_size
        i_end = min(ii + block_size, n)

        for jj in range(0, n, block_size):
            j_end = min(jj + block_size, n)

            for kk in range(0, n, block_size):
                k_end = min(kk + block_size, n)

                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a = A[i, k]

                        for j in range(jj, j_end):
                            C[i, j] += a * B[k, j]

def benchmark(func, A, B, block_size=None):
    times = []
    result = None

    for _ in range(REPEATS):
        C = np.zeros((N, N), dtype=np.float64)

        start = time.perf_counter()

        if block_size is None:
            func(A, B, C)
        else:
            func(A, B, C, block_size)

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        result = C

    return float(np.median(times)), result

def main():
    cores = get_physical_cores()
    logical = os.cpu_count() or 1
    set_num_threads(cores)

    rng = np.random.default_rng(STUDENT_ID)
    A = rng.random((N, N), dtype=np.float64)
    B = rng.random((N, N), dtype=np.float64)

    warm_A = A[:32, :32].copy()
    warm_B = B[:32, :32].copy()
    warm_C = np.zeros((32, 32), dtype=np.float64)

    non_tiled(warm_A, warm_B, warm_C)
    warm_C.fill(0)
    tiled(warm_A, warm_B, warm_C, 16)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {logical}")
    print(f"Physical cores: {cores}")
    print(f"Threads: {cores}")
    print(f"Matrix size: {N} x {N}")
    print("Data type: float64")
    print(f"Repeats: {REPEATS}")
    print()

    baseline_time, C0 = benchmark(non_tiled, A, B)
    tile16_time, C16 = benchmark(tiled, A, B, 16)
    tile64_time, C64 = benchmark(tiled, A, B, 64)

    reference = A @ B

    valid0 = np.allclose(C0, reference, rtol=1e-8, atol=1e-8)
    valid16 = np.allclose(C16, reference, rtol=1e-8, atol=1e-8)
    valid64 = np.allclose(C64, reference, rtol=1e-8, atol=1e-8)

    print(f"{'Method':<20}{'Wall Time (ms)':<20}{'Speedup':<15}")
    print("-" * 55)

    results = [
        ("Non-Tiled", baseline_time),
        ("Tiled 16x16", tile16_time),
        ("Tiled 64x64", tile64_time)
    ]

    for name, elapsed in results:
        speedup = baseline_time / elapsed
        print(f"{name:<20}{elapsed:<20.2f}{speedup:<15.2f}")

    print()
    print("VALIDATION")
    print(f"Non-Tiled Correct: {valid0}")
    print(f"Tiled 16x16 Correct: {valid16}")
    print(f"Tiled 64x64 Correct: {valid64}")

    if not all([valid0, valid16, valid64]):
        raise RuntimeError("Matrix multiplication validation failed")

    fastest = min(results, key=lambda x: x[1])

    print()
    print("PERFORMANCE ANALYSIS")
    print(f"Fastest method: {fastest[0]}")
    print(f"Best runtime: {fastest[1]:.2f} ms")
    print(f"Best speedup: {baseline_time / fastest[1]:.2f}x")

    print()
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
