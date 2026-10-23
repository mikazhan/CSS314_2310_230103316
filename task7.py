
import os
import time
import math
import subprocess
import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103316
N = 30_000_000
REPEATS = 3

def get_physical_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())

@njit(nogil=True)
def memory_kernel(a, b, start, end):
    for i in range(start, end):
        b[i] = a[i]

@njit(nogil=True)
def compute_kernel(a, b, start, end):
    for i in range(start, end):
        x = a[i]
        b[i] = math.sin(x) * math.cos(x) + math.tan(x)

def run_kernel(pool, kernel, a, b, workers):
    n = len(a)
    futures = []

    for i in range(workers):
        start = i * n // workers
        end = (i + 1) * n // workers
        futures.append(
            pool.submit(kernel, a, b, start, end)
        )

    for future in futures:
        future.result()

def benchmark(pool, kernel, a, b, workers):
    run_kernel(pool, kernel, a, b, workers)
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        run_kernel(pool, kernel, a, b, workers)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return float(np.median(times))

def test_kernel(name, kernel, a, b, worker_counts, pool):
    print(f"\n{name}")
    print(
        f"{'Cores':<10}"
        f"{'Wall Time (ms)':<20}"
        f"{'Speedup':<15}"
        f"{'Efficiency':<15}"
    )
    print("-" * 60)

    results = {}
    baseline = None

    for workers in worker_counts:
        elapsed = benchmark(pool, kernel, a, b, workers)

        if baseline is None:
            baseline = elapsed

        speedup = baseline / elapsed
        efficiency = speedup / workers

        results[workers] = {
            "time": elapsed,
            "speedup": speedup,
            "efficiency": efficiency
        }

        print(
            f"{workers:<10}"
            f"{elapsed:<20.2f}"
            f"{speedup:<15.2f}"
            f"{efficiency * 100:<14.2f}%"
        )

    return results

def main():
    cores = get_physical_cores()
    logical = os.cpu_count() or 1
    worker_counts = list(dict.fromkeys([1, 2, cores]))

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random(N, dtype=np.float64)
    b = np.empty_like(a)

    memory_kernel(a, b, 0, 1)
    compute_kernel(a, b, 0, 1)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {logical}")
    print(f"Physical cores: {cores}")
    print(f"Array size: {N:,}")
    print(f"Data type: {a.dtype}")
    print(f"Repetitions: {REPEATS}")

    with ThreadPoolExecutor(max_workers=cores) as pool:
        memory_results = test_kernel(
            "KERNEL A: MEMORY-BOUND",
            memory_kernel, a, b, worker_counts, pool
        )

        if not np.array_equal(a, b):
            raise RuntimeError("Memory kernel validation failed")

        compute_results = test_kernel(
            "KERNEL B: COMPUTE-BOUND",
            compute_kernel, a, b, worker_counts, pool
        )

        expected = np.sin(a) * np.cos(a) + np.tan(a)

        if not np.allclose(b, expected):
            raise RuntimeError("Compute kernel validation failed")

        del expected

    max_cores = worker_counts[-1]
    memory_speedup = memory_results[max_cores]["speedup"]
    compute_speedup = compute_results[max_cores]["speedup"]

    print("\nSCALING ANALYSIS")
    print(f"Memory kernel speedup:  {memory_speedup:.2f}x")
    print(f"Compute kernel speedup: {compute_speedup:.2f}x")

    memory_eff = memory_results[max_cores]["efficiency"]
    compute_eff = compute_results[max_cores]["efficiency"]

    if memory_eff < compute_eff:
        print("Kernel A has lower scaling efficiency.")
        print("Memory bandwidth saturation is a possible cause.")
    elif compute_eff < memory_eff:
        print("Kernel B has lower scaling efficiency.")
        print("Memory saturation is not established by these results.")
    else:
        print("Both kernels have similar scaling efficiency.")

    two_cores = worker_counts[1]

    if memory_results[max_cores]["time"] >= memory_results[two_cores]["time"]:
        print("Memory kernel stalled between 2 and max cores.")
    else:
        print("Memory kernel continued improving up to max cores.")

    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
