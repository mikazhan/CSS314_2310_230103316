
import os
import sys
import time
import subprocess
import numpy as np
from numba import njit
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103316
N = 2_000_000
REPEATS = 5

def get_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())

@njit(nogil=True)
def serial_scan(data, output):
    total = 0

    for i in range(len(data)):
        total += data[i]
        output[i] = total

@njit(nogil=True)
def local_scan(data, output, start, end):
    total = 0

    for i in range(start, end):
        total += data[i]
        output[i] = total

    return total

@njit(nogil=True)
def apply_offset(output, start, end, offset):
    for i in range(start, end):
        output[i] += offset

def parallel_scan(data, output, workers, pool):
    n = len(data)

    ranges = [
        (i * n // workers, (i + 1) * n // workers)
        for i in range(workers)
    ]

    futures = [
        pool.submit(local_scan, data, output, start, end)
        for start, end in ranges
    ]

    chunk_totals = [f.result() for f in futures]

    offsets = []
    running_total = 0

    for total in chunk_totals:
        offsets.append(running_total)
        running_total += total

    futures = [
        pool.submit(apply_offset, output, start, end, offsets[i])
        for i, (start, end) in enumerate(ranges)
    ]

    for future in futures:
        future.result()

def benchmark(function):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return float(np.median(times))

def main():
    cores = get_cores()
    logical = os.cpu_count() or 1

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.integers(1, 11, size=N, dtype=np.int64)

    serial_output = np.empty(N, dtype=np.int64)
    parallel_output = np.empty(N, dtype=np.int64)

    warm_data = np.array([1, 2, 3], dtype=np.int64)
    warm_output = np.empty_like(warm_data)

    serial_scan(warm_data, warm_output)
    local_scan(warm_data, warm_output, 0, 3)
    apply_offset(warm_output, 0, 3, 5)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {logical}")
    print(f"Physical cores: {cores}")
    print(f"Array size: {N:,}")
    print("Data type: int64")
    print(f"Workers: {cores}")
    print(f"Repeats: {REPEATS}")
    print()

    with ThreadPoolExecutor(max_workers=cores) as pool:
        serial_time = benchmark(
            lambda: serial_scan(data, serial_output)
        )

        parallel_time = benchmark(
            lambda: parallel_scan(data, parallel_output, cores, pool)
        )

    try:
        assert parallel_output[-1] == serial_output[-1]
        assert np.array_equal(parallel_output, serial_output)
        status = "PASS"
    except AssertionError:
        status = "FAIL"

    speedup = serial_time / parallel_time

    print(f"{'Serial Time (ms)':<20}{'Parallel Time (ms)':<22}{'Speedup':<12}{'Verification':<15}")
    print("-" * 70)
    print(f"{serial_time:<20.4f}{parallel_time:<22.4f}{speedup:<12.2f}{status:<15}")

    print()
    print(f"Serial final sum:   {serial_output[-1]}")
    print(f"Parallel final sum: {parallel_output[-1]}")
    print(f"Verification Status: {status}")

    if status == "FAIL":
        sys.exit(1)

    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
