
import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103316
NUM_TASKS = 10_000
HASH_REPEATS = 500
DATA = b"A" * 1024
RUNS = 3

def cpu_task(index):
    payload = DATA + str(index).encode()
    digest = payload

    for _ in range(HASH_REPEATS):
        digest = hashlib.sha256(digest).digest()

    return digest

def run_batch(pool_size):
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(cpu_task, range(NUM_TASKS)))

    elapsed = (time.perf_counter() - start) * 1000

    if len(results) != NUM_TASKS:
        raise RuntimeError("Task execution failed")

    return elapsed

def main():
    cores = os.cpu_count() or 1
    pool_sizes = list(dict.fromkeys([
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256
    ]))

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {cores}")
    print(f"Total tasks: {NUM_TASKS:,}")
    print(f"Hash repetitions per task: {HASH_REPEATS}")
    print(f"Input size: {len(DATA)} bytes")
    print(f"Benchmark runs: {RUNS}")
    print()

    print(
        f"{'Pool Size':<15}"
        f"{'Execution Time (ms)':<25}"
        f"{'Speedup':<15}"
        f"{'Status':<20}"
    )
    print("-" * 75)

    baseline = None
    previous_time = None
    first_worsening = None

    for size in pool_sizes:
        times = []

        for _ in range(RUNS):
            elapsed = run_batch(size)
            times.append(elapsed)

        median_time = statistics.median(times)

        if baseline is None:
            baseline = median_time

        speedup = baseline / median_time

        if previous_time is None:
            status = "BASELINE"
        elif median_time > previous_time:
            status = "WORSENING"

            if first_worsening is None:
                first_worsening = size
        else:
            status = "IMPROVING"

        print(
            f"{size:<15}"
            f"{median_time:<25.2f}"
            f"{speedup:<15.2f}"
            f"{status:<20}"
        )

        previous_time = median_time

    print()
    print("OVERSUBSCRIPTION ANALYSIS")

    if first_worsening is not None:
        print(f"First observed worsening: {first_worsening} threads")
    else:
        print("No performance worsening detected.")

    print()
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
