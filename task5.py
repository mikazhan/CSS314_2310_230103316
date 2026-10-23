import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit

STUDENT_ID = 230103316
SIZES = [100, 1000, 10000, 100000, 1000000, 10000000]
A = 2.5
B = 1.5
REPEATS = 7

def get_physical_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())

@njit(nogil=True)
def vector_operation(x, y, a, b, start, end):
    for i in range(start, end):
        y[i] = a * x[i] + b

def run_serial(x, y):
    vector_operation(x, y, A, B, 0, len(x))

def run_parallel(x, y, pool, workers):
    n = len(x)
    futures = []

    for i in range(workers):
        start = i * n // workers
        end = (i + 1) * n // workers

        futures.append(
            pool.submit(
                vector_operation,
                x, y, A, B, start, end
            )
        )

    for future in futures:
        future.result()

def measure(function):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return float(np.median(times))

def main():
    cores = get_physical_cores()
    logical = os.cpu_count() or 1
    rng = np.random.default_rng(STUDENT_ID)

    warm_x = np.ones(1000, dtype=np.float64)
    warm_y = np.empty_like(warm_x)
    vector_operation(warm_x, warm_y, A, B, 0, 1000)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Physical cores: {cores}")
    print(f"Logical cores: {logical}")
    print(f"Workers: {cores}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Repeats: {REPEATS}")
    print()

    print(
        f"{'N':<13}"
        f"{'Serial (ms)':<16}"
        f"{'Parallel (ms)':<17}"
        f"{'Speedup':<12}"
        f"{'Winner':<12}"
    )
    print("-" * 70)

    results = []

    with ThreadPoolExecutor(max_workers=cores) as pool:
        for n in SIZES:
            x = rng.random(n, dtype=np.float64)
            y_serial = np.empty_like(x)
            y_parallel = np.empty_like(x)

            run_serial(x, y_serial)
            run_parallel(x, y_parallel, pool, cores)

            serial_time = measure(
                lambda: run_serial(x, y_serial)
            )

            parallel_time = measure(
                lambda: run_parallel(x, y_parallel, pool, cores)
            )

            if not np.allclose(y_serial, y_parallel):
                raise RuntimeError("Results do not match")

            speedup = serial_time / parallel_time
            winner = "Parallel" if speedup > 1 else "Serial"

            results.append((n, serial_time, parallel_time))

            print(
                f"{n:<13}"
                f"{serial_time:<16.4f}"
                f"{parallel_time:<17.4f}"
                f"{speedup:<12.2f}"
                f"{winner:<12}"
            )

    print()
    print("CROSSOVER ANALYSIS")

    crossovers = []

    for i in range(1, len(results)):
        prev_n, prev_s, prev_p = results[i - 1]
        curr_n, curr_s, curr_p = results[i]

        prev_faster = prev_p < prev_s
        curr_faster = curr_p < curr_s

        if not prev_faster and curr_faster:
            crossovers.append((prev_n, curr_n))

    if crossovers:
        for low, high in crossovers:
            print(
                f"Parallel becomes faster between "
                f"N = {low:,} and N = {high:,}"
            )
    else:
        print("No slower-to-faster crossover detected.")
        print("The crossover may be outside tested sizes.")

    print()
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
