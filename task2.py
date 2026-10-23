
import os
import platform
import time
import multiprocessing as mp
import numpy as np

student_id = 230103316
n = 20_000_000
P = 0.95
shared_data = None

def init_worker(array):
    global shared_data
    shared_data = np.frombuffer(array, dtype=np.float64)

def partial_sum(bounds):
    start, end = bounds
    total = 0.0
    for i in range(start, end):
        total += float(shared_data[i])
    return total

def benchmark(array, workers):
    chunk_size = n // workers
    chunks = []

    for i in range(workers):
        start = i * chunk_size
        end = n if i == workers - 1 else (i + 1) * chunk_size
        chunks.append((start, end))

    with mp.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(array,)
    ) as pool:
        start_time = time.perf_counter()
        results = pool.map(partial_sum, chunks)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

    return sum(results), elapsed_ms

def main():
    cores = os.cpu_count() or 1
    workers_list = list(dict.fromkeys([1, 2, 4, cores, cores * 2]))

    rng = np.random.default_rng(student_id)
    data = mp.RawArray("d", n)
    np.frombuffer(data, dtype=np.float64)[:] = rng.random(n)

    print(f"Student ID: {student_id}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor() or platform.machine()}")
    print(f"Logical cores: {cores}")
    print(f"Array size: {n}")
    print(f"Data type: float64")
    print(f"Parallelizable fraction: {P}")
    print()

    print(f"{'Cores':<10}{'Wall Time (ms)':<20}{'Speedup':<15}{'Amdahl Upper Bound':<20}")
    print("-" * 65)

    baseline = None

    for workers in workers_list:
        total, elapsed = benchmark(data, workers)

        if baseline is None:
            baseline = elapsed

        speedup = baseline / elapsed
        amdahl = 1 / ((1 - P) + P / workers)

        print(f"{workers:<10}{elapsed:<20.2f}{speedup:<15.2f}{amdahl:<20.2f}")

    print()
    print(f"Array sum: {total:.6f}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
