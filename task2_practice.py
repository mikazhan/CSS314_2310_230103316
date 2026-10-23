
import math
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103316
NUM_THREADS = 4
TOTAL_SAMPLES = 40_000_000
BATCH_SIZE = 10_000

def count_inside(rng, samples):
    inside = 0

    for _ in range(samples):
        x = rng.random()
        y = rng.random()

        if x * x + y * y <= 1.0:
            inside += 1

    return inside

def shared_worker(samples, rng, lock):
    inside = 0
    remaining = samples

    while remaining > 0:
        batch = min(BATCH_SIZE, remaining)

        with lock:
            inside += count_inside(rng, batch)

        remaining -= batch

    return inside

def independent_worker(thread_index, samples):
    seed = STUDENT_ID * 10 + thread_index
    rng = random.Random(seed)

    return count_inside(rng, samples)

def benchmark_shared():
    rng = random.Random(STUDENT_ID)
    lock = threading.Lock()
    samples = TOTAL_SAMPLES // NUM_THREADS

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(shared_worker, samples, rng, lock)
            for _ in range(NUM_THREADS)
        ]

        inside = sum(f.result() for f in futures)

    elapsed = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * inside / TOTAL_SAMPLES

    return pi_value, elapsed

def benchmark_independent():
    samples = TOTAL_SAMPLES // NUM_THREADS

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(independent_worker, i, samples)
            for i in range(NUM_THREADS)
        ]

        inside = sum(f.result() for f in futures)

    elapsed = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * inside / TOTAL_SAMPLES

    return pi_value, elapsed

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Total samples: {TOTAL_SAMPLES:,}")
    print(f"Samples per thread: {TOTAL_SAMPLES // NUM_THREADS:,}")
    print(f"Actual PI: {math.pi:.10f}")
    print()

    shared_pi, shared_time = benchmark_shared()
    independent_pi, independent_time = benchmark_independent()

    shared_error = abs(shared_pi - math.pi)
    independent_error = abs(independent_pi - math.pi)
    speedup = shared_time / independent_time

    print(f"{'Mode':<20}{'Computed PI':<17}{'Absolute Error':<20}{'Wall Time (ms)':<18}")
    print("-" * 75)

    print(
        f"{'Shared PRNG':<20}"
        f"{shared_pi:<17.10f}"
        f"{shared_error:<20.10f}"
        f"{shared_time:<18.2f}"
    )

    print(
        f"{'Independent PRNG':<20}"
        f"{independent_pi:<17.10f}"
        f"{independent_error:<20.10f}"
        f"{independent_time:<18.2f}"
    )

    print()
    print(f"Speedup (Shared / Independent): {speedup:.2f}x")
    print()
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    main()
