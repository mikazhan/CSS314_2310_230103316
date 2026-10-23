import threading
import time
import os

STUDENT_ID = 230103316
THREADS = 4
ITERATIONS = 2_000_000
EXPECTED = THREADS * ITERATIONS

counter = 0
lock = threading.Lock()

def unsafe_worker():
    global counter

    for i in range(ITERATIONS):
        value = counter

        if i % 1000 == 0:
            time.sleep(0)

        counter = value + 1

def safe_worker():
    global counter

    for i in range(ITERATIONS):
        with lock:
            value = counter

            if i % 1000 == 0:
                time.sleep(0)

            counter = value + 1

def benchmark(worker):
    global counter
    counter = 0

    threads = [
        threading.Thread(target=worker)
        for _ in range(THREADS)
    ]

    start = time.perf_counter()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = (time.perf_counter() - start) * 1000

    return counter, elapsed

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {os.uname().sysname}")
    print("CPU: Apple M2")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Threads: {THREADS}")
    print(f"Increments per thread: {ITERATIONS:,}")
    print()

    unsafe_count, unsafe_time = benchmark(unsafe_worker)

    lost = EXPECTED - unsafe_count
    lost_percent = (lost / EXPECTED) * 100

    safe_count, safe_time = benchmark(safe_worker)

    print("UNSAFE VERSION (WITHOUT LOCK)")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {unsafe_count:,}")
    print(f"Lost updates:         {lost:,}")
    print(f"Lost percentage:      {lost_percent:.2f}%")
    print(f"Wall time:            {unsafe_time:.2f} ms")
    print()

    print("SAFE VERSION (WITH LOCK)")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {safe_count:,}")
    print(f"Lost updates:         {EXPECTED - safe_count:,}")
    print(f"Wall time:            {safe_time:.2f} ms")
    print()

    print("SYNCHRONIZATION OVERHEAD")
    print(f"Time penalty: {safe_time - unsafe_time:.2f} ms")
    print(f"Time ratio:   {safe_time / unsafe_time:.2f}x")
    print(f"Correct result: {safe_count == EXPECTED}")

if __name__ == "__main__":
    main()
