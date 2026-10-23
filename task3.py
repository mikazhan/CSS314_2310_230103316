
import multiprocessing as mp
import ctypes
import os
import subprocess
import time

STUDENT_ID = 230103316
ITERATIONS = 50_000_000

def get_physical_cores():
    result = subprocess.run(
        ["sysctl", "-n", "hw.physicalcpu"],
        capture_output=True,
        text=True
    )
    return int(result.stdout.strip())

def worker_shared(counts, index):
    for _ in range(ITERATIONS):
        counts[index] += 1

def worker_local(counts, index):
    local = 0
    for _ in range(ITERATIONS):
        local += 1
    counts[index] = local

def benchmark(cores, mode):
    counts = mp.RawArray(ctypes.c_longlong, cores)
    processes = []

    start = time.perf_counter()

    for i in range(cores):
        target = worker_shared if mode == "shared" else worker_local
        p = mp.Process(target=target, args=(counts, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    elapsed = (time.perf_counter() - start) * 1000

    if any(value != ITERATIONS for value in counts):
        raise RuntimeError("Incorrect results")

    return elapsed

if __name__ == "__main__":
    cores = get_physical_cores()

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {os.uname().sysname}")
    print(f"CPU: Apple M2")
    print(f"Physical cores: {cores}")
    print(f"Iterations per worker: {ITERATIONS:,}")
    print()

    unpadded = benchmark(cores, "shared")
    local = benchmark(cores, "local")

    print(f"Unpadded Shared: {unpadded:.2f} ms")
    print(f"Padded/Local:   {local:.2f} ms")
    print(f"Execution Ratio: {unpadded / local:.2f}x")
