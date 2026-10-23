
import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

STUDENT_ID = 230103316
SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3

def pipe_worker(conn):
    try:
        while True:
            data = conn.recv()
            if data is None:
                break
            conn.send((len(data), float(data[0]), float(data[-1])))
    finally:
        conn.close()

def shared_worker(conn):
    try:
        while True:
            message = conn.recv()
            if message is None:
                break

            name, shape, dtype = message
            shm = shared_memory.SharedMemory(name=name)

            try:
                data = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
                result = (len(data), float(data[0]), float(data[-1]))
            finally:
                del data
                shm.close()

            conn.send(result)
    finally:
        conn.close()

def benchmark_pipe(data):
    parent, child = mp.Pipe()
    process = mp.Process(target=pipe_worker, args=(child,))
    process.start()
    child.close()

    times = []
    expected = (len(data), float(data[0]), float(data[-1]))

    try:
        for _ in range(REPEATS):
            start = time.perf_counter()
            parent.send(data)
            result = parent.recv()
            elapsed = (time.perf_counter() - start) * 1000

            if result != expected:
                raise RuntimeError("Pipe verification failed")

            times.append(elapsed)
    finally:
        if process.is_alive():
            parent.send(None)
        parent.close()
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join()

    return float(np.median(times))

def benchmark_shared(data):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data

    parent, child = mp.Pipe()
    process = mp.Process(target=shared_worker, args=(child,))
    process.start()
    child.close()

    times = []
    expected = (len(data), float(data[0]), float(data[-1]))
    message = (shm.name, data.shape, data.dtype.str)

    try:
        for _ in range(REPEATS):
            start = time.perf_counter()
            parent.send(message)
            result = parent.recv()
            elapsed = (time.perf_counter() - start) * 1000

            if result != expected:
                raise RuntimeError("Shared memory verification failed")

            times.append(elapsed)
    finally:
        if process.is_alive():
            parent.send(None)
        parent.close()
        process.join(timeout=10)

        if process.is_alive():
            process.terminate()
            process.join()

        del shared_array
        shm.close()
        shm.unlink()

    return float(np.median(times))

def main():
    rng = np.random.default_rng(STUDENT_ID)

    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Logical cores: {os.cpu_count()}")
    print("Data type: float64")
    print(f"Repetitions: {REPEATS}")
    print()

    print(
        f"{'N':<15}"
        f"{'Pipe IPC (ms)':<20}"
        f"{'Shared Memory (ms)':<22}"
        f"{'Ratio':<12}"
    )
    print("-" * 69)

    for n in SIZES:
        data = rng.random(n, dtype=np.float64)

        pipe_time = benchmark_pipe(data)
        shared_time = benchmark_shared(data)

        ratio = pipe_time / shared_time

        print(
            f"{n:<15}"
            f"{pipe_time:<20.3f}"
            f"{shared_time:<22.3f}"
            f"{ratio:<12.2f}x"
        )

    print()
    print("Benchmark completed successfully.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
