
import time
import threading
import queue
import os

STUDENT_ID = 230103316
NUM_ITEMS = 1000
NUM_WORKERS = 4

def workload(i):
    time.sleep(0.0001 * (i % 20))

def static_worker(worker_id, items, completion_times):
    for i in items:
        workload(i)
    completion_times[worker_id] = time.perf_counter()

def dynamic_worker(worker_id, task_queue, completion_times):
    while True:
        try:
            i = task_queue.get_nowait()
        except queue.Empty:
            break

        workload(i)
        task_queue.task_done()

    completion_times[worker_id] = time.perf_counter()

def benchmark_static():
    threads = []
    completion_times = [0.0] * NUM_WORKERS
    chunk_size = NUM_ITEMS // NUM_WORKERS

    start = time.perf_counter()

    for worker_id in range(NUM_WORKERS):
        start_idx = worker_id * chunk_size
        end_idx = (worker_id + 1) * chunk_size

        items = range(start_idx, end_idx)

        thread = threading.Thread(
            target=static_worker,
            args=(worker_id, items, completion_times)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = (time.perf_counter() - start) * 1000
    delta = (max(completion_times) - min(completion_times)) * 1000

    return elapsed, delta

def benchmark_dynamic():
    task_queue = queue.Queue()

    for i in range(NUM_ITEMS):
        task_queue.put(i)

    threads = []
    completion_times = [0.0] * NUM_WORKERS

    start = time.perf_counter()

    for worker_id in range(NUM_WORKERS):
        thread = threading.Thread(
            target=dynamic_worker,
            args=(worker_id, task_queue, completion_times)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed = (time.perf_counter() - start) * 1000
    delta = (max(completion_times) - min(completion_times)) * 1000

    return elapsed, delta

def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {os.uname().sysname}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Total items: {NUM_ITEMS}")
    print("Workload: sleep(0.0001 * (i % 20))")
    print()

    static_time, static_delta = benchmark_static()
    dynamic_time, dynamic_delta = benchmark_dynamic()

    print(f"{'Strategy':<20}{'Wall Time (ms)':<20}{'Completion Delta (ms)':<25}")
    print("-" * 65)

    print(f"{'Static':<20}{static_time:<20.2f}{static_delta:<25.2f}")
    print(f"{'Dynamic':<20}{dynamic_time:<20.2f}{dynamic_delta:<25.2f}")

    print()
    print(f"Speedup (Static/Dynamic): {static_time / dynamic_time:.2f}x")

if __name__ == "__main__":
    main()
