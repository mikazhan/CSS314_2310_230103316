import os
import platform
import subprocess
import time
import math
import numpy as np

student_id = 230103316

cpu_model = platform.processor()

if platform.system() == "Darwin":
    cpu_model = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True
    ).stdout.strip() or platform.machine()

print(f"Student ID: {student_id}")
print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {cpu_model}")
print(f"Logical cores: {os.cpu_count()}")

n = 20_000_000
rng = np.random.default_rng(student_id)
data = rng.random(n, dtype=np.float64)

start = time.perf_counter()

total = 0.0
for value in data:
    total += float(value)

mean = total / n

squared_sum = 0.0
for value in data:
    squared_sum += (float(value) - mean) ** 2

std = math.sqrt(squared_sum / n)

end = time.perf_counter()

elapsed_ms = (end - start) * 1000

print(f"Array size: {n}")
print(f"Data type: {data.dtype}")
print(f"Sum: {total:.6f}")
print(f"Standard deviation: {std:.6f}")
print(f"Serial execution time: {elapsed_ms:.2f} ms")
