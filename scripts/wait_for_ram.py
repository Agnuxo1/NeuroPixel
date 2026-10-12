"""Exit when available RAM stays at or above a threshold for consecutive checks (no neural import)."""
import sys
import time

import psutil

threshold, checks_needed, interval, max_hours = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
streak, deadline = 0, time.time() + max_hours * 3600
while time.time() < deadline:
    available = psutil.virtual_memory().available / 2**30
    streak = streak + 1 if available >= threshold else 0
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), f"{available:.2f} GiB", "streak", streak, flush=True)
    if streak >= checks_needed:
        print("RAM_READY", flush=True)
        sys.exit(0)
    time.sleep(interval)
print("RAM_WAIT_TIMEOUT", flush=True)
sys.exit(3)
