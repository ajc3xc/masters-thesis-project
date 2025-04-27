# compare_imports_subprocess.py

import os, time, threading, subprocess, sys

# (Optional) Preload newer libstdc++ if needed
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    candidate = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(candidate):
        os.environ['LD_PRELOAD'] = candidate + ':' + os.environ.get('LD_PRELOAD', '')

# === MODULES TO IMPORT ===
to_import = [
    'torch',
    'skimage',
    'onnxruntime',
    'PIL',
    'cv2',
]

print("\n--- Import Timing Comparison (with subprocess reset) ---")
print(f"Modules: {to_import}")

# === Turbo Import using Threads ===

def import_module(name):
    print(f"🔹 Importing {name}...")
    if name == 'torch':
        import torch
    elif name == 'skimage':
        import skimage
    elif name == 'onnxruntime':
        import onnxruntime as ort
    elif name == 'PIL':
        from PIL import Image
    elif name == 'cv2':
        import cv2
    else:
        raise ValueError(f"Unknown module: {name}")

def test_turbo_import():
    start = time.time()

    threads = []
    for name in to_import:
        t = threading.Thread(target=import_module, args=(name,))
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    end = time.time()
    return end - start

# === Normal Sequential Import ===

def test_normal_import():
    start = time.time()

    for name in to_import:
        import_module(name)

    end = time.time()
    return end - start

# === MAIN FLOW ===

if os.environ.get('COMPARE_MODE') == 'normal':
    # Child process: run normal imports only
    print("🚶 Running Normal (sequential) Import...")
    normal_time = test_normal_import()
    print(f"✅ Normal import finished in {normal_time:.2f} seconds.")
    sys.exit(0)  # Important: exit cleanly
else:
    # Parent process: run turbo first
    print("\n🚀 Running Turbo (parallel threaded) Import...")
    turbo_time = test_turbo_import()
    print(f"✅ Turbo import finished in {turbo_time:.2f} seconds.")

    # Now run the same script as a subprocess for normal import
    print("\n♻️ Restarting fresh Python for Normal Import...\n")
    result = subprocess.run(
        [sys.executable, __file__],
        env={**os.environ, 'COMPARE_MODE': 'normal'},
        capture_output=True, text=True
    )

    # Parse normal time output
    normal_time = None
    for line in result.stdout.splitlines():
        if "Normal import finished" in line:
            normal_time = float(line.split()[-2])

    # === Summary ===
    print("\n--- Import Timing Summary ---")
    print(f"Turbo threaded import:    {turbo_time:.2f} seconds")
    print(f"Normal sequential import: {normal_time:.2f} seconds")
