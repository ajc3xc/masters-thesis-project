import os
import openvino as ov

# Set environment variables programmatically
os.environ["OPENCL_VENDOR_PATH"] = "/etc/OpenCL/vendors"
os.environ["OV_GPU_PLUGIN_LOG_LEVEL"] = "DEBUG"
os.environ["InferenceEngine_DIR"] = "/usr/local/lib/python3.10/dist-packages/openvino"  # adjust if needed

# Proceed with OpenVINO initialization
core = ov.Core()
print("Available devices:", core.available_devices)
