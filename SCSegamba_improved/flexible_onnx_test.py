import onnxruntime as ort
import numpy as np

# 1) point to your ONNX file
onnx_path = "onnx_exports/TUT_eca_dynamic.onnx"

# 2) create an inference session
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# 3) get expected input shape (to check expected channel count)
input_info = sess.get_inputs()[0]
input_shape = input_info.shape  # e.g. [None, 3, None, None]
expected_channels = input_shape[1]

# 4) define a few different (H, W) to test
test_sizes = [
    (256, 256),
    (384, 384),
    (512, 512),
    (1024, 1024),
    (2048, 2048)
]

for H, W in test_sizes:
    # random grayscale input [1, 1, H, W]
    x = np.random.randn(1, 1, H, W).astype(np.float32)
    
    # If model expects 3-channel input, repeat the single channel
    if expected_channels == 3 and x.shape[1] == 1:
        x = np.repeat(x, 3, axis=1)

    outputs = sess.run(None, {"input": x})
    y = outputs[0]
    print(f"Input shape: {x.shape} → Output shape: {y.shape}")
