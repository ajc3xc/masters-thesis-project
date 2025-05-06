#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np

def test_flexible(onnx_path, sizes):
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider','CPUExecutionProvider']
    )
    for (H,W) in sizes:
        inp = np.random.randn(1,3,H,W).astype(np.float32)
        out = sess.run(None, {'input': inp})
        print(f"Input {H}×{W} → Output shape {out[0].shape}")

if __name__ == "__main__":

    # test a handful of sizes
    test_sizes = [
      (128,128),
      (256,512),
      (512,256),
      (300,450),
      (1024,1024),
    ]
    test_flexible("dscf_dynamic.onnx", test_sizes)
