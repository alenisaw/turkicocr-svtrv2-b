---
license: apache-2.0
base_model: alenisaw/turkicocr-svtrv2-b
library_name: onnx
pipeline_tag: image-to-text
language:
  - kk
  - ky
  - ru
tags:
  - ocr
  - text-recognition
  - int8
  - quantization
  - svtrv2
  - turkicocr
  - cyrillic
datasets:
  - alenisaw/turkicocr-cyrillic
metrics:
  - cer
  - wer
  - chrf
---

<p align="center">
  <img src="turkicocr-svtrv2-b-banner.png" alt="TurkicOCR-SVTRv2-B — Lightweight Line-Grounded Recognizer for Kazakh and Kyrgyz Optical Character Recognition" width="100%">
</p>

<h1 align="center">TurkicOCR-SVTRv2-B-INT8</h1>

This is the **quantized INT8** variant of **TurkicOCR-SVTRv2-B**, optimized for low memory footprint, high CPU throughput, and deployment on edge or mobile devices. The underlying model is fine-tuned on the independent [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) dataset.

## 🔗 Official Links & Resources

* **GitHub Repository**: [https://github.com/alenisaw/turkicocr-svtrv2-b](https://github.com/alenisaw/turkicocr-svtrv2-b)
* **Kaggle Models Hub**: [https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b](https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b)
* **Hugging Face Dataset**: [https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic)
* **Model Variations**:
  * 🟢 **PyTorch FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b](https://huggingface.co/alenisaw/turkicocr-svtrv2-b)
  * ⚡ **ONNX FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx)
  * ⚡ **ONNX INT8 Quantized**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8)

## Model Details

* **Format**: ONNX (INT8 quantized).
* **Backbone**: SVTRv2-B / OpenOCR.
* **Training Dataset**: [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) (large configuration).
* **Quantization Method**: Dynamic weight-only quantization targeting `MatMul` (Linear) layers, mapping 32-bit floating point weights to 8-bit signed integers.
* **Size**: **~28 MB** (reduced from 74 MB — over **62.3% size reduction**).
* **Target Environment**: Edge devices, embedded CPU servers, mobile platforms, and low-latency environments.
* **Supported Scripts**: Kazakh Cyrillic, Kyrgyz Cyrillic, Russian Cyrillic.

### Tensor Shape Specifications
* **Input Tensor (`x`)**: Expects a 4D float32 tensor of shape `[batch_size, 1, 48, 640]`.
  * Grayscale input (1 channel).
  * Height fixed to 48 pixels.
  * Width fixed to 640 pixels.
* **Output Tensor (`logits`)**: Yields a 3D float32 tensor of shape `[batch_size, 80, vocabulary_size]`.
  * 80 sequence steps representing horizontal features after 1/8 downsampling.

## Quantization Mechanics and Acceleration

Quantization was executed using the `onnxruntime.quantization` toolchain. In dynamic quantization, the weights are quantized to 8-bit integers offline, while activations are dynamically quantized to 8-bit integers at runtime during inference.

### Hardware Acceleration Benefits
* **Instruction Set Optimization**: Modern CPU architectures can accelerate low-precision integer matrix multiplications in hardware (e.g., **Intel AVX-512 VNNI** or **ARM NEON** instructions). 
* **Latency Reduction**: This hardware acceleration reduces the average CPU latency per crop from 130.1 ms (ONNX FP32) to **117.5 ms** (ONNX INT8) — achieving a **~10.7% speedup** while running on CPU.
* **Accuracy Trade-off**: The quantization process exhibits negligible accuracy regression, adding only **+0.12% CER** compared to the ONNX FP32 variant on the validation benchmark.

## Performance Comparison

Evaluated on a 64-sample diagnostic subset to verify quantization loss:

| Variant | CER (%) | WER (%) | chrF | Exact Match (%) | Latency (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Reference Model | 2.12 | 4.28 | 0.9705 | 92.19% | 14.70 (GPU) | ~75 MB |
| ONNX FP32 | 2.17 | 4.28 | 0.9704 | 92.19% | 130.1 (CPU) | ~74 MB |
| **ONNX INT8** | **2.29** | **4.28** | **0.9701** | **92.19%** | **117.5 (CPU)** | **~28 MB** |

On the independent, page-level external benchmark (`henrygagnier/kazakh-ocr`, 1,000 pages under a shared detected-page pipeline wrapper), the INT8 variant reaches **21.56% CER**, nearly identical to the reference model (21.31% CER).

## Usage (ONNX Runtime in Python)

```python
import cv2
import numpy as np
import onnxruntime as ort

# Configure optimized session options for CPU execution
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4  # Set to match available CPU cores

# Initialize the quantized ONNX session
session = ort.InferenceSession("model.int8.onnx", sess_options=session_options, providers=["CPUExecutionProvider"])

# Preprocessing helper matching training specifications
def preprocess_image(image_path, target_height=48, target_width=640):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    ratio = w / h
    new_w = int(target_height * ratio)
    new_w = min(new_w, target_width)
    
    img_resized = cv2.resize(img, (new_w, target_height))
    padded = np.zeros((target_height, target_width), dtype=np.float32)
    padded[:, :new_w] = img_resized
    
    # Normalize image to [-1.0, 1.0] range
    padded = (padded / 255.0 - 0.5) / 0.5
    padded = np.expand_dims(padded, axis=(0, 1))
    return padded.astype(np.float32)

# Load and preprocess image crop
input_data = preprocess_image("line_crop.png")

# Run inference session
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], {input_name: input_data})
raw_logits = outputs[0]  # Shape: [1, 80, vocab_size]

# Decode sequence (Greedy CTC decoding)
best_path = np.argmax(raw_logits[0], axis=-1)
# (Filter out blank tokens and consecutive duplicates to obtain final text)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@inproceedings{issayev2026turkicocr,
  title={TurkicOCR-SVTRv2-B: Lightweight Line-Grounded Recognizer for Kazakh and Kyrgyz Optical Character Recognition},
  author={Issayev, Alen and Zhalgas, Aidana},
  booktitle={Analysis of Images, Social Networks and Texts (AIST 2026)},
  series={Lecture Notes in Computer Science (LNCS)},
  publisher={Springer},
  year={2026},
  doi={10.1007/978-3-031-XXXXX-X_XX}
}

@misc{issayev2026turkicocrcyrillic,
  title={TurkicOCR Synthetic Cyrillic Dataset},
  author={Issayev, Alen},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic}},
  doi={10.57967/hf/9255}
}
```

## License and Attribution

* **Code**: [Apache-2.0](https://github.com/alenisaw/turkicocr-svtrv2-b/blob/main/LICENSE)
* **Dataset**: `alenisaw/turkicocr-cyrillic` (CC BY 4.0, DOI: `10.57967/hf/9255`)
* **Base Model**: OpenOCR / SVTRv2-B
