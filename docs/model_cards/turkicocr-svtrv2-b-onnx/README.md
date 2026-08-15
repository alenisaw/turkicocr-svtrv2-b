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
  - onnx
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

<h1 align="center">TurkicOCR-SVTRv2-B-ONNX</h1>

This is the **ONNX FP32** export variant of **TurkicOCR-SVTRv2-B**, optimized for portable, cross-platform CPU and GPU deployment. The model is fine-tuned on the independent [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) dataset and exported to ONNX format for execution using the ONNX Runtime engine.

## 🔗 Official Links & Resources

* **GitHub Repository**: [https://github.com/alenisaw/turkicocr-svtrv2-b](https://github.com/alenisaw/turkicocr-svtrv2-b)
* **Kaggle Models Hub**: [https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b](https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b)
* **Hugging Face Dataset**: [https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic)
* **Model Variations**:
  * 🟢 **PyTorch FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b](https://huggingface.co/alenisaw/turkicocr-svtrv2-b)
  * ⚡ **ONNX FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx)
  * ⚡ **ONNX INT8 Quantized**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8)

## Model Details

* **Format**: ONNX (32-bit floating point).
* **Backbone**: SVTRv2-B / OpenOCR.
* **Training Dataset**: [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) (large configuration).
* **Opset Version**: Exported using ONNX Opset 14 (compatible with ONNX Runtime 1.10+).
* **Size**: ~74 MB.
* **Supported Scripts**: Kazakh Cyrillic, Kyrgyz Cyrillic, Russian Cyrillic.

### Tensor Shape Specifications
The ONNX computation graph defines the following input and output structures:
* **Input Tensor (`x`)**: Expects a 4D float32 tensor of shape `[batch_size, 1, 48, 640]`, where:
  * `batch_size` is a dynamic dimension.
  * `channels` is fixed to 1 (grayscale input).
  * `height` is fixed to 48 pixels.
  * `width` is fixed to 640 pixels (matching the fine-tuned model config).
* **Output Tensor (`logits`)**: Yields a 3D float32 tensor of shape `[batch_size, 80, vocabulary_size]`, where:
  * `80` is the spatial sequence length (derived from the backbone's 1/8 horizontal downsampling: $640 / 8 = 80$).
  * `vocabulary_size` is the number of supported characters in the Turkic Cyrillic charset (including blank/CTC tokens).

> [!TIP]
> **Export Parity**: The maximum absolute logit difference between this ONNX export and the original reference model is **$3.9 \times 10^{-6}$**, which mathematically guarantees zero accuracy regression on inference.

## Performance Comparison

Evaluated on a 64-sample diagnostic subset to verify export consistency:

| Variant | CER (%) | WER (%) | chrF | Exact Match (%) | Latency (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Reference Model | 2.12 | 4.28 | 0.9705 | 92.19% | 14.70 (GPU) | 75 MB |
| **ONNX FP32** | **2.17** | **4.28** | **0.9704** | **92.19%** | **130.1 (CPU)** | **74 MB** |

## Usage and Optimization (ONNX Runtime)

To achieve maximum throughput during deployment, ONNX Runtime session options should be configured to enable full graph optimizations and control execution threads:

```python
import cv2
import numpy as np
import onnxruntime as ort

# Configure optimized session options
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
session_options.intra_op_num_threads = 4  # Adjust to match CPU cores

# Initialize ONNX Runtime session
providers = [
    ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kNextPowerOfTwo"}),
    "CPUExecutionProvider"
]
session = ort.InferenceSession("model.onnx", sess_options=session_options, providers=providers)

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
