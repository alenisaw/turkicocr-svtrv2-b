---
title: TurkicOCR-SVTRv2-B Collection
license: apache-2.0
tags:
  - ocr
  - text-recognition
  - openocr
  - svtrv2
  - turkicocr
  - kazakh
  - kyrgyz
  - cyrillic
datasets:
  - alenisaw/turkicocr-cyrillic
---

<p align="center">
  <img src="turkicocr-svtrv2-b-banner.png" alt="TurkicOCR-SVTRv2-B — Lightweight Line-Grounded Recognizer for Kazakh and Kyrgyz Optical Character Recognition" width="100%">
</p>

<h1 align="center">TurkicOCR-SVTRv2-B Model Collection</h1>

This collection houses a family of lightweight, high-performance Cyrillic text recognition models optimized specifically for Kazakh, Kyrgyz, and bilingual (Russian-mixed) documents. All models in this collection are fine-tuned from the state-of-the-art SVTRv2-B / OpenOCR architecture on the [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) dataset.

## 🔗 Official Links & Resources

* **GitHub Repository**: [https://github.com/alenisaw/turkicocr-svtrv2-b](https://github.com/alenisaw/turkicocr-svtrv2-b)
* **Kaggle Models Hub**: [https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b](https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b)
* **Hugging Face Dataset**: [https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic)
* **Model Variations**:
  * 🟢 **PyTorch FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b](https://huggingface.co/alenisaw/turkicocr-svtrv2-b)
  * ⚡ **ONNX FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx)
  * ⚡ **ONNX INT8 Quantized**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8)

## Architectural Overview

The models in this collection utilize the **SVTRv2-B** (Single Visual Model Text Recognizer v2, Base size) architecture. It has approximately **35 million parameters** and uses a sequence-to-sequence model designed specifically for scene text and document region recognition. 

### Key Architectural Design Elements
* **Single Visual Backbone**: Instead of using heavy convolutional backbones combined with recurrent layers (like CRNN), SVTRv2 uses a visual-only attention-based network that merges local feature extraction with global contextual reasoning. This design ensures that fine character details (such as тюркские диакритические знаки) are preserved.
* **CTC Supervision**: The model is trained with Connectionist Temporal Classification (CTC) sequence supervision. This separates sequence prediction from explicit character boundaries, meaning the model reads characters based on visual cues without needing bounding box annotations for individual glyphs.
* **Input-Output Mapping**: It accepts a single bounding box crop representing a text line, short field, or table cell, and yields a variable-length Unicode string representing the exact visible characters.

## Model Variants

This collection contains a fine-tuned model and its ONNX Runtime exports. All variants share the same vocabulary and training parameters, optimized for different execution runtimes:

| Model ID | Framework | Format | Size (MB) | Target Environment / Use Case | CPU Latency | GPU Latency |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| [**turkicocr-svtrv2-b**](https://huggingface.co/alenisaw/turkicocr-svtrv2-b) | Native | `.pth` | ~75 MB | Base training, fine-tuning, GPU server environments | ~230 ms | **14.7 ms** |
| [**turkicocr-svtrv2-b-onnx**](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx) | ONNX Runtime | `.onnx` | ~74 MB | Cross-platform deployment (C++, Go, Rust, Python, etc.) | ~130 ms | **14.5 ms** |
| [**turkicocr-svtrv2-b-int8**](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8) | ONNX Runtime | `.onnx` (INT8) | **~28 MB** | Edge devices, mobile CPU, low-memory embedded applications | **~117 ms** | N/A |

## Scientific Context and Benchmarks

### The Challenge of low-resource Cyrillic OCR
Standard Cyrillic OCR baselines degrade when encountering Turkic Cyrillic alphabets, which contain specific letters like `Ә/ә, Ғ/ғ, Қ/қ, Ң/ң, Ө/ө, Ұ/ұ, Ү/ү, Һ/һ, I/i`. These characters are visually close to base Russian Cyrillic characters and are frequently misrecognized by systems that are only trained on base Cyrillic data.

TurkicOCR-SVTRv2-B overcomes this by applying Connectionist Temporal Classification (CTC) sequence supervision on the [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) (large configuration) dataset, a large-scale, privacy-preserving synthetic corpus containing 90,000 train pages, 5,000 validation pages, and 5,000 test pages across 29 layout structures (worksheets, administrative certificates, tables, forms) and 7 controlled degradation profiles.

### Page-Level Pipeline Performance
The models were evaluated inside a full layout-aware page processing pipeline (Detection $\rightarrow$ Recognition $\rightarrow$ Reading Order Reconstruction) on the independent `henrygagnier/kazakh-ocr` benchmark (1,000 pages):

![External Comparison](fig3_external_benchmark.png)

### Performance Analysis
As shown in the external benchmark bar chart, both the base reference (21.31% CER, 65.50% WER) and the quantized ONNX INT8 variant (21.56% CER, 67.29% WER) of TurkicOCR-SVTRv2-B outperform all baseline systems. It reduces the Character Error Rate (CER) by more than 10% absolute compared to Kazakh TrOCR (31.96% CER) and by over 22% absolute compared to Tesseract OCR (43.52% CER). 

Standard multilingual and Cyrillic recognizers like Russian TrOCR, PP-OCR, and Cyrillic HTR fail to recognize language-specific letters and mixed-language layouts under the layout-aware reading order pipeline, resulting in high CER values exceeding 60%.

## Citation

If you use this model collection, please cite:

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
