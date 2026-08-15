---
license: apache-2.0
base_model: OpenOCR/SVTRv2-B
library_name: openocr
pipeline_tag: image-to-text
language:
  - kk
  - ky
  - ru
tags:
  - ocr
  - text-recognition
  - openocr
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

<h1 align="center">TurkicOCR-SVTRv2-B (Base Model)</h1>

**TurkicOCR-SVTRv2-B** is a fine-tuned text recognition model based on the **SVTRv2-B / OpenOCR** architecture (~35M parameters). It is fine-tuned on the independent [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) dataset for lightweight, high-fidelity line-level Cyrillic text recognition.

## 🔗 Official Links & Resources

* **GitHub Repository**: [https://github.com/alenisaw/turkicocr-svtrv2-b](https://github.com/alenisaw/turkicocr-svtrv2-b)
* **Kaggle Models Hub**: [https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b](https://www.kaggle.com/models/alenissayev/turkicocr-svtrv2-b)
* **Hugging Face Dataset**: [https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic)
* **Model Variations**:
  * 🟢 **PyTorch FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b](https://huggingface.co/alenisaw/turkicocr-svtrv2-b)
  * ⚡ **ONNX FP32**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-onnx)
  * ⚡ **ONNX INT8 Quantized**: [https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8](https://huggingface.co/alenisaw/turkicocr-svtrv2-b-int8)

## Model Details

* **Architecture**: SVTRv2-B (Single Visual Model Text Recognizer v2, Base configuration).
* **Supervision**: Connectionist Temporal Classification (CTC) sequence-to-sequence loss (no character bounding boxes needed).
* **Parameters**: ~35M.
* **Training Dataset**: [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) (large configuration, CC-BY-4.0).
* **Input**: Bounded document image crops (lines, cells, or fields) resized to $48 \times 640$ px.
* **Output**: UTF-8 encoded Cyrillic text string.
* **Supported Scripts**: Kazakh Cyrillic, Kyrgyz Cyrillic, Russian Cyrillic (including mixed/bilingual text).

## Fine-Tuning and Optimization Parameters

The model was fine-tuned using a distributed training setup with the following hyperparameter configurations:

* **Hardware and Distributed Setup**: Training was distributed across 4 GPU nodes using PyTorch DDP.
* **Batch Size**: 32 samples per GPU (effective global batch size of 128).
* **Precision**: Mixed precision (FP16) training using Automatic Mixed Precision (AMP) to optimize memory and training speed.
* **Optimizer and Learning Rate**: Adam optimizer with a base learning rate of `0.0003`, a weight decay of `0.00005`, and 1 epoch of warmup.
* **Regularization**: Gradient clipping norm of `5.0` was applied to prevent exploding gradients.
* **Data Balancing**: The training batch generation balanced the inputs by language script composition (Kazakh, Kyrgyz, Russian, mixed), degradation profiles, and rare character frequencies to address low-resource data distribution issues.
* **Sequence Constraints**: Maximum input text length set to 256 characters.

## Training Profile and Evaluation

The optimal model weights were selected after analyzing convergence and training stability across the final training phase.

![Training Profile](fig1_checkpoint_profile.png)

*Panel (a) shows aggregate CER, WER, and rare-character CER on validation samples; the red star denotes the selected optimal model. Panel (b) shows the CER trajectory separated by language groups.*

### Quantitative Metrics
Evaluated on the full held-out test split of the [alenisaw/turkicocr-cyrillic](https://huggingface.co/datasets/alenisaw/turkicocr-cyrillic) dataset (293,814 crop regions):

| Metric | Point Estimate | 95% Confidence Interval (Bootstrap) |
| :--- | :---: | :---: |
| **Character Error Rate (CER)** | **4.51%** | [4.45%, 4.56%] |
| **Word Error Rate (WER)** | **9.14%** | [9.04%, 9.24%] |
| **chrF Score** | **0.9377** | [0.9370, 0.9384] |
| **Success Rate (Exact Match)** | **86.01%** | [85.88%, 86.14%] |
| **Mean Inference Latency** | **14.70 ms** | per crop on NVIDIA L4 (Torch FP32) |

## Character-Specific Error Analysis

A scientific audit over the character recognition mistakes shows that recognition issues are heavily concentrated in low-resource language-specific letters:

| Character | Language | References | Errors | Error Rate | 95% Wilson CI | Deletion share (%) |
| :---: | :---: | ---: | ---: | ---: | :---: | :---: |
| **Ә / ә** | Kazakh | 54,067 | 4,294 | **7.94%** | [7.72%, 8.17%] | 74.36% |
| **Ғ / ғ** | Kazakh | 32,178 | 2,299 | **7.14%** | [6.87%, 7.43%] | 67.07% |
| **Қ / қ** | Kazakh | 190,840 | 13,473 | **7.06%** | [6.95%, 7.18%] | 76.12% |
| **Ң / ң** | Kazakh/Kyrgyz | 43,935 | 3,874 | **8.82%** | [8.56%, 9.09%] | 69.05% |
| **Ө / ө** | Kazakh/Kyrgyz | 139,759 | 11,759 | **8.41%** | [8.27%, 8.56%] | 69.92% |
| **Ұ / ұ** | Kazakh | 52,087 | 3,788 | **7.27%** | [7.05%, 7.50%] | 71.83% |
| **Ү / ү** | Kazakh/Kyrgyz | 139,327 | 12,136 | **8.71%** | [8.56%, 8.86%] | 75.68% |
| **Һ / һ** | Kazakh | 469 | 103 | **21.96%** | [18.45%, 25.93%] | 78.64% |
| **I / i** | Kazakh | 356,776 | 23,710 | **6.65%** | [6.56%, 6.73%] | 83.93% |

> [!NOTE]
> **Key Scientific Insight**: Residual errors are heavily dominated by **deletion-like behavior** (accounting for **67% to 84%** of errors across all audited characters). This indicates that the recognizer tends to omit the character completely or drop its language-specific mark/diacritic, whereas visual substitutions (e.g., swapping `Қ` for `К`) are secondary.

## Usage

To run inference using the native backend via the `openocr` toolchain:

```python
import sys
import yaml
from pathlib import Path

# Adjust paths to your local OpenOCR installation
openocr_root = "/path/to/openocr/repo"
sys.path.insert(0, openocr_root)
sys.path.insert(0, str(Path(openocr_root) / "tools"))

from tools.infer_rec import OpenRecognizer

# Load model configuration
cfg = yaml.safe_load(Path("configs/train_svtrv2_b_rec.yaml").read_text(encoding="utf-8"))
cfg["Global"]["checkpoints"] = "./best.pth"  # Set to downloaded checkpoint path

# Initialize OpenRecognizer
recognizer = OpenRecognizer(config=cfg, mode="server", backend="torch", numId=0)

# Run inference (input image shape expects height=48, width=640)
result = recognizer(img_path="line_crop.png", batch_num=1)[0]
print("Text:", result.get("text"))
print("Score:", result.get("score"))
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
