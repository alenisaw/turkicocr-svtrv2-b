from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

from .image_io import open_image
from .utils import iter_jsonl, write_json

DEFAULT_OCR_PROMPT = ""


class OCRBackend(Protocol):
    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str: ...


class EchoReferenceBackend:
    """Development-only backend.

    It returns the reference from the manifest when available. Use it only to
    validate the evaluation pipeline, never as a reported baseline.
    """

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        return ""


class PPOCRBackend:
    def __init__(self, device: str | None = None):
        self.device = device
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("Install paddleocr to run the PP-OCR baseline.") from exc
        self._model = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        self._lazy_load()
        output = self._model.ocr(image_url)
        if not output or not isinstance(output, list):
            return ""
        if len(output) > 0 and isinstance(output[0], dict) and "rec_texts" in output[0]:
            return "\n".join(str(t) for t in output[0]["rec_texts"] if t)
        if len(output) > 0 and output[0] and isinstance(output[0], list):
            texts = []
            for line in output[0]:
                if line and len(line) > 1 and isinstance(line[1], tuple):
                    texts.append(str(line[1][0]))
            return "\n".join(texts)
        return _coerce_prediction_text(output)


class TesseractBackend:
    def __init__(self, languages: str = "kaz+rus+kir"):
        self.languages = languages

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        try:
            import os

            import pytesseract
        except ImportError as exc:
            raise RuntimeError("Install pillow and pytesseract to run the Tesseract baseline.") from exc
        from turkicocr.utils import get_asset_root

        asset_dir = get_asset_root()
        binary = asset_dir / "conda-tesseract/bin/tesseract"
        if binary.exists():
            pytesseract.pytesseract.tesseract_cmd = str(binary)
        tessdata = asset_dir / "tessdata"
        if tessdata.exists():
            os.environ.setdefault("TESSDATA_PREFIX", str(tessdata))
        with open_image(image_url) as image:
            return str(pytesseract.image_to_string(image, lang=self.languages)).strip()


class TransformersVisionEncoderDecoderBackend:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self._processor = None
        self._model = None
        self._torch_device = None

    def _load_processor(self):
        from transformers import AutoProcessor

        try:
            return AutoProcessor.from_pretrained(self.checkpoint)
        except Exception as first_error:
            checkpoint_path = Path(self.checkpoint)
            tokenizer_json = checkpoint_path / "tokenizer.json"
            if tokenizer_json.exists():
                try:
                    from transformers import (
                        PreTrainedTokenizerFast,
                        TrOCRProcessor,
                        ViTImageProcessor,
                    )

                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json))
                    image_size = 384
                    config_path = checkpoint_path / "config.json"
                    if config_path.exists():
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        image_size = int(
                            (config.get("encoder") or {}).get("image_size") or image_size
                        )
                    tokenizer_config = checkpoint_path / "tokenizer_config.json"
                    if tokenizer_config.exists():
                        data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
                        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
                            if data.get(key):
                                setattr(tokenizer, key, data[key])
                    return TrOCRProcessor(
                        image_processor=ViTImageProcessor(size={"height": image_size, "width": image_size}),
                        tokenizer=tokenizer,
                    )
                except Exception:
                    pass
            try:
                return AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Could not load processor for {self.checkpoint}: {first_error}; "
                    f"fallback failed: {fallback_error}"
                ) from fallback_error

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "Install torch and transformers to run TrOCR/HTR baselines."
            ) from exc
        wants_gpu = self.device and self.device not in {"cpu", "CPU"}
        cuda_device = "cuda"
        if wants_gpu and ":" in self.device:
            cuda_device = f"cuda:{self.device.split(':')[1]}"
        self._torch_device = cuda_device if wants_gpu and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._torch_device != "cpu" else torch.float32
        self._processor = self._load_processor()
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            local_files_only=Path(self.checkpoint).exists(),
        )
        self._model.to(self._torch_device).eval()

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        self._lazy_load()
        with open_image(image_url) as image:
            pixel_values = self._processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._torch_device)
        generated_ids = self._model.generate(pixel_values, use_cache=True)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def predict_batch(self, image_urls: list[str], prompts: list[str] | None = None) -> list[str]:
        import torch
        self._lazy_load()
        images = []
        for url in image_urls:
            try:
                with open_image(url) as img:
                    images.append(img.copy())
            except Exception:
                from PIL import Image
                images.append(Image.new("RGB", (384, 384), color="white"))
        pixel_values = self._processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._torch_device)
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values, use_cache=True)
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)


class OptionalExternalBackend:
    def __init__(self, backend_name: str, checkpoint: str | None = None):
        self.backend_name = backend_name
        self.checkpoint = checkpoint

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        raise RuntimeError(
            f"Backend '{self.backend_name}' requires its upstream toolkit and integration. "
            "Install requirements-inference.txt and connect the model-specific pipeline before reporting it."
        )


class TransformersVisionLanguageBackend:
    def __init__(self, checkpoint: str, device: str | None = None, max_new_tokens: int = 256):
        self.checkpoint = checkpoint
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._torch_device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install torch and transformers to run VLM document baselines.") from exc
        wants_gpu = self.device and self.device not in {"cpu", "CPU"}
        cuda_device = "cuda"
        if wants_gpu and ":" in self.device:
            cuda_device = f"cuda:{self.device.split(':')[1]}"
        self._torch_device = cuda_device if wants_gpu and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self._torch_device != "cpu" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            local_files_only=Path(self.checkpoint).exists(),
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=Path(self.checkpoint).exists(),
        )
        self._model.to(self._torch_device).eval()

    def predict(self, image_url: str, prompt: str = DEFAULT_OCR_PROMPT) -> str:
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None
        try:
            import torch
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("Install qwen-vl-utils to run Qwen-family VLM baselines.") from exc
        instruction = prompt or "Extract the visible text from this image. Return only the text."
        with open_image(image_url) as image:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        return self._processor.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()


class OpenOCRRecognizerBackend:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.checkpoint = checkpoint
        self.device = device
        self._recognizer = None

    def _lazy_load(self) -> None:
        if self._recognizer is not None:
            return
        import sys
        from pathlib import Path

        import yaml

        from scripts.run_openocr_train import write_resolved_openocr_config
        
        from turkicocr.utils import get_asset_root

        asset_root = str(get_asset_root())
        openocr_root = str(get_asset_root() / "openocr/repo")
        turkicocr_config = "configs/train_svtrv2_b_rec_line.yaml"
        
        if openocr_root not in sys.path:
            sys.path.insert(0, openocr_root)
        tools_root = str((Path(openocr_root) / "tools").resolve())
        if tools_root not in sys.path:
            sys.path.insert(0, tools_root)
        from tools.infer_rec import OpenRecognizer
        
        resolved = write_resolved_openocr_config(turkicocr_config, openocr_root, asset_root)
        cfg = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        
        p = Path(self.checkpoint)
        if p.is_dir():
            for name in ("best.pth", "latest.pth"):
                candidate = p / name
                if candidate.exists():
                    p = candidate
                    break
        cfg["Global"]["checkpoints"] = str(p)
        cfg["Global"]["device"] = "gpu" if self.device and "cpu" not in self.device else "cpu"
        cfg["Global"]["distributed"] = False
        cfg["Global"]["use_amp"] = False
        cfg["Eval"]["loader"]["batch_size_per_card"] = 1
        
        eval_transforms = cfg["Eval"]["dataset"]["transforms"]
        for item in eval_transforms:
            if "RecDynamicResize" in item:
                item["RecDynamicResize"]["padding"] = True
        
        device_id = 0
        if self.device and ":" in self.device:
            try:
                device_id = int(self.device.split(":")[1])
            except ValueError:
                pass
                
        self._recognizer = OpenRecognizer(config=cfg, mode="server", backend="torch", numId=device_id)

    def predict(self, image_url: str, prompt: str = "") -> str:
        self._lazy_load()
        res = self._recognizer(img_path=image_url, batch_num=1)[0]
        return str(res.get("text", ""))


class OpenOCRONNXBackend:
    def __init__(self, onnx_model: str, device: str | None = None):
        self.onnx_model = onnx_model
        self.device = device
        self._recognizer = None

    def _lazy_load(self) -> None:
        if self._recognizer is not None:
            return
        import sys
        from pathlib import Path

        import yaml

        from scripts.run_openocr_train import write_resolved_openocr_config
        
        from turkicocr.utils import get_asset_root

        asset_root = str(get_asset_root())
        openocr_root = str(get_asset_root() / "openocr/repo")
        turkicocr_config = "configs/train_svtrv2_b_rec_line.yaml"
        
        openocr_root = str(Path(openocr_root).resolve())
        tools_root = str((Path(openocr_root) / "tools").resolve())
        for item in (openocr_root, tools_root):
            if item not in sys.path:
                sys.path.insert(0, item)
        from tools.infer_rec import OpenRecognizer
        
        resolved = write_resolved_openocr_config(turkicocr_config, openocr_root, asset_root)
        cfg = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
        use_gpu = (
            os.environ.get("TURKICOCR_ONNX_USE_GPU", "0") == "1"
            and self.device
            and "cpu" not in self.device
        )
        cfg["Global"]["checkpoints"] = None
        cfg["Global"]["pretrained_model"] = None
        cfg["Global"]["device"] = "gpu" if use_gpu else "cpu"
        cfg["Global"]["distributed"] = False
        cfg["Global"]["use_amp"] = False
        cfg["Global"]["backend"] = "onnx"
        cfg["Global"]["onnx_model_path"] = str(Path(self.onnx_model).resolve())
        cfg["Eval"]["loader"]["batch_size_per_card"] = 1
        for item in cfg["Eval"]["dataset"]["transforms"]:
            if "RecDynamicResize" in item:
                item["RecDynamicResize"]["padding"] = True
                
        self._recognizer = OpenRecognizer(
            config=cfg,
            mode="mobile",
            backend="onnx",
            use_gpu="true" if use_gpu else "false",
        )

    def predict(self, image_url: str, prompt: str = "") -> str:
        self._lazy_load()
        import cv2
        image = cv2.imread(image_url)
        if image is None:
            return ""
        result = self._recognizer(img_numpy_list=[image], batch_num=1)[0]
        return str(result.get("text", ""))


def create_backend(
    backend_name: str,
    checkpoint: str | None = None,
    device: str | None = None,
    max_new_tokens: int | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> OCRBackend:
    if backend_name == "ppocr":
        return PPOCRBackend(device=device)
    if backend_name == "tesseract":
        return TesseractBackend(languages=checkpoint or "kaz+rus+kir")
    if backend_name == "transformers_vision_encoder_decoder":
        if not checkpoint:
            raise ValueError("Transformers baseline requires a checkpoint.")
        return TransformersVisionEncoderDecoderBackend(checkpoint=checkpoint, device=device)
    if backend_name in {"vlm_document_ocr", "vlm_document_parser"}:
        if not checkpoint:
            return OptionalExternalBackend(backend_name=backend_name, checkpoint=checkpoint)
        return TransformersVisionLanguageBackend(
            checkpoint=checkpoint,
            device=device,
            max_new_tokens=max_new_tokens or 256,
        )
    if backend_name == "openocr_svtrv2":
        if not checkpoint:
            raise ValueError("openocr_svtrv2 requires a checkpoint.")
        return OpenOCRRecognizerBackend(checkpoint=checkpoint, device=device)
    if backend_name == "openocr_onnx":
        if not checkpoint:
            raise ValueError("openocr_onnx requires an ONNX model path.")
        return OpenOCRONNXBackend(onnx_model=checkpoint, device=device)
    raise ValueError(f"Unknown backend: {backend_name}")



def _coerce_prediction_text(output: Any) -> str:
    chunks: list[str] = []
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return _dict_to_text(output)
    if isinstance(output, list | tuple):
        for item in output:
            chunks.append(_coerce_prediction_text(item))
        return "\n".join(chunk for chunk in chunks if chunk)
    for attr in ("markdown", "text", "content", "res", "json", "data"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if callable(value):
                continue
            chunks.append(_coerce_prediction_text(value))
    if chunks:
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(output)


def _dict_to_text(data: dict[str, Any]) -> str:
    for key in ("markdown", "text", "content", "rec_text", "transcription"):
        value = data.get(key)
        if value:
            return str(value)
    chunks = []
    for value in data.values():
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list | tuple | dict):
            text = _coerce_prediction_text(value)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def extract_prompt_and_reference(sft_row: dict[str, Any]) -> tuple[str, str]:
    if "text" in sft_row:
        return DEFAULT_OCR_PROMPT, str(sft_row["text"])
    prompt = DEFAULT_OCR_PROMPT
    reference = ""
    for item in sft_row.get("text_info", []):
        if item.get("tag") == "mask":
            prompt = str(item.get("text", prompt))
        elif item.get("tag") == "no_mask":
            reference = str(item.get("text", reference))
    return prompt, reference


def extract_image_url(sft_row: dict[str, Any]) -> str:
    if "image_path" in sft_row:
        return str(sft_row["image_path"])
    image_info = sft_row.get("image_info") or []
    if not image_info:
        return ""
    return str(image_info[0].get("image_url", ""))


def expand_manifest_paths(manifest: str | Path | list[str] | list[Path]) -> list[Path]:
    values = manifest if isinstance(manifest, list) else [manifest]
    paths: list[Path] = []
    for value in values:
        text = str(value)
        matches = sorted(glob.glob(text))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(text))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Manifest file(s) not found: {missing}")
    return paths


def iter_manifest_rows(manifest: str | Path | list[str] | list[Path]):
    for path in expand_manifest_paths(manifest):
        yield from iter_jsonl(path)


def run_manifest_inference(
    manifest: str | Path | list[str] | list[Path],
    out_path: str | Path,
    checkpoint: str,
    backend_name: str = "ppocr",
    development_echo: bool = False,
    device: str | None = None,
    max_new_tokens: int | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    max_samples: int | None = None,
) -> None:
    import signal

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Prediction timed out")

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)

    started = time.time()
    peak_cuda_memory_bytes = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            peak_cuda_memory_bytes = 0
    except Exception:
        torch = None  # type: ignore[assignment]
    if development_echo:
        backend: OCRBackend | None = None
    else:
        backend = create_backend(
            backend_name=backend_name,
            checkpoint=checkpoint,
            device=device,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    batch_size = 32
    has_batch = hasattr(backend, "predict_batch") and not development_echo

    with out.open("w", encoding="utf-8") as f:
        if has_batch:
            batch_rows = []
            for idx, row in enumerate(iter_manifest_rows(manifest)):
                if max_samples is not None and count >= max_samples:
                    break
                batch_rows.append((idx, row))
                count += 1
                if len(batch_rows) >= batch_size:
                    _process_and_write_batch(backend, batch_rows, f)
                    batch_rows = []
            if batch_rows:
                _process_and_write_batch(backend, batch_rows, f)
        else:
            for idx, row in enumerate(iter_manifest_rows(manifest)):
                if max_samples is not None and count >= max_samples:
                    break
                row_started = time.time()
                prompt, reference = extract_prompt_and_reference(row)
                image_url = extract_image_url(row)
                if development_echo:
                    prediction = reference
                else:
                    assert backend is not None
                    signal.alarm(30)
                    try:
                        prediction = backend.predict(image_url=image_url, prompt=prompt)
                    except TimeoutException:
                        print(f"Warning: Prediction timed out on image {image_url} (skipped)", flush=True)
                        prediction = "[TIMEOUT]"
                    except Exception as exc:
                        print(f"Warning: Prediction failed on image {image_url} with error: {exc}", flush=True)
                        prediction = "[FAILED]"
                    finally:
                        signal.alarm(0)
                latency_ms = round((time.time() - row_started) * 1000.0, 3)
                payload = {
                    "id": row.get("id") or row.get("sample_id") or idx,
                    "image_url": image_url,
                    "reference": reference,
                    "prediction": prediction,
                    "latency_ms": latency_ms,
                    "metadata": row.get("metadata", {}),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1

    # Restore signal handler
    signal.signal(signal.SIGALRM, old_handler)
    if peak_cuda_memory_bytes is not None:
        try:
            peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated())  # type: ignore[union-attr]
        except Exception:
            peak_cuda_memory_bytes = None
    elapsed_sec = time.time() - started
    summary = {
        "checkpoint": checkpoint,
        "backend": backend_name,
        "device": device,
        "count": count,
        "elapsed_sec": round(elapsed_sec, 3),
        "latency_per_page_ms": round((elapsed_sec / count) * 1000.0, 3) if count else None,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
    }
    write_json(out.with_suffix(".summary.json"), summary)


def _process_and_write_batch(backend, batch_rows, f) -> None:
    image_urls = []
    prompts = []
    references = []
    for _idx, row in batch_rows:
        prompt, reference = extract_prompt_and_reference(row)
        image_url = extract_image_url(row)
        image_urls.append(image_url)
        prompts.append(prompt)
        references.append(reference)

    t0 = time.time()
    try:
        predictions = backend.predict_batch(image_urls=image_urls, prompts=prompts)
    except Exception as exc:
        print(f"Warning: Batch prediction failed: {exc}", flush=True)
        predictions = ["[FAILED]"] * len(batch_rows)
    latency_ms = round(((time.time() - t0) / len(batch_rows)) * 1000.0, 3)

    for (idx, row), pred, ref, url in zip(batch_rows, predictions, references, image_urls, strict=True):
        payload = {
            "id": row.get("id") or row.get("sample_id") or idx,
            "image_url": url,
            "reference": ref,
            "prediction": pred,
            "latency_ms": latency_ms,
            "metadata": row.get("metadata", {}),
        }
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
