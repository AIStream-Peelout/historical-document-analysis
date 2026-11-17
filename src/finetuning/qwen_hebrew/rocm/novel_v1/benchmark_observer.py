"""Observe the unchanged training workload for the CUDA/ROCm comparison (private tooling)."""

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def json_hash(value: Any) -> str:
    """Hash a JSON-serializable value deterministically.

    :param value: Value to hash.
    :return: SHA256 digest.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class BenchmarkObserver(TrainerCallback):
    """Append timings and existing trainer metrics without changing optimizer or logging state."""

    def __init__(self, torch_module: Any, path: Path, backend: str) -> None:
        """Hold the accelerator API and output file.

        :param torch_module: PyTorch, whose cuda API also serves HIP devices.
        :param path: Append-only metrics file.
        :param backend: Backend label.
        """
        self.torch, self.path, self.backend = torch_module, path, backend
        self.started = 0.0

    def emit(self, event: dict[str, Any]) -> None:
        """Append one timestamped event.

        :param event: Metrics without text or image data.
        """
        with self.path.open("a") as stream:
            stream.write(json.dumps({"unix_time": time.time(), "backend": self.backend, **event}) + "\n")

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Start timing an optimizer step after synchronizing the device.

        :param args: Trainer arguments.
        :param state: Trainer state.
        :param control: Trainer control object.
        :param kwargs: Additional trainer objects.
        :return: Unmodified control object.
        """
        self.torch.cuda.synchronize()
        self.started = time.perf_counter()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        """Record synchronized step time, excluding later evaluation and checkpoint callbacks.

        :param args: Trainer arguments.
        :param state: Trainer state.
        :param control: Trainer control object.
        :param kwargs: Additional trainer objects.
        :return: Unmodified control object.
        """
        self.torch.cuda.synchronize()
        self.emit({"event": "optimizer_step", "step": int(state.global_step),
                   "seconds": time.perf_counter() - self.started,
                   "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": self.torch.cuda.max_memory_reserved()})
        return control

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None,
               **kwargs: Any) -> Any:
        """Copy trainer metrics without emitting a second trainer or W&B log.

        :param args: Trainer arguments.
        :param state: Trainer state.
        :param control: Trainer control object.
        :param logs: Existing trainer metrics.
        :param kwargs: Additional trainer objects.
        :return: Unmodified control object.
        """
        self.emit({"event": "trainer_log", "step": int(state.global_step), "metrics": logs or {}})
        return control


def attach_observer(namespace: dict[str, Any], backend: str, source_sha256: str, source_commit: str) -> None:
    """Persist the resolved runtime and attach the same observer to either backend.

    :param namespace: Notebook globals after the pre-training gates and step-zero evaluation.
    :param backend: cuda or rocm.
    :param source_sha256: Hash of the canonical notebook.
    :param source_commit: Source checkout commit.
    """
    torch = namespace["torch"]
    assert (backend == "rocm") == bool(torch.version.hip), "backend label disagrees with PyTorch"
    trainer, model = namespace["trainer"], namespace["model"]
    out = Path(namespace["OUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    packages = ("torch", "torchvision", "unsloth", "unsloth_zoo", "transformers", "trl", "peft",
                "accelerate", "bitsandbytes", "datasets", "tokenizers")
    manifest = {
        "backend": backend, "source_notebook_sha256": source_sha256, "source_commit": source_commit,
        "dataset_repo": namespace["V22_REPO"], "dataset_revision": namespace["V22_REVISION"],
        "warm_repo": namespace["WARM_CKPT_REPO"], "warm_revision": namespace["WARM_REVISION"],
        "warm_step": namespace["WARM_STEP"], "base_name": model.config._name_or_path,
        "base_revision": getattr(model.config, "_commit_hash", None),
        "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda": torch.version.cuda, "hip": torch.version.hip,
        "packages": {name: version(name) for name in packages},
        "bf16": trainer.args.bf16, "fp16": trainer.args.fp16,
        "step0_eval_loss": namespace["STEP0_EVAL_LOSS"],
        "train_rows": len(namespace["train_ds"]), "val_rows": len(namespace["eval_ds"]),
        "row_order_sha256": json_hash([int(i) for i in namespace["_order"]]),
        "trainable_tensor_spec_sha256": json_hash([(n, list(p.shape)) for n, p in model.named_parameters() if p.requires_grad]),
        "max_steps": namespace["MAX_STEPS"], "max_sequence": namespace["MAX_SEQ"],
        "min_pixels": namespace["MIN_PIX"], "max_pixels": namespace["MAX_PIX"],
        "checkpoint_repo": namespace["CKPT_REPO"],
    }
    for split in ("train", "val"):
        with (Path(namespace["DATA_DIR"]) / "rows" / f"{split}.parquet").open("rb") as stream:
            manifest[f"{split}_parquet_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    trainer.add_callback(BenchmarkObserver(torch, out / "benchmark_metrics.jsonl", backend))
    print("BENCHMARK observer ready:", json.dumps(manifest), flush=True)
