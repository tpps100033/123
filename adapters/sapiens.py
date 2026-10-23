"""Sapiens-1B COCO-17 adapter for fair_reference.

The model download remains in Drive model_cache; it is not part of timed work.
No AMP, torch.compile, quantization, or TensorRT are used.
"""
from pathlib import Path
from time import perf_counter
import json
import re
import sys
import numpy as np
from PIL import Image
from .base import BaseAdapter
from benchmark_core.contracts import Prediction


class Sapiens1BCoco(BaseAdapter):
    model_id = "sapiens_1b_coco"

    def __init__(self, checkpoint, demo_root):
        self.checkpoint = Path(checkpoint)
        self.demo_root = Path(demo_root)
        self._prepared = {}
        self._avg_inference_ms = 0.0
        self.thermal_stop = None
        self.gpu_peak = None

    def load(self, profile):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing Sapiens checkpoint: {self.checkpoint}. Copy the 4,676,154,239-byte "
                "checkpoint from the research Drive cache into this WSL cache path."
            )
        import torch
        if not self.demo_root.is_dir():
            raise FileNotFoundError("Sapiens-Lite demo source was not installed.")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.runtime_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        print(f"✅ Sapiens-1B device: {self.device}")

    def _gpu_snapshot(self):
        if self.device == "cpu":
            return None
        try:
            import pynvml
            if not getattr(self, "_nvml_ready", False):
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_ready = True
            handle = self._nvml_handle
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            used = round(memory.used / 1024**2, 2)
            if self.gpu_peak is None:
                self.gpu_peak = {"device_used_mb": used, "temperature_c_max": temp}
            else:
                self.gpu_peak["device_used_mb"] = max(self.gpu_peak["device_used_mb"], used)
                self.gpu_peak["temperature_c_max"] = max(self.gpu_peak["temperature_c_max"], temp)
            return {"device_used_mb": used, "temperature_c": temp}
        except Exception:
            return None

    def _infer_many(self, image_paths):
        """Run the official demo once for many images so the 1B weight is loaded once."""
        import subprocess, sys, tempfile
        demo_dir = self.demo_root / "lite" / "demo"
        demo = demo_dir / "vis_pose.py"
        if not demo.is_file():
            raise FileNotFoundError(
                f"Sapiens Lite pose demo missing: {demo}. "
                "The 00 source cache must contain lite/demo/vis_pose.py."
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            image_list = tmp_root / "image_list.txt"
            output_root = tmp_root / "outputs"
            output_root.mkdir()
            image_list.write_text(
                "\n".join(str(Path(path).resolve()) for path in image_paths) + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable, str(demo), str(self.checkpoint),
                "--input", str(image_list), "--output-root", str(output_root),
                "--num_keypoints", "17", "--batch-size", "1",
                "--device", self.device,
            ]
            # The official demo uses carriage-return progress output, which can
            # remain invisible when called from a notebook. Track its N/total
            # counter and render a stable parent-side tqdm bar instead.
            from tqdm.auto import tqdm
            import threading

            progress = tqdm(total=len(image_paths), desc="Sapiens-1B inference", unit="img")
            process = subprocess.Popen(
                command,
                cwd=str(demo_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            seen = 0
            output_chunks = []
            buffer = ""
            per_image_wall_ms = []
            last_progress = perf_counter()
            thermal_limit = int(__import__("os").environ.get("INFANTPOSE_MAX_GPU_TEMP_C", "85"))
            thermal_interval = max(1, int(__import__("os").environ.get("INFANTPOSE_TEMP_CHECK_INTERVAL", "1")))
            while True:
                chunk = process.stdout.read(1)
                if chunk == "":
                    break
                output_chunks.append(chunk)
                buffer = (buffer + chunk)[-2000:]
                matches = re.findall(r"(\d+)\s*/\s*(\d+)", buffer)
                if matches:
                    current = min(len(image_paths), int(matches[-1][0]))
                    if current > seen:
                        now = perf_counter()
                        increment = max(1, current - seen)
                        per_image_wall_ms.extend([(now - last_progress) * 1000.0 / increment] * increment)
                        last_progress = now
                        progress.update(current - seen)
                        seen = current
                        if seen % thermal_interval == 0 or seen == len(image_paths):
                            snapshot = self._gpu_snapshot()
                            if snapshot and snapshot["temperature_c"] > thermal_limit:
                                self.thermal_stop = {"temperature_c": snapshot["temperature_c"], "limit_c": thermal_limit, "processed": seen, "phase": "inference"}
                                process.terminate()
                                break
            return_code = process.wait()
            if seen < len(image_paths) and return_code == 0:
                progress.update(len(image_paths) - seen)
            progress.close()
            demo_output = "".join(output_chunks)
            if self.thermal_stop:
                raise RuntimeError(f"Sapiens inference stopped at {self.thermal_stop['temperature_c']} C (limit {thermal_limit} C).")
            if return_code != 0:
                raise RuntimeError(
                    "Sapiens Lite demo failed "
                    f"(exit={return_code}). Command={' '.join(command)}\n"
                    f"demo output:\n{demo_output[-4000:]}"
                )
            payloads = {}
            for image_path in image_paths:
                result_path = output_root / (Path(image_path).stem + ".json")
                if not result_path.is_file():
                    raise RuntimeError(
                        f"Sapiens Lite completed but JSON output is missing: {result_path}. "
                        f"The official demo exited without producing the expected JSON. "
                        f"Output:\n{demo_output[-2000:]}"
                    )
                payloads[str(Path(image_path).resolve())] = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
            if per_image_wall_ms and len(per_image_wall_ms) < len(image_paths):
                fallback = per_image_wall_ms[-1] if per_image_wall_ms else 0.0
                per_image_wall_ms.extend([fallback] * (len(image_paths) - len(per_image_wall_ms)))
            return payloads, {"stdout": demo_output[-2000:], "stderr": "streamed_to_process", "gpu_peak": self.gpu_peak, "per_image_wall_ms": per_image_wall_ms}

    def prepare_dataset(self, rows, profile):
        """Prepare one dataset with one model process instead of one process per image."""
        if not rows:
            self._prepared = {}
            return
        import time
        image_paths = [row["image_path"] for row in rows]
        started = time.perf_counter()
        payloads, streams = self._infer_many(image_paths)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._avg_inference_ms = elapsed_ms / len(rows)
        per_image = streams.get("per_image_wall_ms", [])
        if len(per_image) != len(rows):
            per_image = [self._avg_inference_ms] * len(rows)
        self._timing_by_image = {
            str(Path(row["image_path"]).resolve()): float(per_image[index])
            for index, row in enumerate(rows)
        }
        self._prepared = {
            key: (payload, streams) for key, payload in payloads.items()
        }

    def predict(self, row, profile):
        started = perf_counter()
        t = perf_counter(); image = Image.open(row["image_path"]).convert("RGB"); decode = (perf_counter()-t)*1000
        del image
        key = str(Path(row["image_path"]).resolve())
        if key not in self._prepared:
            raise RuntimeError("Sapiens dataset cache is not prepared; runner must call prepare_dataset().")
        native, streams = self._prepared[key]
        inference = self._timing_by_image.get(key, self._avg_inference_ms)
        t = perf_counter()
        people = native.get(
            "instances",
            native.get("predictions", native.get("instance_info", [])),
        )
        if not people:
            post = (perf_counter()-t)*1000
            timings = {"decode": decode, "preprocess": 0.0, "inference": inference, "postprocess": post, "end_to_end": (perf_counter()-started)*1000}
            return Prediction(row["sample_id"], row["dataset_id"], self.model_id, "", "no_detection", timings_ms=timings, error_code="NO_DETECTION", native={"sapiens": native, **streams})
        best = max(people, key=lambda p: float(p.get("score", p.get("bbox_score", 0.0))))
        points = best.get("keypoints")
        scores = best.get("keypoint_scores", best.get("keypoint_score"))
        if not points or len(points) != 17:
            raise ValueError("Sapiens output does not contain exactly 17 COCO keypoints")
        points = [[float(p[0]), float(p[1])] for p in points]
        scores = [float(x) for x in scores] if scores is not None else [float(best.get("score", 0.0))] * 17
        post = (perf_counter()-t)*1000
        timings = {"decode": decode, "preprocess": 0.0, "inference": inference, "postprocess": post, "end_to_end": (perf_counter()-started)*1000}
        return Prediction(row["sample_id"], row["dataset_id"], self.model_id, "", "ok", points, scores, timings_ms=timings, native={"sapiens": native, **streams})

    def provenance(self):
        source_commit = None
        if (self.demo_root / ".git").is_dir():
            import subprocess
            try:
                source_commit = subprocess.check_output(["git", "-C", str(self.demo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                pass
        return {"model_id": self.model_id, "adapter": type(self).__name__, "weight_source": "Google Drive model_cache checkpoint from the source research project", "checkpoint": str(self.checkpoint), "precision": "fp32", "amp": False, "compile": False, "quantization": False, "tensorrt": False, "runtime_device": getattr(self, "runtime_device", "unknown"), "gpu_peak": self.gpu_peak, "source_commit": source_commit, "latency_distribution_available": True, "latency_source": "official demo per-image progress wall clock"}
