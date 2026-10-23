"""Official OpenPose 1.7.0 COCO-18 inference through its CUDA Python API.

Do not import cv2 in this process. The OpenPose build links Ubuntu OpenCV while
the benchmark venv's wheel ships another OpenCV build; loading both has caused
native crashes. Pillow decodes input, and OpenPose handles its own preprocessing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .base import BaseAdapter
from benchmark_core.contracts import Prediction


# OpenPose COCO-18 order -> benchmark COCO-17, omitting the Neck joint.
COCO17_FROM_OPENPOSE = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
DEFAULT_NET_RESOLUTION = "480x368"


class OpenPoseCoco18(BaseAdapter):
    model_id = "openpose_coco18"

    def __init__(self, prototxt: str | Path, weights: str | Path):
        self.prototxt = Path(prototxt)
        self.weights = Path(weights)
        self.wrapper = None

    def load(self, profile):
        """Start the official Caffe/CUDA wrapper once; predictions reuse it."""
        module_dir = Path(os.environ.get(
            "INFANTPOSE_OPENPOSE_PYTHON_DIR",
            Path.home() / ".local/share/infantpose/build/openpose-cuda-final/python/openpose",
        )).expanduser()
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        import pyopenpose as op

        self.op = op
        self.module_path = str(Path(op.__file__).resolve())
        self.net_resolution = os.environ.get("INFANTPOSE_OPENPOSE_NET_RESOLUTION", DEFAULT_NET_RESOLUTION)
        model_folder = Path(os.environ.get(
            "INFANTPOSE_OPENPOSE_MODEL_FOLDER",
            Path.home() / ".local/share/infantpose/src/openpose/models",
        )).expanduser()
        if not self.prototxt.is_file() or not self.weights.is_file():
            raise FileNotFoundError(f"OpenPose project cache is incomplete: {self.prototxt}, {self.weights}")
        if not (model_folder / "pose/coco/pose_iter_440000.caffemodel").is_file():
            raise FileNotFoundError(f"OpenPose CUDA model folder has no COCO weights: {model_folder}")
        self.wrapper = op.WrapperPython()
        self.wrapper.configure({
            "model_folder": str(model_folder),
            "model_pose": "COCO",
            "net_resolution": self.net_resolution,
            "keypoint_scale": "0",
            "num_gpu": "1",
            "num_gpu_start": "0",
            "scale_number": "1",
            "number_people_max": "-1",
            "display": "0",
            "render_pose": "0",
            "logging_level": "3",
        })
        self.wrapper.start()
        self.runtime_device = "cuda:0"
        self.openpose_version = "1.7.0"
        self.gpu_peak = {"memory_used_mb": 0, "temperature_c": 0, "utilization_pct": 0}
        self._record_gpu_metrics()

    def _record_gpu_metrics(self):
        """Keep OpenPose VRAM/temperature peaks, since Caffe is outside PyTorch."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            values = {
                "memory_used_mb": round(memory.used / 1024**2, 2),
                "temperature_c": int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
                "utilization_pct": int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
            }
            for key, value in values.items():
                self.gpu_peak[key] = max(self.gpu_peak.get(key, 0), value)
        except Exception as exc:
            self.gpu_metrics_error = type(exc).__name__

    def predict(self, row, profile):
        started = perf_counter()
        t = perf_counter()
        rgb = np.asarray(Image.open(row["image_path"]).convert("RGB"))
        height, width = rgb.shape[:2]
        decode_ms = (perf_counter() - t) * 1000

        # OpenPose's cv::Mat input convention is BGR. Pass the NumPy array
        # directly; importing the venv's cv2 wheel beside system OpenCV crashes.
        t = perf_counter()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        datum = self.op.Datum()
        datum.cvInputData = bgr
        preprocess_ms = (perf_counter() - t) * 1000

        t = perf_counter()
        self.wrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        inference_ms = (perf_counter() - t) * 1000
        self._record_gpu_metrics()

        t = perf_counter()
        raw = datum.poseKeypoints
        if raw is None or raw.size == 0 or raw.shape[0] == 0:
            postprocess_ms = (perf_counter() - t) * 1000
            timings = {
                "decode": decode_ms, "preprocess": preprocess_ms,
                "inference": inference_ms, "postprocess": postprocess_ms,
                "end_to_end": (perf_counter() - started) * 1000,
            }
            return Prediction(
                row["sample_id"], row["dataset_id"], self.model_id, "", "no_detection",
                timings_ms=timings, error_code="NO_DETECTION",
                native={"runtime_device": self.runtime_device, "network_resolution": self.net_resolution},
            )

        if raw.ndim != 3 or raw.shape[1:] != (18, 3):
            raise RuntimeError(f"Expected OpenPose COCO-18 (people,18,3), got {raw.shape}")
        # The COCO handoff keeps one person per image. If the detector returns
        # several, retain the person with the most confident visible joints.
        person = max(raw, key=lambda p: (int(np.count_nonzero(p[:, 2] > 0)), float(p[:, 2].sum())))
        points18 = person[:, :2].astype(np.float64, copy=True)
        scores18 = person[:, 2].astype(np.float64, copy=True)
        points17 = points18[list(COCO17_FROM_OPENPOSE)].tolist()
        scores17 = scores18[list(COCO17_FROM_OPENPOSE)].tolist()
        valid = [point for point, score in zip(points17, scores17) if score > 0]
        status = "ok" if valid and max(scores17, default=0.0) >= 0.05 else "no_detection"
        bbox = None
        if valid:
            xs = [point[0] for point in valid]
            ys = [point[1] for point in valid]
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        postprocess_ms = (perf_counter() - t) * 1000
        timings = {
            "decode": decode_ms, "preprocess": preprocess_ms,
            "inference": inference_ms, "postprocess": postprocess_ms,
            "end_to_end": (perf_counter() - started) * 1000,
        }
        if status == "no_detection":
            points17 = scores17 = None
        return Prediction(
            row["sample_id"], row["dataset_id"], self.model_id, "", status,
            points17, scores17, bbox_xywh=bbox, timings_ms=timings,
            error_code=None if status == "ok" else "NO_DETECTION",
            native={
                "keypoints_coco18": points18.tolist(),
                "keypoint_scores_coco18": scores18.tolist(),
                "runtime_device": self.runtime_device,
                "network_resolution": self.net_resolution,
            },
        )

    def close(self):
        if self.wrapper is not None:
            self._record_gpu_metrics()
            self.wrapper.stop()
            self.wrapper = None

    def provenance(self):
        return {
            "model_id": self.model_id,
            "adapter": type(self).__name__,
            "weight_source": "OpenPose 1.7.0 COCO-18 Caffe model (locally cached)",
            "runtime_device": getattr(self, "runtime_device", "unknown"),
            "inference_backend": "OpenPose Caffe CUDA",
            "openpose_version": getattr(self, "openpose_version", "unknown"),
            "openpose_module": getattr(self, "module_path", None),
            "network_resolution": getattr(self, "net_resolution", DEFAULT_NET_RESOLUTION),
            "cuda_architecture": "sm_120",
            "amp": False,
            "compile": False,
            "quantization": False,
            "tensorrt": False,
            "batch_execution": "one image per forward; persistent CUDA model context",
            "cv2_imported": False,
            "gpu_peak": getattr(self, "gpu_peak", None),
            "gpu_metrics_error": getattr(self, "gpu_metrics_error", None),
        }
