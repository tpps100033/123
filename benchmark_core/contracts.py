from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import math

STATUSES = {'ok','no_detection','input_error','model_error','mapping_error'}

@dataclass
class Prediction:
    sample_id: str; dataset_id: str; model_id: str; run_id: str; status: str
    keypoints_xy: list | None = None; keypoint_scores: list | None = None
    bbox_xywh: list | None = None; timings_ms: dict | None = None
    error_code: str | None = None; native: dict | None = None
    def validate(self):
        if self.status not in STATUSES: raise ValueError('invalid status')
        if self.status == 'ok':
            if len(self.keypoints_xy or []) != 17 or len(self.keypoint_scores or []) != 17: raise ValueError('ok requires COCO-17')
            if not all(len(p)==2 and all(math.isfinite(float(v)) for v in p) for p in self.keypoints_xy): raise ValueError('invalid points')
            if not all(math.isfinite(float(v)) for v in self.keypoint_scores): raise ValueError('invalid confidences')
        required={'decode','preprocess','inference','postprocess','end_to_end'}
        if not required.issubset((self.timings_ms or {})): raise ValueError('missing segmented timing')
        return self
    def record(self): self.validate(); return asdict(self)

@dataclass
class RunManifest:
    benchmark_id: str; schema_version: str; run_id: str; dataset_id: str; model_id: str
    profile: str; dataset_manifest_sha256: str; model_provenance: dict; environment: dict
    protocol: dict; counts: dict; timing: dict
    def record(self): return asdict(self)
