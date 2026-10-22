"""Report GPU visibility without claiming inference support from an import alone."""
import json, platform, shutil, subprocess, sys

report = {"platform": platform.platform(), "python": sys.version.split()[0], "nvidia_smi": shutil.which("nvidia-smi")}
if report["nvidia_smi"]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True)
    report["nvidia_smi_output"] = result.stdout.strip()
    report["nvidia_smi_error"] = result.stderr.strip() or None
try:
    import torch
    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.rand((1024, 1024), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        report["torch_device"] = torch.cuda.get_device_name(0)
        report["torch_capability"] = list(torch.cuda.get_device_capability(0))
        report["torch_smoke"] = float(y[0, 0].item())
except Exception as exc:
    report["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    import tensorflow as tf
    report["tensorflow"] = tf.__version__
    report["tensorflow_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
except Exception as exc:
    report["tensorflow_error"] = f"{type(exc).__name__}: {exc}"
try:
    import onnxruntime as ort
    report["onnxruntime"] = ort.__version__
    report["onnxruntime_providers"] = ort.get_available_providers()
except Exception as exc:
    report["onnxruntime_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, indent=2, ensure_ascii=False))
