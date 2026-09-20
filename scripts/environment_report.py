"""Record runtime versions and GPU facts without personal paths or credentials."""
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


def main():
    packages = ["torch", "numpy", "transformers", "safetensors", "fastapi", "pytest"]
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.system(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        x = torch.randn(256, 256, device="cuda")
        report.update(gpu=prop.name, total_vram_bytes=prop.total_memory,
                      cuda_matmul_finite=bool(torch.isfinite(x @ x).all()))
    try:
        report["nvidia_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        report["nvidia_driver"] = None
    dest = Path("artifacts/environment.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
