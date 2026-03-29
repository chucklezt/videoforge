"""ROCm environment validation utilities."""

import os
import shutil
import subprocess
import sys


def check_rocm_env() -> list[tuple[str, bool, str]]:
    """Run all ROCm environment checks. Returns list of (check_name, passed, detail)."""
    results = []

    # Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results.append(("Python", True, py_ver))

    # OS
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
                    results.append(("OS", True, os_name))
                    break
    except FileNotFoundError:
        results.append(("OS", False, "Could not read /etc/os-release"))

    # ROCm
    rocm_version_file = "/opt/rocm/.info/version"
    if os.path.exists(rocm_version_file):
        with open(rocm_version_file) as f:
            rocm_ver = f.read().strip()
        results.append(("ROCm", True, rocm_ver))
    else:
        results.append(("ROCm", False, "Not found at /opt/rocm"))

    # HSA override
    hsa = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if hsa == "10.3.0":
        results.append(("HSA_OVERRIDE_GFX_VERSION", True, hsa))
    elif hsa:
        results.append(("HSA_OVERRIDE_GFX_VERSION", True, f"{hsa} (expected 10.3.0)"))
    else:
        results.append(("HSA_OVERRIDE_GFX_VERSION", False, "NOT SET"))

    # PyTorch
    try:
        import torch
        results.append(("PyTorch", True, torch.__version__))
        gpu_available = torch.cuda.is_available()
        results.append(("CUDA/ROCm available", gpu_available, str(gpu_available)))

        if gpu_available:
            dev_name = torch.cuda.get_device_name(0)
            results.append(("GPU", True, dev_name))
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            results.append(("VRAM", True, f"{vram_gb:.1f} GB"))

            # Compute test
            try:
                x = torch.randn(256, 256, device="cuda")
                _ = x @ x.T
                results.append(("GPU compute", True, "PASSED"))
            except Exception as e:
                results.append(("GPU compute", False, str(e)))

            # SDPA test
            try:
                q = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16)
                k = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16)
                v = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16)
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                results.append(("SDPA attention", True, "PASSED"))
            except Exception as e:
                results.append(("SDPA attention", False, str(e)))
    except ImportError:
        results.append(("PyTorch", False, "Not installed"))

    # Key libraries
    libs = [
        ("accelerate", "accelerate"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("cv2", "opencv"),
        ("scenedetect", "scenedetect"),
    ]
    for module_name, display_name in libs:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            results.append((display_name, True, ver))
        except ImportError:
            results.append((display_name, False, "Not installed"))

    # FFmpeg
    if shutil.which("ffmpeg"):
        try:
            out = subprocess.check_output(["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True)
            ffmpeg_ver = out.split("\n")[0].split(" ")[2] if out else "unknown"
            results.append(("FFmpeg", True, ffmpeg_ver))
        except Exception:
            results.append(("FFmpeg", True, "available (version unknown)"))
    else:
        results.append(("FFmpeg", False, "Not found"))

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        results.append(("RAM", True, f"{ram_gb:.0f} GB"))
    except ImportError:
        results.append(("RAM", False, "psutil not installed"))

    # Disk
    try:
        disk = shutil.disk_usage(os.path.expanduser("~"))
        free_gb = disk.free / (1024**3)
        results.append(("Disk free", True, f"{free_gb:.0f} GB"))
    except Exception as e:
        results.append(("Disk free", False, str(e)))

    # xformers (should NOT be available on AMD)
    try:
        import xformers
        results.append(("xformers", True, f"{xformers.__version__} (unexpected on AMD)"))
    except ImportError:
        results.append(("xformers", True, "Not installed (expected on AMD, using SDPA)"))

    return results


def print_validation_report(results: list[tuple[str, bool, str]]) -> bool:
    """Print a formatted validation report. Returns True if all critical checks pass."""
    print("VideoForge Environment Check")
    print("=" * 50)

    all_passed = True
    for name, passed, detail in results:
        marker = "OK  " if passed else "FAIL"
        print(f"  {marker}  {name}: {detail}")
        if not passed and name in ("PyTorch", "CUDA/ROCm available", "GPU", "FFmpeg", "ROCm"):
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("  Environment is ready.")
    else:
        print("  Some critical checks failed. Review above.")
    return all_passed
