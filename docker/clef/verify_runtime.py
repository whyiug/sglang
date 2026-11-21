"""Check the runtime supplied by the official base and print an image build receipt."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

# Native/Python ABI versions used for the CLEF correctness and performance run.
ABI = {
    "apache-tvm-ffi": "0.1.11",
    "flashinfer-python": "0.7.1",
    "nvidia-cuda-runtime": "13.0.96",
    "sglang-kernel": "0.4.9",
    "tokenizers": "0.23.2",
    "torch": "2.14.1",
    "transformers": "5.19.0",
    "triton": "3.8.0",
}


def jit_tools(cuda_version):
    home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(home) / "bin/nvcc") if home else shutil.which("nvcc")
    if not nvcc:
        raise ValueError("nvcc is required for JIT compilation")
    root = Path(home) if home else Path(nvcc).resolve().parents[1]
    if not (root / "include/cuda_runtime.h").is_file() or not any(
        (root / lib / "libcudart.so").is_file() for lib in ("lib64", "lib")
    ):
        raise ValueError(f"CUDA headers or linkable libcudart are missing at {root}")

    versions = {}
    for tool in ("nvcc", "gcc", "g++", "ninja"):
        executable = nvcc if tool == "nvcc" else tool
        output = subprocess.check_output(
            [executable, "--version"], text=True, timeout=15
        )
        if tool == "nvcc":
            match = re.search(r"\brelease (\d+\.\d+), (V[\d.]+)", output)
            if not match or match[1] != cuda_version:
                raise ValueError(
                    f"expected nvcc for CUDA {cuda_version}, found {output}"
                )
            versions[tool] = match[2]
        else:
            versions[tool] = output.splitlines()[0]
    return versions


def verify(base, fork, python_tree, upstream):
    if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", base):
        raise ValueError("the base image must be pinned by registry digest")
    for git_id in (fork, python_tree, upstream):
        if not re.fullmatch(r"[0-9a-f]{40}", git_id):
            raise ValueError(f"expected a full Git object ID, got {git_id!r}")
    if (platform.system(), platform.machine(), sys.version_info[:2]) != (
        "Linux",
        "x86_64",
        (3, 12),
    ):
        raise ValueError("this release requires Linux/amd64 and CPython 3.12")

    installed = {package: version(package) for package in ABI}
    for name, expected in ABI.items():
        if installed[name].partition("+")[0] != expected:
            raise ValueError(
                f"{name}: installed {installed[name]}, expected {expected}"
            )

    import sglang
    import torch
    from sglang.srt import clef_policy_cache
    from sglang.version import __version__

    package = (Path(__file__).resolve().parent / "python/sglang").resolve()
    for module in (sglang, clef_policy_cache):
        if not Path(module.__file__).resolve().is_relative_to(package):
            raise ValueError(f"{module.__name__} was imported outside {package}")
    cuda = ".".join(installed["nvidia-cuda-runtime"].split(".")[:2])
    if torch.version.cuda != cuda:
        raise ValueError(
            f"torch was built for CUDA {torch.version.cuda}, expected {cuda}"
        )

    return {
        "base": base,
        "base_build_commit": os.environ.get("SGLANG_BUILD_COMMIT"),
        "fork_commit": fork,
        "upstream_commit": upstream,
        "python_tree": python_tree,
        "python": platform.python_version(),
        "sglang_version": __version__,
        "torch_cuda": torch.version.cuda,
        "abi_packages": installed,
        "jit_tools": jit_tools(cuda),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--fork", required=True)
    parser.add_argument("--python-tree", required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.base, args.fork, args.python_tree, args.upstream), indent=2
        )
    )
