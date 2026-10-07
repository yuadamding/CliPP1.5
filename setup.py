"""Build one package-owned native library with verifiable source/build identity."""

from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
namespace = {}
exec((ROOT / "src/clipp/_flags.py").read_text(), namespace)
parse_flag = namespace["parse_flag"]


def _nvidia_package_paths(name):
    try:
        spec = importlib.util.find_spec(name)
    except (ModuleNotFoundError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    return (
        (root / "include", root / "lib") if (root / "include").is_dir() and (root / "lib").is_dir() else None
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Reproducible default: CPU. CUDA is an explicit, separately qualified build.
use_cuda = parse_flag(os.environ.get("CLIPP_USE_CUDA"), "CLIPP_USE_CUDA") is True
sources = [
    "src/clipp/csrc/kernel_common.cpp",
    "src/clipp/csrc/kernel_cpu.cpp",
    "src/clipp/csrc/kernel_dispatch.cpp",
    "src/clipp/csrc/kernel_contract.cpp",
]
include_dirs, link_args, macros, library_dirs, libraries, rpaths = [], [], [], [], [], []
if use_cuda:
    runtime = _nvidia_package_paths("nvidia.cuda_runtime")
    nvrtc = _nvidia_package_paths("nvidia.cuda_nvrtc")
    driver = next(
        (
            Path(d)
            for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/cuda/lib64"]
            if (Path(d) / "libcuda.so").exists()
        ),
        None,
    )
    if not runtime or not nvrtc or driver is None:
        raise RuntimeError(
            "Explicit CUDA build requires libcuda.so, nvidia-cuda-runtime-cu12 and nvidia-cuda-nvrtc-cu12 in the build environment"
        )
    include_dirs += [str(runtime[0]), str(nvrtc[0])]
    for directory, name in ((runtime[1], "libcudart.so.12"), (nvrtc[1], "libnvrtc.so.12")):
        if not (directory / name).is_file():
            raise RuntimeError(f"Missing CUDA build library: {directory / name}")
        link_args.append(str(directory / name))
        rpaths.append(str(directory))
    library_dirs.append(str(driver))
    libraries.append("cuda")
    macros.append(("USE_CUDA", "1"))
    sources.append("src/clipp/csrc/kernel_cuda_backend.cpp")


class BuildExt(build_ext):
    def get_source_files(self):
        return super().get_source_files() + [
            str(p.relative_to(ROOT))
            for directory in ("sample", "tests")
            for p in sorted((ROOT / directory).glob("*"))
            if p.is_file()
        ]

    def finalize_options(self):
        super().finalize_options()
        self.force = True

    def build_extensions(self):
        source_paths = sorted(
            p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".cpp", ".h", ".inc"}
        )
        source_paths += [ROOT / name for name in ("setup.py", "pyproject.toml")]
        source_hashes = {str(p.relative_to(ROOT)): _sha(p) for p in source_paths}
        try:
            top = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if Path(top).resolve() != ROOT:
                raise OSError("Archive is not this Git checkout")
            commit = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None  # Source archives are identified by hashes, never invented commits.
        self.identity = {
            "native_abi_version": 3,
            "model_version": "uniform_1_to_major_v1",
            "cuda_compiled": use_cuda,
            "source_hashes": source_hashes,
            "compiler": self.compiler.compiler_cxx,
            "compiler_version": subprocess.check_output(
                [*self.compiler.compiler_cxx, "--version"], text=True
            ).strip(),
            "compiler_so": self.compiler.compiler_so,
            "linker_so": self.compiler.linker_so,
            "compile_args": ["-O3", "-std=c++17"],
            "link_args": link_args,
            "include_dirs": include_dirs,
            "library_dirs": library_dirs,
            "libraries": libraries,
            "runtime_library_dirs": rpaths,
            "compiler_environment": {
                key: os.environ.get(key) for key in ("CC", "CXX", "CFLAGS", "CPPFLAGS", "CXXFLAGS", "LDFLAGS")
            },
            "source_identity_policy": "hashes identify built source; commit is checkout ancestry, not a claim of a clean tree",
            "python": sys.version,
            "source_commit": commit,
        }
        self.identity["native_build_id"] = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode()
        ).hexdigest()
        for ext in self.extensions:
            ext.define_macros.append(("CLIPP_BUILD_ID", '"' + self.identity["native_build_id"] + '"'))
        super().build_extensions()

    def run(self):
        super().run()
        for ext in self.extensions:
            path = Path(self.get_ext_fullpath(ext.name))
            info = {**self.identity, "library": path.name, "library_sha256": _sha(path)}
            (path.parent / "_build_info.json").write_text(json.dumps(info, sort_keys=True, indent=2) + "\n")


setup(
    ext_modules=[
        Extension(
            "clipp._native",
            sources,
            depends=[
                "src/clipp/csrc/kernel_common.h",
                "src/clipp/csrc/kernel_cuda_backend.cpp",
                "src/clipp/csrc/kernel_cuda_kernels.inc",
            ],
            language="c++",
            include_dirs=include_dirs,
            define_macros=macros,
            libraries=libraries,
            library_dirs=library_dirs,
            runtime_library_dirs=rpaths,
            extra_compile_args=["-O3", "-std=c++17"],
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
