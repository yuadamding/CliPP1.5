"""Load exactly the packaged library, verifying ABI, sources and build identity."""

import ctypes
import hashlib
import json
from pathlib import Path

from .versions import IDENTITIES, NUMERICS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_identity(info):
    """Reconstruct the identifier embedded in the binary from its build record."""
    payload = {
        key: value
        for key, value in info.items()
        if key not in {"native_build_id", "library", "library_sha256"}
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_native():
    root = Path(__file__).resolve().parent
    try:
        info = json.loads((root / "_build_info.json").read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError("Native build identity missing; install a wheel or rebuild this source") from error
    if build_identity(info) != info.get("native_build_id"):
        raise RuntimeError("Native build record identity mismatch")
    name = info.get("library", "")
    if Path(name).name != name or not name.startswith("_native."):
        raise RuntimeError("Invalid package-owned native library name")
    path = root / name
    if path.is_symlink() or not path.is_file() or sha256(path) != info.get("library_sha256"):
        raise RuntimeError("Native library hash mismatch; rebuild/reinstall this package")
    if info.get("native_abi_version") != IDENTITIES["native_abi_version"]:
        raise RuntimeError("Native ABI identity mismatch")
    expected = {
        key.removeprefix("src/clipp/"): value
        for key, value in info["source_hashes"].items()
        if key.startswith("src/clipp/")
    }
    actual = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.suffix in {".py", ".R"}}
    if actual != expected:
        raise RuntimeError("Python source does not match the native build; rebuild before fitting")
    library = ctypes.CDLL(str(path))
    for symbol, value in (
        ("CliPPBuildId", info["native_build_id"]),
        ("CliPPModelVersion", IDENTITIES["model_version"]),
    ):
        entry = getattr(library, symbol, None)
        if entry is None:
            raise RuntimeError(f"Native identity symbol missing: {symbol}")
        entry.argtypes, entry.restype = [], ctypes.c_char_p
        if entry().decode() != value:
            raise RuntimeError(f"Native identity mismatch: {symbol}")
    library.CliPPMultiplicityVersion.argtypes = []
    library.CliPPMultiplicityVersion.restype = ctypes.c_int
    if library.CliPPMultiplicityVersion() != IDENTITIES["native_abi_version"]:
        raise RuntimeError("Native ABI version mismatch")
    library.CliPPNumerics.argtypes = []
    library.CliPPNumerics.restype = ctypes.c_char_p
    if json.loads(library.CliPPNumerics()) != NUMERICS["native"]:
        raise RuntimeError("Native numerical configuration mismatch")
    library.CliPPCUDACompiled.argtypes = []
    library.CliPPCUDACompiled.restype = ctypes.c_int
    if bool(library.CliPPCUDACompiled()) != info.get("cuda_compiled"):
        raise RuntimeError("Native backend identity mismatch")
    return library, info
