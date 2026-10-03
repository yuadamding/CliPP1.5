"""Resolve fitting configuration once, before any work or output creation."""

from dataclasses import dataclass, fields
import math
import os

from ._flags import parse_flag

MAX_CLUSTERS = 20


@dataclass(frozen=True)
class FitConfig:
    sample_id: str = "sample"
    device: str = "auto"
    clusters: int | None = None
    max_clusters: int = 10
    subsample_size: int | None = None
    replicates: int = 1
    seed: int = 0
    window_size: float = 0.05
    overlap: float = 0.0
    assembly: str = "unspecified"

    def __post_init__(self):
        if (
            not isinstance(self.sample_id, str)
            or not self.sample_id.strip()
            or any(c in self.sample_id for c in "/\\\t\r\n")
        ):
            raise ValueError("sample_id must be a nonempty name, separate from the output path")
        if (
            not isinstance(self.assembly, str)
            or not self.assembly.strip()
            or any(c in self.assembly for c in "\t\r\n")
        ):
            raise ValueError("assembly must be a nonempty label")
        if not isinstance(self.device, str) or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        for name, value in (("max_clusters", self.max_clusters), ("clusters", self.clusters)):
            if (name == "max_clusters" or value is not None) and (
                type(value) is not int or not 1 <= value <= MAX_CLUSTERS
            ):
                raise ValueError(f"{name} must be an integer in 1..{MAX_CLUSTERS}")
        if self.subsample_size is not None and (
            type(self.subsample_size) is not int or self.subsample_size < 1
        ):
            raise ValueError("subsample_size must be a positive integer")
        if type(self.replicates) is not int or self.replicates < 1:
            raise ValueError("replicates must be a positive integer")
        if self.subsample_size is None and self.replicates != 1:
            raise ValueError("Multiple replicates require subsampling")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1 - self.replicates:
            raise ValueError("seed + replicate must fit uint32")
        if not (
            type(self.window_size) in (float, int)
            and type(self.overlap) in (float, int)
            and math.isfinite(self.window_size)
            and math.isfinite(self.overlap)
            and 0 <= self.overlap < self.window_size <= 1
        ):
            raise ValueError("Require 0 <= overlap < window_size <= 1")

    @classmethod
    def from_dict(cls, value):
        """Deserialize the complete recorded contract without resolving environment flags."""
        if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
            raise ValueError("Recorded configuration must contain exactly the FitConfig fields")
        return cls(**value)

    def resolved_device(self):
        forced = parse_flag(os.environ.get("CLIPP_FORCE_CPU"), "CLIPP_FORCE_CPU") is True
        required = parse_flag(os.environ.get("CLIPP_REQUIRE_CUDA"), "CLIPP_REQUIRE_CUDA") is True
        if (forced and required) or (forced and self.device == "cuda") or (required and self.device == "cpu"):
            raise ValueError("Conflicting device and CLIPP_FORCE_CPU/CLIPP_REQUIRE_CUDA flags")
        return "cpu" if forced else "cuda" if required else self.device

    def capacities(self, count):
        if self.clusters is not None and self.clusters > count:
            raise ValueError("Requested K exceeds retained mutations")
        values = (
            [self.clusters]
            if self.clusters is not None
            else list(range(1, min(count, self.max_clusters) + 1))
        )
        if not values or (self.subsample_size is not None and self.subsample_size < max(values)):
            raise ValueError("Retained/subsample size must support every requested K")
        return values
