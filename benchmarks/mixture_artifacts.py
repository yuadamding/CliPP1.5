"""Resolve a verified remote artifact tree without rewriting its original receipts."""
import json
from pathlib import Path


class ArtifactPaths:
    def __init__(self, manifest=None):
        self.mapping = None
        if manifest is not None:
            value = json.loads(Path(manifest).read_text())
            if set(value) != {'schema', 'remote_root', 'local_root'} or value['schema'] != 'clipp1d.artifact_tree_mapping.v1':
                raise ValueError('Unrecognized artifact path mapping')
            self.remote = Path(value['remote_root'])
            self.local = Path(value['local_root'])
            if (not self.remote.is_absolute() or '..' in self.remote.parts or
                    self.remote == Path('/') or not self.local.is_absolute() or
                    not self.local.is_dir() or self.local.resolve() != self.local):
                raise ValueError('Unsafe artifact roots')
            self.mapping = value

    def __call__(self, name):
        path = Path(name)
        if self.mapping is None or not path.is_relative_to(self.remote):
            return path
        relative = path.relative_to(self.remote)
        if '..' in relative.parts:
            raise ValueError('Artifact escapes mapped root')
        candidate = self.local/relative
        if candidate.resolve() != candidate or not candidate.resolve().is_relative_to(self.local):
            raise ValueError('Symlink or escaping artifact')
        return candidate
