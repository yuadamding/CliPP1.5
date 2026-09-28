"""Receipt-bound compiler settings; the mathematical source is unchanged."""
import json
from pathlib import Path
import runpy
import sys

import torch._inductor.config as config

generation = Path(sys.argv[1]).resolve()
settings = json.loads(generation.read_text())['compiler_settings']
if settings != {'triton.persistent_reductions': False, 'epilogue_fusion': False,
                'max_fusion_size': 1, 'allow_buffer_reuse': False}:
    raise ValueError('Unsupported compiler execution contract')
arguments = sys.argv[2:]
if arguments[0] == '-B':
    arguments = arguments[1:]
with config.patch(settings):
    if (config.triton.persistent_reductions or config.epilogue_fusion or
            config.max_fusion_size != 1 or config.allow_buffer_reuse):
        raise ValueError('Compiler settings did not take effect')
    print(json.dumps(dict(compiler_execution_settings=settings)), flush=True)
    if arguments[0] == '-m':
        module = arguments[1]
        sys.argv = arguments[1:]
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    else:
        script = arguments[0]
        sys.argv = arguments
        sys.path.insert(0, str(Path(script).resolve().parent))
        runpy.run_path(script, run_name='__main__')
