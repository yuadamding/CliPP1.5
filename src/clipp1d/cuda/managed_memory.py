"""Explicit, bounded host-backed storage for otherwise unchanged CUDA inference.

This is an opt-in execution experiment, not CPU numerical fallback. All tensor
operations still run on CUDA. A private PyTorch caching pool uses CUDA managed
allocations, whose pages the driver may migrate to host RAM. The reservation,
actual host availability, cgroup limit and unchanged memory headroom all gate
admission. Allocation failures propagate; ordinary fits use their old VRAM gate.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import threading

import torch

from .policy import CudaPolicy


# Allocation plumbing only: no likelihood, graph, solver or numerical kernels.
# Included in this Python file so source_provenance binds the native source too.
NATIVE_SOURCE = r'''
#include <cuda_runtime_api.h>
#include <sys/types.h>
#include <pthread.h>
#include <stdint.h>

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static uint64_t ceiling=0, current=0, peak=0, calls=0, failed=0;
static int selected=-1, release_error=0;
int clipp_configure(uint64_t limit, int device) {
  pthread_mutex_lock(&lock);
  if (ceiling || current || !limit) { pthread_mutex_unlock(&lock); return -1; }
  int managed=0, concurrent=0;
  if (cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device)!=cudaSuccess ||
      cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device)!=cudaSuccess ||
      !managed || !concurrent) { pthread_mutex_unlock(&lock); return -2; }
  ceiling=limit; selected=device;
  pthread_mutex_unlock(&lock); return 0;
}
void* clipp_managed_alloc(ssize_t bytes, int device, cudaStream_t stream) {
  pthread_mutex_lock(&lock);
  if (bytes<=0 || device!=selected || (uint64_t)bytes>ceiling-current) {
    ++failed; pthread_mutex_unlock(&lock); return NULL;
  }
  void* pointer=NULL;
  cudaError_t error=cudaSetDevice(device);
  if (error==cudaSuccess) error=cudaMallocManaged(&pointer, bytes, cudaMemAttachGlobal);
  if (error!=cudaSuccess) { ++failed; cudaGetLastError(); pthread_mutex_unlock(&lock); return NULL; }
  current+=bytes; peak=current>peak?current:peak; ++calls;
  pthread_mutex_unlock(&lock); return pointer;
}
void clipp_managed_free(void* pointer, size_t bytes, int device, cudaStream_t stream) {
  pthread_mutex_lock(&lock);
  if (device!=selected || (uint64_t)bytes>current) { release_error=-1; pthread_mutex_unlock(&lock); return; }
  cudaError_t error=cudaSetDevice(device);
  if (error==cudaSuccess) error=cudaFree(pointer);
  if (error!=cudaSuccess) { release_error=(int)error; pthread_mutex_unlock(&lock); return; }
  current-=bytes; pthread_mutex_unlock(&lock);
}
int clipp_is_managed(void* pointer) {
  struct cudaPointerAttributes attributes;
  cudaError_t error=cudaPointerGetAttributes(&attributes,pointer);
  return error==cudaSuccess && attributes.type==cudaMemoryTypeManaged;
}
#define GETTER(name,field) uint64_t name(void) { pthread_mutex_lock(&lock); uint64_t value=field; pthread_mutex_unlock(&lock); return value; }
GETTER(clipp_current,current)
GETTER(clipp_peak,peak)
GETTER(clipp_calls,calls)
GETTER(clipp_failed,failed)
int clipp_release_error(void) { pthread_mutex_lock(&lock); int value=release_error; pthread_mutex_unlock(&lock); return value; }
'''

_ACTIVE = ContextVar('clipp_managed_cuda_workspace', default=None)
_KEEPALIVE = []
HOST_HEADROOM = 16 * 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def host_capacity(reservation, available, cgroup_available=None):
    """CPU-only resource arithmetic; no inference or GPU-capacity substitution."""
    values = [reservation, available]
    if cgroup_available is not None:
        values.append(cgroup_available)
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError('Managed storage needs positive integer host byte limits')
    usable = min(values) - HOST_HEADROOM
    if usable <= 0:
        raise MemoryError('Insufficient host RAM after explicit headroom')
    return usable


def _host_available():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].split()[0]) * 1024
    limits = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            base, limit_name, used_name = Path('/sys/fs/cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            base = Path('/sys/fs/cgroup/memory')
            limit_name, used_name = 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        # A container may expose its cgroup directly as the mount root.
        candidates = {base, (base/relative.lstrip('/')).resolve()}
        for candidate in list(candidates):
            if candidate.is_relative_to(base):
                candidates.update(p for p in candidate.parents if p.is_relative_to(base))
        for path in candidates:
            if not path.is_relative_to(base):
                continue
            if not (path/limit_name).is_file() or not (path/used_name).is_file():
                continue
            bound = (path/limit_name).read_text().strip()
            if bound != 'max':
                limits.append(max(0, int(bound)-int((path/used_name).read_text())))
    return available, min(limits) if limits else None


def build_allocator(destination, compiler):
    """Compile allocation plumbing using the bound environment's CUDA runtime."""
    destination = Path(destination)
    destination.mkdir()
    import triton
    headers = Path(triton.__file__).resolve().parent/'backends/nvidia/include'
    cuda = Path(torch.__file__).resolve().parent.parent/'nvidia/cuda_runtime'
    runtime, = (cuda/'lib').glob('libcudart.so.*')
    source = destination/'managed_allocator.c'
    source.write_text(NATIVE_SOURCE)
    library = destination/'managed_allocator.so'
    arguments = [str(compiler), '-x', 'c', '-std=c11', '-shared', '-fPIC', '-O2',
        '-pthread', str(source), '-I'+str(cuda/'include'),
        '-I'+str(headers), '-x', 'none', str(runtime),
        '-Wl,-rpath,'+str(runtime.parent), '-o', str(library)]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError('Managed allocator compilation failed: '+result.stderr[-4000:])
    return dict(library=str(library.resolve()), library_sha256=sha(library),
        native_source_sha256=sha(source), compiler=str(compiler), compiler_sha256=sha(compiler),
        cuda_runtime=str(runtime), cuda_runtime_sha256=sha(runtime), arguments=arguments)


def current_capacity(device):
    """The API can recognize only a live, verified pool in the calling thread."""
    active = _ACTIVE.get()
    if active is None:
        return None
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if (active['pid'] != os.getpid() or active['thread'] != threading.get_ident() or
            active['device'] != index or active['native'].clipp_release_error()):
        raise RuntimeError('Managed workspace identity or release invariant changed')
    return dict(active['admission'])


def native_stats(native):
    return dict(current_bytes=native.clipp_current(), peak_bytes=native.clipp_peak(),
                allocations=native.clipp_calls(), rejected_allocations=native.clipp_failed(),
                release_error=native.clipp_release_error())


@contextmanager
def workspace(binding, host_reservation_bytes, *, device='cuda:0'):
    """Use a bound managed pool, never a silent retry or a global allocator swap."""
    if _ACTIVE.get() is not None:
        raise ValueError('Nested managed workspaces are not supported')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Managed seed inference requires exactly one assigned CUDA device')
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index != 0:
        raise ValueError('Managed seed workspace requires the assigned CUDA device zero')
    for name in ('library', 'compiler', 'cuda_runtime'):
        if sha(binding[name]) != binding[name+'_sha256']:
            raise ValueError('Changed managed allocator '+name)
    if hashlib.sha256(NATIVE_SOURCE.encode()).hexdigest() != binding['native_source_sha256']:
        raise ValueError('Different allocation plumbing source')
    available, cgroup_available = _host_available()
    usable = host_capacity(host_reservation_bytes, available, cgroup_available)
    fraction = CudaPolicy().memory_fraction
    limit = int(fraction * usable)
    native = ctypes.CDLL(binding['library'])
    native.clipp_configure.argtypes = [ctypes.c_uint64, ctypes.c_int]
    native.clipp_configure.restype = ctypes.c_int
    native.clipp_is_managed.argtypes = [ctypes.c_void_p]
    native.clipp_is_managed.restype = ctypes.c_int
    for name in ('current', 'peak', 'calls', 'failed'):
        getattr(native, 'clipp_'+name).restype = ctypes.c_uint64
    if native.clipp_configure(limit, index):
        raise RuntimeError('CUDA managed memory/concurrent access is unsupported or allocator was reused')
    allocator = torch.cuda.memory.CUDAPluggableAllocator(binding['library'], 'clipp_managed_alloc', 'clipp_managed_free')
    pool = torch.cuda.MemPool(allocator.allocator(), use_on_oom=False)
    # Keep the library alive through tensor/cache destructors at process exit.
    _KEEPALIVE.append((native, allocator, pool))
    with torch.cuda.use_mem_pool(pool, device=index):
        probe = torch.zeros(8, dtype=torch.float64, device=device)
        torch.cuda.synchronize(index)
        if native.clipp_is_managed(probe.data_ptr()) != 1 or native.clipp_calls() == 0:
            raise RuntimeError('PyTorch did not route the probe into managed CUDA storage')
        del probe
        admission = dict(backend='pytorch_cuda_managed_host_backing_v1',
            capacity_bytes=usable, allocation_limit_bytes=limit, memory_fraction=fraction,
            host_reservation_bytes=host_reservation_bytes, host_available_bytes=available,
            cgroup_available_bytes=cgroup_available, host_headroom_bytes=HOST_HEADROOM,
            allocator_binding=binding, numerical_device='cuda:0', cpu_numeric_fallback=False)
        state = dict(pid=os.getpid(), thread=threading.get_ident(), device=index,
                     native=native, admission=admission)
        token = _ACTIVE.set(state)
        try:
            yield state
            torch.cuda.synchronize(index)
            if native.clipp_release_error():
                raise RuntimeError('CUDA managed allocation release failed')
        finally:
            _ACTIVE.reset(token)
