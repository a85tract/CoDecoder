"""Minimal CUDA driver-API access from Python (ctypes on libcuda.so.1, no compiler or extra package on the host): load a
cubin, set function attributes, launch a kernel whose single parameter is a ctypes.Structure, on torch's current
stream (so the launch is captured by torch.cuda.graph). Used by engine.py to load and launch the d = 7 kernel."""
import ctypes

import torch

_cu = None
MAX_DYNAMIC_SHARED = 8                  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
PROGRAMMATIC_SERIALIZATION = 6          # CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
NON_PORTABLE_CLUSTER = 14               # CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
NUM_REGS = 4


class LaunchAttr(ctypes.Structure):     # CUlaunchAttribute: id, 4 pad bytes, a 64-byte value union
    _fields_ = [("id", ctypes.c_int), ("pad", ctypes.c_char * 4), ("value", ctypes.c_int), ("rest", ctypes.c_char * 60)]


class LaunchConfig(ctypes.Structure):   # CUlaunchConfig
    _fields_ = [("gx", ctypes.c_uint), ("gy", ctypes.c_uint), ("gz", ctypes.c_uint), ("bx", ctypes.c_uint),
                ("by", ctypes.c_uint), ("bz", ctypes.c_uint), ("smem", ctypes.c_uint), ("stream", ctypes.c_void_p),
                ("attrs", ctypes.POINTER(LaunchAttr)), ("nattrs", ctypes.c_uint)]


def cu():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_void_p
        _cu.cuModuleLoadData.argtypes = [ctypes.POINTER(v), ctypes.c_char_p]
        _cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(v), v, ctypes.c_char_p]
        _cu.cuFuncSetAttribute.argtypes = [v, ctypes.c_int, ctypes.c_int]
        _cu.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, v]
        _cu.cuLaunchKernel.argtypes = [v, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, v, ctypes.POINTER(v), ctypes.POINTER(v)]
        _cu.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        _cu.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        _cu.cuLaunchKernelEx.argtypes = [ctypes.POINTER(LaunchConfig), v, ctypes.POINTER(v), ctypes.POINTER(v)]
    return _cu


def check(r, what):
    if r:
        name = ctypes.c_char_p()
        cu().cuGetErrorName(r, ctypes.byref(name))
        raise RuntimeError(f"{what}: {name.value.decode() if name.value else r}")


def driver_version():
    v = ctypes.c_int()
    check(cu().cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
    return v.value


class Module:
    def __init__(self, image: bytes):
        torch.cuda.init()
        torch.zeros(1, device="cuda")                      # the primary context is current on this thread
        self.image = image
        self.mod = ctypes.c_void_p()
        check(cu().cuModuleLoadData(ctypes.byref(self.mod), image), "cuModuleLoadData")

    def kernel(self, name, smem=0, nonportable=False):
        return Kernel(self, name, smem, nonportable)


class Kernel:
    def __init__(self, mod, name, smem, nonportable):
        self.name, self.mod = name, mod
        self.f = ctypes.c_void_p()
        check(cu().cuModuleGetFunction(ctypes.byref(self.f), mod.mod, name.encode()), f"get {name}")
        if smem > 48 * 1024:
            check(cu().cuFuncSetAttribute(self.f, MAX_DYNAMIC_SHARED, int(smem)), f"{name}: dynamic smem {smem}")
        if nonportable:
            check(cu().cuFuncSetAttribute(self.f, NON_PORTABLE_CLUSTER, 1), f"{name}: non-portable cluster")

    def regs(self):
        v = ctypes.c_int()
        check(cu().cuFuncGetAttribute(ctypes.byref(v), NUM_REGS, self.f), "regs")
        return v.value

    def __call__(self, grid, block, smem, params, stream=None, pdl=False):
        """params: one ctypes object (a kernel with one parameter, e.g. a struct) or a list of them (one per
        parameter, in order). pdl: launch with programmatic stream serialization (the kernel may start before the
        previous kernel in the stream has finished; it must wait with griddepcontrol.wait before using its outputs)."""
        s = torch.cuda.current_stream().cuda_stream if stream is None else stream
        ps = params if isinstance(params, (list, tuple)) else [params]
        arr = (ctypes.c_void_p * len(ps))(*[ctypes.addressof(p) for p in ps])
        if not pdl:
            check(cu().cuLaunchKernel(self.f, grid, 1, 1, block, 1, 1, int(smem), ctypes.c_void_p(s), arr, None),
                  f"launch {self.name}")
            return
        at = LaunchAttr(id=PROGRAMMATIC_SERIALIZATION, value=1)
        cfg = LaunchConfig(gx=grid, gy=1, gz=1, bx=block, by=1, bz=1, smem=int(smem), stream=s,
                           attrs=ctypes.pointer(at), nattrs=1)
        check(cu().cuLaunchKernelEx(ctypes.byref(cfg), self.f, arr, None), f"launch (PDL) {self.name}")
