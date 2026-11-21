"""Per-dtype kernel registry and the ahead-of-time binary loader.

Every kernel file exposes ``build(_dtype) -> {name: python function}`` and
registers the result once per supported dtype via :func:`register`. The
float32 kernels live in the Warp module named after the Python module, the
float64 ones in ``<module>.f64``, so a process only compiles (or loads) the
variant it uses.

Kernel *names* are fixed and free of any content hash (``strip_hash``), which
is what lets a shipped PTX built from the full source be loaded against a copy
of the package whose kernel bodies are stubs.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform

import torch
import warp as wp

DTYPES = ((torch.float32, wp.float32, ""), (torch.float64, wp.float64, ".f64"))

# every registered (module_name, torch dtype) -> wp.Module, in registration order
MODULES: dict = {}


def register(module_name: str, build, sources: dict[str, str] | None = None) -> dict[torch.dtype, dict[str, wp.Kernel]]:
    """Instantiate ``build`` for each dtype and register the kernels.

    Args:
        module_name: ``__name__`` of the calling kernel file.
        build: callable ``(wp_dtype) -> {kernel_name: function}``; the
            functions must carry full type annotations.
        sources: optional ``{kernel_name: def-source}`` for functions whose
            source cannot be recovered with ``inspect`` (synthesized stubs).

    Returns:
        ``{torch.dtype: {kernel_name: wp.Kernel}}``
    """
    out = {}
    for tdtype, wdtype, sfx in DTYPES:
        module = wp.get_module(module_name + sfx)
        MODULES[(module_name, tdtype)] = module
        fns = build(wdtype)
        out[tdtype] = {
            name: wp.Kernel(
                fn,
                key=name,
                module=module,
                options={"enable_backward": False},
                source=(sources or {}).get(name),
            )
            for name, fn in fns.items()
        }
    return out


def wp_dtype(dtype: torch.dtype):
    return wp.float32 if dtype == torch.float32 else wp.float64


# ---------------------------------------------------------------------------
# ahead-of-time binaries
# ---------------------------------------------------------------------------
BIN_DIR_ENV = "RPG_SIM_BIN_DIR"
MANIFEST = "manifest.json"
STUB_MARKER = "STUBBED"  # written by tools/package_student_sim.py next to the kernel files
PTX_ARCH = 75  # forward-compatible PTX target: Turing and everything newer


def default_bin_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bin")


def bin_dir() -> str | None:
    """Directory holding the shipped binaries, or None when running from source."""
    d = os.environ.get(BIN_DIR_ENV) or default_bin_dir()
    return d if os.path.isfile(os.path.join(d, MANIFEST)) else None


SIGNATURES = "signatures.json"
_ALIASES = (
    "_vec3 = wp.types.vector(length=3, dtype=_dtype)",
    "_vec4 = wp.types.vector(length=4, dtype=_dtype)",
    "_mat33 = wp.types.matrix(shape=(3, 3), dtype=_dtype)",
)


def export_signatures(path: str) -> None:
    """Write what a stub tree needs to bind the binaries: per kernel module the
    ``P_*`` param-table indices and every kernel's argument list (annotation
    source text). No kernel body is included."""
    import ast
    import sys

    data = {}
    for (mod_name, tdtype), _ in MODULES.items():
        if tdtype != torch.float32:
            continue
        pymod = sys.modules[mod_name]
        consts = {k: v for k, v in vars(pymod).items() if k.startswith("P_") and isinstance(v, int)}
        kernels = {}
        for name, kernel in pymod.KERNELS[torch.float32].items():
            fn = ast.parse(kernel.adj.source).body[0]
            kernels[name] = [(a.arg, ast.unparse(a.annotation)) for a in fn.args.args]
        data[mod_name] = {"constants": consts, "kernels": kernels}
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)


def _stub_source(entry: dict) -> tuple[str, dict[str, str]]:
    lines = ["def build(_dtype):"] + [f"    {a}" for a in _ALIASES]
    per_kernel = {}
    for name, args in entry["kernels"].items():
        sig = ", ".join(f"{n}: {t}" for n, t in args)
        src = f"def {name}({sig}):\n    pass\n"
        per_kernel[name] = src
        lines += ["    " + ln for ln in src.splitlines()]
    lines.append("    return {" + ", ".join(f"{n!r}: {n}" for n in entry["kernels"]) + "}")
    return "\n".join(lines) + "\n", per_kernel


def synthesize_stub_modules(directory: str) -> None:
    """Student mode: create the ``rpg_simulator.kernels.<name>`` modules from
    ``signatures.json`` (no kernel sources on disk) so the glue can import
    ``KERNELS`` / ``P_*`` and the binaries bind by name."""
    import sys
    import types

    with open(os.path.join(directory, SIGNATURES)) as f:
        data = json.load(f)
    pkg = sys.modules[__package__]
    for mod_name, entry in data.items():
        src, per_kernel = _stub_source(entry)
        mod = types.ModuleType(mod_name)
        mod.__dict__.update(entry["constants"])
        ns = {"wp": wp, "__name__": mod_name}
        exec(compile(src, f"<{mod_name} stub>", "exec", dont_inherit=True), ns)
        mod.KERNELS = register(mod_name, ns["build"], sources=per_kernel)
        sys.modules[mod_name] = mod
        setattr(pkg, mod_name.rsplit(".", 1)[1], mod)


# "" = warp's generic x86-64 target: -target-cpu generic plus the +f16c warp always adds
# (implies AVX) -> runs on any Intel/AMD CPU with AVX (2012+). The default
# -march=native would tie the .o to the build host's ISA (AVX-512 here). Note warp's
# embedded clang ignores other -march= values.
CPU_FLAGS = ""


@contextlib.contextmanager
def _cpu_aot_options(module):
    """``wp.launch`` on the CPU always uses ``block_dim=1`` and looks the module
    exec up under that key, while ``load_aot_module`` keys it by the module's
    default (256): compile/load CPU objects under 1, without touching the CUDA
    default of the same module. Also pins the portable compiler flags (they
    are part of the object file name)."""
    saved = module.options["block_dim"], module.options["cpu_compiler_flags"]
    module.options["block_dim"] = 1
    module.options["cpu_compiler_flags"] = CPU_FLAGS
    try:
        yield
    finally:
        module.options["block_dim"], module.options["cpu_compiler_flags"] = saved


def compile_binaries(
    out_dir: str, *, arch: int = PTX_ARCH, cpu: bool = True, extra: dict | None = None
) -> None:
    """Compile every registered kernel module to PTX (and a CPU object) in ``out_dir``.

    Import the kernel modules before calling this (``import rpg_simulator.kernels``
    does). Writes ``manifest.json`` and removes the generated ``.cu``/``.cpp`` sources.
    The CPU objects are host-specific (``platform.machine()``, same warp version).
    """
    import rpg_simulator.kernels  # noqa: F401  registers everything

    os.makedirs(out_dir, exist_ok=True)
    names = []
    for (mod_name, tdtype), module in MODULES.items():
        if wp.is_cuda_available():
            wp.compile_aot_module(module, arch=arch, module_dir=out_dir, use_ptx=True, strip_hash=True)
        if cpu:
            with _cpu_aot_options(module):
                wp.compile_aot_module(module, device="cpu", module_dir=out_dir, strip_hash=True)
        names.append(module.name)
    for f in os.listdir(out_dir):
        if f.endswith((".cu", ".cpp")):
            os.remove(os.path.join(out_dir, f))
    manifest = {
        "warp_version": wp.__version__,
        "arch": arch if wp.is_cuda_available() else None,
        "cpu": f"{platform.system().lower()}-{platform.machine()}" if cpu else None,
        "cpu_flags": CPU_FLAGS if cpu else None,
        "modules": names,
        **(extra or {}),
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    export_signatures(os.path.join(out_dir, SIGNATURES))


def load_binaries(device, dtype: torch.dtype, directory: str | None = None) -> None:
    """Load the shipped PTX / CPU object for every kernel module of ``dtype`` on ``device``.

    Raises if a binary is missing or was built with a different Warp version:
    silently falling back to JIT would compile the stub kernel bodies.
    """
    directory = directory or bin_dir()
    if directory is None:
        raise FileNotFoundError("rpg_simulator binaries not found; set RPG_SIM_BIN_DIR or install the source package")
    with open(os.path.join(directory, MANIFEST)) as f:
        manifest = json.load(f)
    if manifest["warp_version"] != wp.__version__:
        raise RuntimeError(
            f"rpg_simulator binaries were built with warp {manifest['warp_version']}, "
            f"this environment has {wp.__version__}"
        )
    if wp.get_device(device).is_cpu:
        host = f"{platform.system().lower()}-{platform.machine()}"
        if manifest.get("cpu") != host:
            raise RuntimeError(f"rpg_simulator binaries have no CPU build for {host} (manifest: {manifest.get('cpu')})")
        if manifest.get("cpu_flags") != CPU_FLAGS:
            raise RuntimeError(f"rpg_simulator CPU binaries were built with cpu_compiler_flags={manifest.get('cpu_flags')!r}, expected {CPU_FLAGS!r}")
    for (mod_name, tdtype), module in MODULES.items():
        if tdtype != dtype:
            continue
        if wp.get_device(device).is_cpu:
            with _cpu_aot_options(module):
                wp.load_aot_module(module, device=device, module_dir=directory, strip_hash=True)
        else:
            # arch from the manifest, not the current device: the shipped PTX is
            # sm_75 and the driver JIT-compiles it forward for newer GPUs.
            wp.load_aot_module(
                module, device=device, arch=manifest["arch"], module_dir=directory, use_ptx=True, strip_hash=True
            )


_LOADED: set[tuple[str, torch.dtype]] = set()


def ensure_loaded(device, dtype: torch.dtype, *modules) -> None:
    """Load binaries once per (device, dtype) when a binary directory is present,
    then warm ``modules`` on ``device`` (JIT-compiles them in a source tree)."""
    if bin_dir() is None:
        if os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), STUB_MARKER)):
            raise RuntimeError(
                "this rpg_simulator ships stub kernels and needs its precompiled binaries: "
                f"none found in {default_bin_dir()} (or ${BIN_DIR_ENV})"
            )
    else:
        key = (wp.get_device(device).alias, dtype)
        if key not in _LOADED:
            load_binaries(device, dtype)
            _LOADED.add(key)
            print(f"[rpg_simulator] loaded precompiled kernels ({dtype}) from {bin_dir()}")
    cpu = wp.get_device(device).is_cpu
    for module in modules:
        module.load(device, block_dim=1 if cpu else None)  # same key wp.launch uses
