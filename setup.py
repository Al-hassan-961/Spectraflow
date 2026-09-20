"""Build/install configuration for Spectraflow.

The native core (``core/``) is an *accelerator*, not a requirement. If a C++17
compiler or PyBind11 is unavailable, the extension build is skipped with a
warning and the package installs anyway -- every DSP routine has an equivalent
pure-NumPy implementation, selected automatically at import time. Set
``SPECTRAFLOW_NO_NATIVE=1`` to skip the native build deliberately.

On Android/Termux the extension must additionally be linked against libpython:
the Termux Python executable does not export its symbols to ``dlopen``-ed
modules, so an otherwise-correct build fails at import with
``cannot locate symbol "PyExc_ImportError"``.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext

try:
    from setuptools import Extension
except ImportError:  # pragma: no cover
    raise SystemExit("setuptools is required to build Spectraflow")

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Long description
# ---------------------------------------------------------------------------
try:
    LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8")
except OSError:  # pragma: no cover
    LONG_DESCRIPTION = "Wi-Fi CSI 3D sensing, pose estimation and vital signs."

NATIVE_DISABLED = os.environ.get("SPECTRAFLOW_NO_NATIVE", "") not in ("", "0", "false")

EXTENSION_NAME = "spectraflow_core"

EXTENSION_SOURCES = [
    "core/bindings.cpp",
    "core/src/csi_packet.cpp",
    "core/src/ring_buffer.cpp",
    "core/src/dsp_filters.cpp",
]


def _pybind11_include() -> str | None:
    try:
        import pybind11  # noqa: PLC0415

        return pybind11.get_include()
    except ImportError:
        return None


def _libpython_for_android() -> tuple[str, str] | None:
    """Return ``(library_name, library_dir)`` for Android/Termux, else ``None``."""
    platform = sysconfig.get_platform().lower()
    if "android" not in platform and "android" not in sys.platform.lower():
        return None
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    # Termux keeps libpython in $PREFIX/lib, which is not always LIBDIR.
    candidates = [
        sysconfig.get_config_var("LIBDIR"),
        str(Path(sys.prefix) / "lib"),
        "/data/data/com.termux/files/usr/lib",
    ]
    for directory in candidates:
        if directory and Path(directory, f"lib{version}.so").is_file():
            return version, directory
    return None


def build_extension() -> Extension | None:
    include = _pybind11_include()
    if include is None:
        print(
            "Spectraflow: pybind11 not found -- skipping the native core.\n"
            "            The pure-NumPy DSP fallback will be used.\n"
            "            Install it with:  pip install pybind11",
            file=sys.stderr,
        )
        return None

    extra_compile_args = ["-std=c++17", "-O3", "-fvisibility=hidden"]
    if sys.platform == "win32":  # pragma: no cover
        extra_compile_args = ["/std:c++17", "/O2"]

    return Extension(
        EXTENSION_NAME,
        sources=EXTENSION_SOURCES,
        include_dirs=[str(ROOT / "core" / "include"), include],
        language="c++",
        extra_compile_args=extra_compile_args,
        define_macros=[("SPECTRAFLOW_NATIVE", "1")],
    )


class OptionalBuildExt(build_ext):
    """Build the native core when possible; never fail the whole install.

    A missing toolchain must degrade to the NumPy implementation rather than
    blocking ``pip install`` on a constrained host -- an ARM Android device, a
    slim container, or a CI runner without a C++ compiler.
    """

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001
            print(
                f"Spectraflow: native core build failed ({exc}).\n"
                "            Continuing with the pure-NumPy fallback.",
                file=sys.stderr,
            )

    def build_extension(self, ext: Extension) -> None:
        android = _libpython_for_android()
        if android is not None:
            library, directory = android
            ext.libraries = list(ext.libraries or []) + [library]
            ext.library_dirs = list(ext.library_dirs or []) + [directory]
            print(f"Spectraflow: linking native core against lib{library}.so")
        super().build_extension(ext)


def main() -> None:
    extension = None if NATIVE_DISABLED else build_extension()

    setup(
        name="spectraflow",
        version="1.0.0",
        description=(
            "Wi-Fi CSI 3D sensing: contactless vital signs, human pose estimation "
            "and a WebGL visualiser."
        ),
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license="MIT",
        packages=find_packages(
            exclude=["tests", "tests.*", "firmware", "firmware.*", "build", "build.*"]
        ),
        python_requires=">=3.9",
        install_requires=[
            # NumPy is the only hard runtime dependency: every DSP stage has a
            # pure-NumPy implementation, so the pipeline runs with nothing else.
            "numpy>=1.22",
            # uvicorn is a pure ASGI server; it is what actually serves the app.
            "uvicorn>=0.20",
        ],
        extras_require={
            "native": ["pybind11>=2.10"],
            "onnx": ["onnxruntime>=1.16"],
            # FastAPI requires pydantic-core (Rust); where it cannot be built the
            # dependency-free ASGI application is served instead.
            "fastapi": ["fastapi>=0.100"],
            "dev": ["pytest>=7.0", "pybind11>=2.10"],
        },
        ext_modules=[extension] if extension is not None else [],
        cmdclass={"build_ext": OptionalBuildExt},
        entry_points={
            "console_scripts": ["spectraflow = spectraflow.server.app:main"],
        },
        include_package_data=True,
        package_data={"spectraflow": ["py.typed"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: System :: Networking :: Monitoring",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
