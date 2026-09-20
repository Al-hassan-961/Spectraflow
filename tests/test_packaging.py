"""Packaging integrity tests.

These guard failure modes that are invisible to the DSP tests but break the very
first thing a new user does. They exist because of a real shipped bug: the
dependency list advertised onnxruntime, fastapi and scipy as "optional" in
comments while leaving them as active requirement lines, so
``pip install -r requirements.txt`` -- the exact command in the README -- aborted
on onnxruntime's missing Android wheel before installing anything else.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
SETUP_PY = ROOT / "setup.py"

#: Distributions that are known not to install on the reference Android/aarch64
#: (Termux) host, and therefore must never appear as an unconditional
#: requirement.
NOT_UNIVERSALLY_INSTALLABLE = {
    "onnxruntime",
    "fastapi",
    "scipy",
    "pydantic",
    "torch",
}


def _active_requirements() -> list[str]:
    """Return requirement lines that are not blank or commented out."""
    lines = REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    active = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if line:
            active.append(line)
    return active


def test_requirements_file_exists():
    assert REQUIREMENTS.is_file()


def test_no_uninstallable_package_is_an_active_requirement():
    """The README's install command must work on every supported host."""
    offenders = []
    for spec in _active_requirements():
        name = re.split(r"[<>=!\[;\s]", spec, maxsplit=1)[0].strip().lower()
        if name in NOT_UNIVERSALLY_INSTALLABLE:
            offenders.append(spec)
    assert not offenders, (
        f"these must be commented out or moved to an extra: {offenders}"
    )


def test_numpy_and_uvicorn_are_the_only_hard_runtime_requirements():
    """The architecture promises NumPy (plus an ASGI server) and nothing else."""
    names = {
        re.split(r"[<>=!\[;\s]", spec, maxsplit=1)[0].strip().lower()
        for spec in _active_requirements()
    }
    assert names == {"numpy", "uvicorn"}, names


def test_every_optional_dependency_is_documented():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    for package in ("pybind11", "onnxruntime", "fastapi", "scipy"):
        assert package in text, f"{package} should still be documented"


def test_setup_py_is_valid_and_declares_the_expected_extras():
    source = SETUP_PY.read_text(encoding="utf-8")
    ast.parse(source)  # must be syntactically valid

    # install_requires must stay minimal, matching requirements.txt.
    assert "onnxruntime" not in source.split("extras_require")[0]
    for extra in ("native", "onnx", "fastapi", "dev"):
        assert f'"{extra}"' in source, f"extras_require is missing {extra!r}"


def test_package_layout_matches_the_documented_structure():
    expected = [
        "core/include/csi_packet.hpp",
        "core/include/ring_buffer.hpp",
        "core/include/dsp_filters.hpp",
        "core/src/ring_buffer.cpp",
        "core/src/dsp_filters.cpp",
        "core/bindings.cpp",
        "spectraflow/__init__.py",
        "spectraflow/ingestion/udp_receiver.py",
        "spectraflow/dsp/phase_sanitizer.py",
        "spectraflow/dsp/clutter_removal.py",
        "spectraflow/dsp/vitals_extractor.py",
        "spectraflow/inference/pose_estimator.py",
        "spectraflow/server/app.py",
        "static/index.html",
        "static/css/style.css",
        "static/js/app.js",
        "static/js/scene.js",
        "static/js/avatar.js",
        "static/js/hud.js",
        "tests/test_dsp.py",
        "tests/test_ingestion.py",
        "tests/test_inference.py",
        "CMakeLists.txt",
        "setup.py",
        "requirements.txt",
    ]
    missing = [p for p in expected if not (ROOT / p).is_file()]
    assert not missing, f"missing from the documented layout: {missing}"


def test_firmware_component_is_present():
    firmware = ROOT / "firmware" / "esp32_csi_node"
    assert (firmware / "main" / "csi_collector.c").is_file()
    assert (firmware / "main" / "csi_collector.h").is_file()
    assert (firmware / "CMakeLists.txt").is_file()


@pytest.mark.parametrize(
    "path",
    [
        "spectraflow/ingestion/udp_receiver.py",
        "spectraflow/dsp/vitals_extractor.py",
        "spectraflow/inference/pose_estimator.py",
        "spectraflow/server/app.py",
    ],
)
def test_public_modules_have_no_placeholder_markers(path):
    """No TODOs, stubs or elisions in shipped code paths."""
    text = (ROOT / path).read_text(encoding="utf-8")
    for marker in ("TODO", "FIXME", "XXX", "NotImplementedError"):
        assert marker not in text, f"{path} contains {marker}"
