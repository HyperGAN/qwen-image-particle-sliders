"""Run only the four cached-fresh95 cases against the genuine installed wheel.

The preceding CI step builds and installs that wheel. This runner neither
builds nor installs packages and never changes the original cohort guards.
"""

import argparse
import ast
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from importlib.metadata import distribution
from pathlib import Path

NODES = (
    "test_new_cached_fresh_initializer_before_optimizers_hostQR_zeroUP_router_criticKEEP_and_identity",
    "test_new_named_policy_determinism_actual_sqrt5_scale_bank002_and_ambient_RNG",
    "test_new_one_original_update_strict_whole_FULL_resume_and_RAW_native_response_skip_init",
    "test_new_provenance_intent_refusals_before_attach_retryable_cleanup_and_foreign_owner",
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def junit_counts(path):
    if not path.exists():
        return None
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    return {
        "tests": len(cases),
        "failures": sum(len(case.findall("failure")) for case in cases),
        "errors": sum(len(case.findall("error")) for case in cases),
        "skipped": sum(len(case.findall("skipped")) for case in cases),
        "names": [case.attrib["name"] for case in cases],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()
    fixture = arguments.fixture.absolute()
    workspace = fixture.parent.parent
    work = arguments.work.absolute()
    if work.exists() or work.is_relative_to(workspace):
        raise RuntimeError("a new isolated directory outside the source checkout is required")
    source = fixture.read_text(encoding="utf-8")
    declared = tuple(
        node.name
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )
    if declared != NODES:
        raise RuntimeError("exactly the four declared cached-fresh95 cases are required")
    work.mkdir(parents=True)
    reports = work / "reports"
    reports.mkdir()
    copied = work / "test_cached_fresh95.py"
    shutil.copyfile(fixture, copied)
    configuration = work / "pytest.ini"
    configuration.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    os.environ.pop("PYTEST_PLUGINS", None)
    # -I prevents the working directory/PYTHONPATH from taking ownership. Also
    # reject an accidentally retained editable-source path before imports.
    sys.path[:] = [value for value in sys.path if not Path(value).absolute().is_relative_to(workspace)]
    os.chdir(work)
    guard_verified = False
    pytest_exit = None
    junit = reports / "junit.xml"
    try:
        installed = distribution("qwen-image-particle-sliders")
        root = Path(installed.locate_file("qwen_image_sliders")).absolute()
        if root.is_relative_to(workspace):
            raise RuntimeError("editable/source ownership is not installed-wheel validation")
        direct_url = installed.read_text("direct_url.json")
        if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
            raise RuntimeError("the original editable distribution must be replaced by its wheel")

        from qwen_image_sliders.fresh95 import ownership

        # This is the unchanged full core577/PG68 and genuine namespace RECORD
        # guard. A mismatch fails before any added training components.
        ownership.require()
        guard_verified = True
        write_json(
            reports / "ownership.json",
            {
                "distribution": installed.metadata["Name"],
                "version": installed.version,
                "installed_namespace": str(root / "fresh95"),
                "original_cohort_and_installed_RECORD_guard_passed": True,
                "guard_modified": False,
                "source_overlay": False,
            },
        )
        import pytest

        pytest_exit = pytest.main(
            [
                "-q",
                "-c",
                str(configuration),
                "--rootdir=" + str(work),
                "--confcutdir=" + str(work),
                "--noconftest",
                "--import-mode=importlib",
                "-p",
                "no:cacheprovider",
                "--junitxml=" + str(junit),
                *(str(copied) + "::" + node for node in NODES),
            ]
        )
    finally:
        counts = junit_counts(junit)
        complete = (
            guard_verified
            and pytest_exit == 0
            and counts
            == {
                "tests": 4,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "names": list(NODES),
            }
        )
        write_json(
            reports / "result.json",
            {
                "schema": "Qwen-cached-fresh95-installed-wheel-FOUR-CI-software-result-v1",
                "complete": bool(complete),
                "original_guards_passed": guard_verified,
                "actual_pytest_exit": pytest_exit,
                "actual_counts": counts,
                "selected_nodes": list(NODES),
                "new_test_file_count": 1,
                "expected_tiny_update_count_if_complete": 1,
                "pretrained_native_or_LR_qualification": False,
                "automatic_retry": False,
            },
        )
    if not complete:
        raise RuntimeError("the installed-wheel FOUR-case software check did not complete")


if __name__ == "__main__":
    main()
