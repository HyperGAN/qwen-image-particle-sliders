"""Refresh canonical locks from the exact installed public core/PG release.

Historical cohort locks and the isolated HyperGAN/HNDL candidate are immutable.
"""

import argparse
import hashlib
import importlib
import json
from importlib import metadata
from pathlib import Path


def closure(package):
    root = Path(importlib.import_module(package).__file__).resolve().parent
    return {
        path.relative_to(root.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-commit", required=True)
    args = parser.parse_args()
    core = metadata.distribution("particle-sliders-core")
    direct = json.loads(core.read_text("direct_url.json") or "{}")
    if core.version != "0.12.0" or direct.get("vcs_info", {}).get("commit_id") != args.core_commit:
        raise ValueError("install the exact reviewed public VCS core0.12.0 commit first")
    if metadata.version("particlegan") != "0.9.4":
        raise ValueError("install released ParticleGAN0.9.4 first")
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / "backend.lock.json").read_text())
    for key in ("wheel_sha256", "sdist_sha256", "canonical_receipt_sha256"):
        document["core"].pop(key, None)
    document["core"].update(
        commit=args.core_commit,
        version=core.version,
        source_sha256=closure("particle_sliders") | closure("concept_slider_core"),
        source_map_scope="all installed non-cache core package source/data files",
        qualification="canonical native PG094 source cohort; CPU fixture qualification only",
    )
    release = json.loads(Path(core.locate_file("particle_sliders/pg094_release.json")).read_text())
    document["particlegan"].update(
        commit=release["release_commit"],
        version=release["version"],
        wheel_sha256=release["wheel_sha256"],
        source_profile=release["source_profile"],
        source_sha256=closure("particlegan"),
        role="exact released core dependency; native spatial image domain, no source overlay",
    )
    document["historical_environment"] = "docs/cohorts/qwen-core081.lock.json; original product eaa753d"
    text = json.dumps(document, indent=2) + "\n"
    for path in (root / "backend.lock.json", root / "src/qwen_image_sliders/backend.lock.json"):
        path.write_text(text)


if __name__ == "__main__":
    main()
