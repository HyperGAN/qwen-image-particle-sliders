"""Record reviewed installed runtime files without model/CUDA allocation."""
from importlib import metadata
import json
from pathlib import Path
from qwen_image_strict.assets import digest
from qwen_image_strict.provenance import VERSIONS, PACKAGES, selected_files

root = Path(__file__).resolve().parents[1] / "src/qwen_image_strict"
result = {}
for name, version in VERSIONS.items():
    dist = metadata.distribution(name)
    assert dist.version.split("+")[0] == version, name
    package = PACKAGES.get(name, name.replace("-", "_"))
    files = {str(f): digest(dist.locate_file(f)) for f in selected_files(dist, package)}
    assert files, name
    result[name] = dict(version=dist.version, sources=files)
(root / "runtime-lock.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print({k: len(v["sources"]) for k, v in result.items()})
