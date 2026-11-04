"""Actual canonical selected files, irrespective of local snapshot folder spelling."""
import hashlib
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def inspect_assets(snapshot,*,full_hashes):
    from qwen_image_sliders.contracts import lock
    from qwen_image_sliders.runtime import verify_snapshot
    root=Path(snapshot)
    if not root.is_dir():raise ValueError('explicit local canonical Qwen snapshot required')
    model=lock()['model']
    for name,spec in model['files'].items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=spec['bytes']:
            raise ValueError('Qwen selected asset missing/size differs: '+name)
    if full_hashes:verify_snapshot(root)
    return dict(repository=model['repository'],canonical_manifest_revision=model['revision'],
                selected_files=model['files'],full_weight_hashes_verified=full_hashes)
