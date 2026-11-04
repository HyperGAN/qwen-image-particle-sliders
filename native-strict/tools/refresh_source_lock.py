"""Refresh explicit candidate closure after reviewing source changes."""
import json
from pathlib import Path
import qwen_image_sliders
from qwen_image_strict.provenance import index
root=Path(__file__).resolve().parents[1]/'src/qwen_image_strict'
own=index(root);own.pop('source-lock.json',None)
packet=dict(base_sources=index(Path(qwen_image_sliders.__file__).parent),product_sources=own)
(root/'source-lock.json').write_text(json.dumps(packet,sort_keys=True,indent=2)+'\n')
print({key:len(value) for key,value in packet.items()})
