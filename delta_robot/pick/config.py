from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_runtime_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    root = path.parent
    for key in ("cad_config", "scene_xml", "output_scene_xml"):
        cfg[key] = (root / cfg[key]).resolve()
    cfg["simulation"]["output_dir"] = (root / cfg["simulation"]["output_dir"]).resolve()
    return cfg, root
