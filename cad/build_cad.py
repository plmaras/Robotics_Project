#!/usr/bin/env python3
"""Compatibility entry point for the unified MuJoCo scene builder.

The CAD mesh exports and manifest are maintained in ``cad/out``. This legacy
command name now delegates to ``build_mjcf.py`` so there is one source of truth
for the generated MuJoCo scene and frame metadata.
"""

from build_mjcf import main


if __name__ == "__main__":
    main()
