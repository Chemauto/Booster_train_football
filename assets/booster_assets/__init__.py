"""Vendored subset of the external ``booster_assets`` package.

Repo-root ``assets/`` plays the role of the upstream ``booster_assets`` checkout::

    assets/                 <- BOOSTER_ASSETS_DIR
      robots/K1/            # K1 URDF + meshes (training + sim2sim)
      motions/K1/amp_paper_waistfix/   # 12 walk + 12 kick AMP clips
      scene/                # soccer field, ball, goal meshes (sim2sim)
      booster_assets/       # this package (import shim)

Prefer this over an installed ``booster_assets``: add this directory to
``sys.path`` (``booster_train`` does it on import) or set
``PYTHONPATH=/path/to/repo/assets``.
"""

import pathlib

BOOSTER_ASSETS_DIR = str(pathlib.Path(__file__).resolve().parents[1])

__all__ = ["BOOSTER_ASSETS_DIR", "motions"]
