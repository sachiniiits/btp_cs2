"""cs2btp: Player Role Consistency and Round Outcomes in CS2.

Pipeline (run the notebooks in order):
  00 setup & smoke test -> 01 parse -> 02 features -> 03 roles (SOM)
  -> 04 consistency -> 05 outcome -> 06 robustness
"""
from . import config, manifest, parsing, qc, features, roles, consistency, outcome, viz  # noqa

__version__ = "0.1.0"
