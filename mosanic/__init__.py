"""
MOSANIC: Multi-mOdal Self-Attention Network for Intercellular Communication

A heterogeneous graph transformer for cell-cell communication inference
from spatial transcriptomics data.

Three node types (cell, gene, metabolite) x seven edge types enable
direct attention-based ligand-receptor scoring without post-hoc correction.

Quick start:
    from mosanic import run_pipeline

    result = run_pipeline(
        "my_tissue.h5ad",
        technology="visium",
        organism="human",
    )

    result.lr_pairs(top_k=20)
    result.plot_spatial("TGFB1", "TGFBR2")
    result.communication_matrix()
    result.knockout_gene("CD163")
"""

__version__ = "1.0.0"
__author__ = "MOSANIC Team"

from .models import MOSANIC, build_model
from .api import MOSANICResult, train, load, run_pipeline, setup, preprocess, train_model

__all__ = ["MOSANIC", "build_model", "MOSANICResult", "train", "load", "run_pipeline",
           "setup", "preprocess", "train_model", "set_verbosity"]

# Quiet-by-default logging — users see only WARNING+ unless they opt in.
# Enable INFO with:  import logging; logging.getLogger("mosanic").setLevel(logging.INFO)
import logging as _logging
_mosanic_logger = _logging.getLogger("mosanic")
_mosanic_logger.setLevel(_logging.WARNING)
_mosanic_logger.propagate = False        # avoid double-printing via root handler
if not _mosanic_logger.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("[mosanic] %(message)s"))
    _mosanic_logger.addHandler(_h)
# Quiet the noisy third-party libs that bypass mosanic's logger
for _name in ("scvi", "lightning", "pytorch_lightning", "anndata"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)


def set_verbosity(level: str = "info") -> None:
    """Adjust mosanic logger verbosity.

    Args:
        level: one of "debug", "info", "warning", "error", "silent".
    """
    levels = {"debug": _logging.DEBUG, "info": _logging.INFO,
              "warning": _logging.WARNING, "error": _logging.ERROR,
              "silent": _logging.CRITICAL + 10}
    _mosanic_logger.setLevel(levels[level.lower()])
