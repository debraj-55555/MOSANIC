"""
MOSAIC: Multi-mOdal Spatial Attention for Intercellular Communication

A heterogeneous graph transformer for cell-cell communication inference
from spatial transcriptomics data.

Three node types (cell, gene, metabolite) x seven edge types enable
direct attention-based ligand-receptor scoring without post-hoc correction.

Quick start:
    from mosaic import run_pipeline

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
__author__ = "MOSAIC Team"

from .models import MOSAIC, build_model
from .api import MOSAICResult, train, load, run_pipeline, setup, preprocess, train_model

__all__ = ["MOSAIC", "build_model", "MOSAICResult", "train", "load", "run_pipeline",
           "setup", "preprocess", "train_model", "set_verbosity"]

# Quiet-by-default logging — users see only WARNING+ unless they opt in.
# Enable INFO with:  import logging; logging.getLogger("mosaic").setLevel(logging.INFO)
import logging as _logging
_mosaic_logger = _logging.getLogger("mosaic")
_mosaic_logger.setLevel(_logging.WARNING)
_mosaic_logger.propagate = False        # avoid double-printing via root handler
if not _mosaic_logger.handlers:
    _h = _logging.StreamHandler()
    _h.setFormatter(_logging.Formatter("[mosaic] %(message)s"))
    _mosaic_logger.addHandler(_h)
# Quiet the noisy third-party libs that bypass mosaic's logger
for _name in ("scvi", "lightning", "pytorch_lightning", "anndata"):
    _logging.getLogger(_name).setLevel(_logging.WARNING)


def set_verbosity(level: str = "info") -> None:
    """Adjust mosaic logger verbosity.

    Args:
        level: one of "debug", "info", "warning", "error", "silent".
    """
    levels = {"debug": _logging.DEBUG, "info": _logging.INFO,
              "warning": _logging.WARNING, "error": _logging.ERROR,
              "silent": _logging.CRITICAL + 10}
    _mosaic_logger.setLevel(levels[level.lower()])
