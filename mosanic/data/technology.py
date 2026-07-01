"""
technology.py — Spatial Technology Detector and Bias Corrector

Detects the spatial transcriptomics technology from AnnData metadata and
produces a tech_profile.json that drives all downstream bias corrections.

Technology differences that affect CCC inference:
  ┌─────────────┬────────────┬────────────┬──────────────────────────────────────┐
  │ Technology  │ Resolution │ Cells/spot │ CCC Bias                             │
  ├─────────────┼────────────┼────────────┼──────────────────────────────────────┤
  │ Visium      │ 55 µm spot │ ~10 cells  │ Juxtacrine over-estimated; spot mix  │
  │ Visium HD   │ 2 µm       │ ~1 cell    │ Near cell-level; minimal bias        │
  │ MERFISH     │ Cell-level │ 1          │ Only panel genes captured            │
  │ Slide-seq   │ ~10 µm     │ ~1-2       │ Lower depth → more dropout           │
  │ Stereo-seq  │ 0.22 µm    │ Sub-cell   │ Aggregation to bins needed           │
  │ CosMx       │ Cell-level │ 1          │ Panel genes only (like MERFISH)      │
  └─────────────┴────────────┴────────────┴──────────────────────────────────────┘

Usage:
    python -m mosanic.data.technology \
        --adata data/processed/breast_new/processed_adata.h5ad \
        --output data/processed/breast_new/tech_profile.json
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technology profile dataclass
# ---------------------------------------------------------------------------
@dataclass
class TechProfile:
    """Complete description of a spatial technology's properties."""

    technology: str                      # canonical name
    resolution_type: str                 # 'spot' | 'cell' | 'subcell'
    spot_diameter_um: float              # physical spot/cell diameter in µm
    center_to_center_um: float          # typical neighbour spacing in µm
    cells_per_spot_estimate: int         # avg number of cells per spot
    requires_deconvolution: bool         # True for Visium (multi-cell spots)
    is_targeted_panel: bool              # True for MERFISH/CosMx (limited gene set)
    recommended_knn_k: int               # k for spatial graph
    recommended_max_dist_um: float       # max edge distance in µm
    recommended_graph_method: str        # 'knn' or 'radius'
    scale_factor_um_per_pixel: float     # pixel→µm conversion (from scalefactors)
    coordinate_unit: str                 # 'pixel' | 'um' | 'mm'
    notes: str = ''


# ---------------------------------------------------------------------------
# Known technology profiles
# ---------------------------------------------------------------------------
KNOWN_PROFILES: Dict[str, TechProfile] = {
    'visium': TechProfile(
        technology='visium',
        resolution_type='spot',
        spot_diameter_um=55.0,
        center_to_center_um=100.0,
        cells_per_spot_estimate=10,
        requires_deconvolution=True,
        is_targeted_panel=False,
        recommended_knn_k=6,
        recommended_max_dist_um=200.0,
        recommended_graph_method='knn',
        scale_factor_um_per_pixel=0.4,
        coordinate_unit='pixel',
        notes=(
            "Spots are 55 µm diameter on a hexagonal grid with 100 µm pitch. "
            "Each spot contains ~10 cells from potentially multiple cell types. "
            "Deconvolution (RCTD) strongly recommended before CCC inference. "
            "Juxtacrine threshold should be ≥100 µm (1 spot centre distance) "
            "to avoid artificial self-signal."
        ),
    ),
    'visium_hd': TechProfile(
        technology='visium_hd',
        resolution_type='cell',
        spot_diameter_um=2.0,
        center_to_center_um=2.0,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=False,
        recommended_knn_k=8,
        recommended_max_dist_um=20.0,
        recommended_graph_method='knn',
        scale_factor_um_per_pixel=0.2,
        coordinate_unit='pixel',
        notes="Near cell-level resolution. Minimal multi-cell spot bias.",
    ),
    'merfish': TechProfile(
        technology='merfish',
        resolution_type='cell',
        spot_diameter_um=10.0,
        center_to_center_um=15.0,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=True,
        recommended_knn_k=8,
        recommended_max_dist_um=50.0,
        recommended_graph_method='radius',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes=(
            "Cell-level resolution. Only genes in the imaging panel are measured. "
            "L-R database must be filtered to panel genes before CCC inference. "
            "Use radius-based graph (cells not on regular grid)."
        ),
    ),
    'slide_seq': TechProfile(
        technology='slide_seq',
        resolution_type='cell',
        spot_diameter_um=10.0,
        center_to_center_um=10.0,
        cells_per_spot_estimate=2,
        requires_deconvolution=False,
        is_targeted_panel=False,
        recommended_knn_k=8,
        recommended_max_dist_um=30.0,
        recommended_graph_method='knn',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes=(
            "~10 µm resolution. Full transcriptome but lower depth than Visium. "
            "Higher technical dropout — consider kNN expression smoothing before "
            "L-R expression thresholding."
        ),
    ),
    'slide_seqv2': TechProfile(  # alias
        technology='slide_seqv2',
        resolution_type='cell',
        spot_diameter_um=10.0,
        center_to_center_um=10.0,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=False,
        recommended_knn_k=8,
        recommended_max_dist_um=30.0,
        recommended_graph_method='knn',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes="Improved Slide-seq. Cell-level resolution.",
    ),
    'stereo_seq': TechProfile(
        technology='stereo_seq',
        resolution_type='subcell',
        spot_diameter_um=0.22,
        center_to_center_um=0.22,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=False,
        recommended_knn_k=6,
        recommended_max_dist_um=10.0,
        recommended_graph_method='knn',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes=(
            "Sub-cellular resolution. Raw bins should be aggregated to cell-level "
            "before CCC inference. Use bin_size ≈ 50 µm for Visium-comparable bins."
        ),
    ),
    'cosmx': TechProfile(
        technology='cosmx',
        resolution_type='cell',
        spot_diameter_um=10.0,
        center_to_center_um=15.0,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=True,
        recommended_knn_k=8,
        recommended_max_dist_um=40.0,
        recommended_graph_method='radius',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes=(
            "Cell-level resolution. Targeted panel (similar to MERFISH). "
            "Filter L-R database to panel genes."
        ),
    ),
    'seqfish': TechProfile(
        technology='seqfish',
        resolution_type='cell',
        spot_diameter_um=8.0,
        center_to_center_um=10.0,
        cells_per_spot_estimate=1,
        requires_deconvolution=False,
        is_targeted_panel=True,
        recommended_knn_k=8,
        recommended_max_dist_um=40.0,
        recommended_graph_method='radius',
        scale_factor_um_per_pixel=1.0,
        coordinate_unit='um',
        notes="Cell-level. Targeted panel.",
    ),
}

# Aliases
KNOWN_PROFILES['10x_visium'] = KNOWN_PROFILES['visium']
KNOWN_PROFILES['10x_visium_hd'] = KNOWN_PROFILES['visium_hd']
KNOWN_PROFILES['seqfish_plus'] = KNOWN_PROFILES['seqfish']


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------
def detect_technology(adata) -> TechProfile:
    """
    Detect spatial technology from AnnData metadata.

    Checks (in order of confidence):
      1. adata.uns['technology'] if explicitly set
      2. adata.uns['spatial'] → library_id → chemistry_description
      3. adata.uns['spatial'] → scalefactors → spot_diameter_fullres presence
      4. Presence of 'visium' in obsm keys
      5. Default: Visium (most common)

    Args:
        adata: AnnData object (scanpy)

    Returns:
        TechProfile with all technology parameters
    """
    uns = adata.uns if hasattr(adata, 'uns') else {}

    # --- Check 1: Explicit technology key ---
    if 'technology' in uns:
        tech_name = uns['technology'].lower().replace(' ', '_').replace('-', '_')
        if tech_name in KNOWN_PROFILES:
            logger.info(f"Detected technology from uns['technology']: {tech_name}")
            return KNOWN_PROFILES[tech_name]

    # --- Check 2: Spatial library metadata ---
    if 'spatial' in uns:
        spatial_meta = uns['spatial']
        for lib_id, lib_data in spatial_meta.items():
            if isinstance(lib_data, dict):
                # Check chemistry description
                images = lib_data.get('images', {})
                scalef = lib_data.get('scalefactors', {})

                if 'spot_diameter_fullres' in scalef:
                    # This is 10x Visium format
                    spot_diam_px = scalef.get('spot_diameter_fullres', 130)
                    scale_um_px  = scalef.get('tissue_hires_scalef', 0.4)

                    # Visium HD has much smaller spot diameter
                    if spot_diam_px < 20:
                        tech_name = 'visium_hd'
                    else:
                        tech_name = 'visium'

                    profile = KNOWN_PROFILES[tech_name]

                    # Override scale factor from actual metadata
                    actual_um_per_px = 55.0 / spot_diam_px  # spot is always 55µm
                    profile = TechProfile(
                        **{**asdict(profile), 'scale_factor_um_per_pixel': actual_um_per_px}
                    )
                    logger.info(f"Detected technology from scalefactors: {tech_name} "
                                f"(scale={actual_um_per_px:.4f} µm/px)")
                    return profile

    # --- Check 3: obsm keys ---
    obsm = adata.obsm if hasattr(adata, 'obsm') else {}
    obsm_keys = ' '.join(obsm.keys()).lower() if obsm else ''
    if 'spatial' in obsm_keys:
        logger.info("Detected Visium-style spatial coordinates in obsm")
        return KNOWN_PROFILES['visium']

    # --- Check 4: var metadata hints ---
    if hasattr(adata, 'var') and 'gene_ids' in adata.var.columns:
        logger.info("Assuming Visium (gene_ids column found)")
        return KNOWN_PROFILES['visium']

    # --- Default ---
    logger.warning(
        "Could not detect technology automatically. Defaulting to Visium. "
        "Set adata.uns['technology'] = '<tech_name>' to specify explicitly."
    )
    return KNOWN_PROFILES['visium']


def get_scale_factor(adata, tech_profile: TechProfile) -> float:
    """
    Extract pixel→µm scale factor from AnnData or tech profile.

    Visium spatial coordinates are in pixel units; we need µm.
    """
    uns = adata.uns if hasattr(adata, 'uns') else {}
    if 'spatial' in uns:
        for lib_id, lib_data in uns['spatial'].items():
            if isinstance(lib_data, dict):
                scalef = lib_data.get('scalefactors', {})
                spot_diam_px = scalef.get('spot_diameter_fullres', None)
                if spot_diam_px and spot_diam_px > 0:
                    return 55.0 / spot_diam_px   # spot is always 55 µm
    return tech_profile.scale_factor_um_per_pixel


def generate_tech_profile(adata_path: str, output_path: str):
    """
    Detect technology from AnnData and save tech_profile.json.

    Args:
        adata_path: Path to processed_adata.h5ad
        output_path: Path to write tech_profile.json
    """
    import scanpy as sc
    logger.info(f"Loading AnnData: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    logger.info(f"  Shape: {adata.shape}")

    profile = detect_technology(adata)
    scale   = get_scale_factor(adata, profile)

    profile_dict = asdict(profile)
    profile_dict['scale_factor_um_per_pixel'] = scale

    # Add dataset-specific summaries
    profile_dict['n_cells'] = int(adata.n_obs)
    profile_dict['n_genes'] = int(adata.n_vars)
    profile_dict['coordinate_source'] = 'adata.obsm["spatial"]' if 'spatial' in adata.obsm else 'unknown'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(profile_dict, f, indent=2)

    logger.info(f"Tech profile saved: {output_path}")
    logger.info(f"  Technology:   {profile.technology}")
    logger.info(f"  Resolution:   {profile.resolution_type}")
    logger.info(f"  Deconvolution needed: {profile.requires_deconvolution}")
    logger.info(f"  Targeted panel:       {profile.is_targeted_panel}")
    logger.info(f"  Recommended k-NN k:   {profile.recommended_knn_k}")
    logger.info(f"  Max distance:         {profile.recommended_max_dist_um} µm")

    if profile.requires_deconvolution:
        logger.warning(
            f"\n{'='*60}\n"
            f"DECONVOLUTION RECOMMENDED:\n"
            f"  Technology '{profile.technology}' has ~{profile.cells_per_spot_estimate} "
            f"cells per spot.\n"
            f"  Run RCTD (R package 'spacexr') before CCC inference.\n"
            f"  Output: spot_cell_type_proportions.csv\n"
            f"{'='*60}"
        )

    if profile.is_targeted_panel:
        logger.warning(
            f"\n{'='*60}\n"
            f"GENE PANEL FILTER REQUIRED:\n"
            f"  Technology '{profile.technology}' only measures a targeted gene panel.\n"
            f"  L-R database must be filtered to panel genes.\n"
            f"  Run lr_database.py with --technology {profile.technology}\n"
            f"{'='*60}"
        )

    return profile_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Detect spatial technology and generate tech_profile.json'
    )
    parser.add_argument('--adata',  required=True, help='Path to processed_adata.h5ad')
    parser.add_argument('--output', required=True, help='Output path for tech_profile.json')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    generate_tech_profile(adata_path=args.adata, output_path=args.output)


if __name__ == '__main__':
    main()
