"""
mosaic/data/preprocessor.py — Unified preprocessing entry point.

Orchestrates all steps from raw data → hetero_ccc_graph.pt.

Cache strategy:
  - Two checkpoint files in processed_dir:
      preprocessing_cache.pt   all intermediate arrays (coords, edges, labels …)
      hetero_ccc_graph.pt      final PyG HeteroData (training input)
  - Each step checks if its key already exists in preprocessing_cache.pt
    before computing. Pass --force to invalidate and recompute everything.

Usage:
    python -m mosaic.data.preprocessor --config mosaic/configs/breast_config.yaml \\
                               --dataset breast_new [--force]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, dataset: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"] = dataset

    # Resolve processed_dir: absolute paths used as-is, relative resolved from root.parent
    processed_path = Path(cfg["paths"]["processed_dir"])
    if processed_path.is_absolute():
        cfg["_processed_dir"] = processed_path / dataset
    else:
        root = Path(cfg["paths"]["root"])
        cfg["_processed_dir"] = root.parent / cfg["paths"]["processed_dir"] / dataset
    cfg["_processed_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(processed_dir: Path) -> dict:
    cache_path = processed_dir / "preprocessing_cache.pt"
    if cache_path.exists():
        log.debug("Loading preprocessing cache …")
        return torch.load(cache_path, weights_only=False)
    return {}


def save_cache(cache: dict, processed_dir: Path):
    cache_path = processed_dir / "preprocessing_cache.pt"
    torch.save(cache, cache_path)


def cached(key, cache, fn, *args, force=False, **kwargs):
    """Run fn(*args, **kwargs) and store result under key in cache, or return cached value."""
    if not force and key in cache:
        log.debug(f"  cache hit: {key}")
        return cache[key]
    t0 = time.time()
    result = fn(*args, **kwargs)
    cache[key] = result
    log.debug(f"  computed:  {key}  ({time.time()-t0:.1f}s)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def _step1_load_anndata(cfg):
    """Load AnnData → cell features (scVI), spatial coords, expression, gene names."""
    import scanpy as sc

    root = Path(cfg["paths"]["root"]).parent
    adata_path = root / cfg["paths"]["raw_adata"]
    # Log the user-given relative path, not the resolved absolute path
    log.info(f"  Loading AnnData from {cfg['paths']['raw_adata']}")
    adata = sc.read_h5ad(adata_path)

    # scVI latent embeddings (128-dim)
    if "X_scvi" in adata.obsm:
        cell_features = adata.obsm["X_scvi"].astype(np.float32)   # [N, 128]
    else:
        log.info("  X_scvi not found in adata.obsm — training scVI (128-dim, 200 epochs)...")
        try:
            import scvi as _scvi
            adata_copy = adata.copy()
            _scvi.model.SCVI.setup_anndata(adata_copy, layer="raw_count")
            vae = _scvi.model.SCVI(adata_copy, n_latent=128)
            vae.train(max_epochs=200, early_stopping=True)
            cell_features = vae.get_latent_representation().astype(np.float32)
            adata.obsm["X_scvi"] = cell_features
            log.info("  scVI trained: %s", cell_features.shape)
        except ImportError:
            raise ImportError(
                "scVI embeddings not found in adata.obsm['X_scvi'] and scvi-tools "
                "not installed. Either pre-compute scVI or: pip install scvi-tools"
            )

    # Spatial pixel coordinates
    coords_px = adata.obsm["spatial"].astype(np.float32)       # [N, 2]

    # Expression matrix (log-normalized — for cell-gene edges)
    import scipy.sparse as sp
    X = adata.X
    expr_matrix = X.toarray().astype(np.float32) if sp.issparse(X) else np.array(X, dtype=np.float32)

    # Raw count matrix (for expression labels)
    raw = adata.layers["raw_count"]
    expr_matrix_raw = raw  # keep as sparse; label generators handle it

    # Gene names
    gene_names = list(adata.var_names)

    # Cell type labels (try annotation columns → leiden → fallback to zeros)
    # Users can specify the column name via config: labels.cell_type_col
    cell_type_col = cfg.get("labels", {}).get("cell_type_col")
    if cell_type_col is not None and cell_type_col not in adata.obs.columns:
        log.warning(f"  Configured cell_type_col='{cell_type_col}' not in adata.obs, auto-detecting...")
        cell_type_col = None

    if cell_type_col is None:
        # Auto-detect: prefer annotation over clusters
        for col in ["cell_type", "cell_type_annot", "celltype", "CellType",
                    "annotation", "cluster", "leiden"]:
            if col in adata.obs.columns:
                cell_type_col = col
                break

    if cell_type_col is not None:
        ct_raw = adata.obs[cell_type_col].values
        uniq, inv = np.unique(ct_raw, return_inverse=True)
        cell_types = inv.astype(np.int64)
        cell_type_names = [str(u) for u in uniq]  # sorted unique names
    else:
        cell_types = np.zeros(adata.n_obs, dtype=np.int64)
        cell_type_names = ["unknown"]

    log.info(f"  Loaded: {adata.n_obs} cells, {len(gene_names)} genes, "
             f"scVI dim={cell_features.shape[1]}, {len(cell_type_names)} cell types "
             f"(from '{cell_type_col or 'none'}')")

    return {
        "cell_features":    cell_features,    # [N, 128]
        "coords_px":        coords_px,        # [N, 2]
        "expr_matrix":      expr_matrix,      # [N, G] log-normalized
        "expr_matrix_raw":  expr_matrix_raw,  # [N, G] raw counts (sparse)
        "gene_names":       gene_names,       # list[str], length G
        "cell_types":       cell_types,       # [N] int64
        "cell_type_names":  cell_type_names,  # list[str], sorted unique
    }


def _step2_spatial_coords(step1, cfg):
    """Convert pixel coords → µm, build k-NN spatial graph."""
    from sklearn.neighbors import NearestNeighbors

    sp = cfg.get("spatial", {})
    um_per_pixel = float(sp.get("um_per_pixel", 0.2917))
    k            = int(sp.get("k_neighbors", 6))
    max_dist_um  = float(sp.get("max_distance_um", 150.0))

    coords_um = step1["coords_px"] * um_per_pixel   # [N, 2] float32

    # k-NN (k+1 because sklearn includes self)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="euclidean")
    nbrs.fit(coords_um)
    distances, indices = nbrs.kneighbors(coords_um)  # [N, k+1]
    distances = distances[:, 1:]   # [N, k] — drop self
    indices   = indices[:, 1:]     # [N, k]

    n_cells = coords_um.shape[0]
    src_list, dst_list, dist_list = [], [], []
    for i in range(n_cells):
        for j in range(k):
            d = float(distances[i, j])
            if d <= max_dist_um:
                src_list.append(i)
                dst_list.append(int(indices[i, j]))
                dist_list.append(d)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    dist_arr   = np.array(dist_list, dtype=np.float32)

    log.info(f"  Spatial graph: {n_cells} cells, {edge_index.shape[1]} edges "
             f"(k={k}, max_dist={max_dist_um}µm, scale={um_per_pixel}µm/px)")
    return {
        "coords_um":  coords_um,     # [N, 2] float32
        "edge_index": edge_index,    # [2, E] long
        "dist_um":    dist_arr,      # [E] float32
    }


def _step3_lr_database(cfg):
    """Load and clean LR database (contact + secreted channels) — SEPARATE from metabolites."""
    import pandas as pd
    from mosaic.data.lr_database import DatabaseLoader
    from mosaic.data.channel_classifier import classify_dataframe

    root = Path(cfg["paths"]["root"]).parent
    lr_path = root / cfg["paths"]["lr_database"]

    loader = DatabaseLoader(source_db_name=cfg.get("lr_database", {}).get("source_name", "CellNEST"))
    lr_df = loader.load_lr_database(str(lr_path))

    if "channel_type" not in lr_df.columns:
        ann_col = "annotation" if "annotation" in lr_df.columns else None
        lr_df["channel_type"] = classify_dataframe(
            lr_df, ligand_col="ligand", receptor_col="receptor", annotation_col=ann_col,
        )

    # Symmetric dedup
    if cfg.get("lr_database", {}).get("symmetric_dedup", True):
        pairs, keep = set(), []
        for _, row in lr_df.iterrows():
            p = tuple(sorted([str(row["ligand"]), str(row["receptor"])]))
            keep.append(p not in pairs)
            pairs.add(p)
        lr_df = lr_df[keep].reset_index(drop=True)

    log.info(f"  LR pairs: {len(lr_df)} | {lr_df['channel_type'].value_counts().to_dict()}")
    return {"lr_df": lr_df}


def _run_scfea(cfg, adata_path, scfea_balance_path):
    """Run scFEA from raw AnnData to produce a balance CSV.

    Follows src5/preprocessing/scfea_runner.py pattern:
      1. Export expression matrix as genes×cells CSV (scFEA input format).
      2. Call scFEA_gpu_safe.py via subprocess (isolated process, no import side-effects).
      3. Copy balance.csv to expected path.

    Paths: all I/O goes under MOSAIC/data/processed/<dataset>/scfea/.
    Subprocess receives resolved absolute paths; stored paths stay relative to project.
    """
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import subprocess
    import shutil
    import sys

    organism = cfg.get("organism", "human")

    # Locate scFEA directory (shipped with MOSAIC)
    scfea_dir    = Path(__file__).resolve().parent.parent / "external" / "scfea"
    scfea_data   = scfea_dir / "data"
    scfea_script = scfea_dir / "src" / "scFEA_gpu_safe.py"

    if not scfea_script.exists():
        raise FileNotFoundError(f"scFEA script not found: {scfea_script}")

    # Choose organism-specific files
    if organism == "mouse":
        module_gene_file = "module_gene_complete_mouse_m168.csv"
        cm_file          = "cmMat_complete_mouse_c70_m168.csv"
        cname_file       = "cName_complete_mouse_c70_m168.csv"
    else:
        module_gene_file = "module_gene_m168.csv"
        cm_file          = "cmMat_c70_m168.csv"
        cname_file       = "cName_c70_m168.csv"

    # All scFEA I/O under processed_dir/scfea/ (relative to MOSAIC root)
    processed_dir   = cfg["_processed_dir"]
    scfea_work_dir  = Path(processed_dir).resolve() / "scfea"
    scfea_input_dir = scfea_work_dir / "input"
    scfea_res_dir   = scfea_work_dir / "output"
    scfea_input_dir.mkdir(parents=True, exist_ok=True)
    scfea_res_dir.mkdir(parents=True, exist_ok=True)

    # Resolve adata_path for reading
    adata_abs = Path(adata_path).resolve()

    # Step 1: Export expression as genes×cells CSV
    # Gene names must match the module-gene file case (mouse=Title, human=Title/UPPER)
    log.info("  scFEA: Loading AnnData for expression export...")
    adata = sc.read_h5ad(str(adata_abs))
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Read module-gene file to determine expected gene name case
    mg_path = scfea_data / module_gene_file
    if mg_path.exists():
        mg_df = pd.read_csv(mg_path)
        module_genes = set()
        for col in mg_df.columns[1:]:
            module_genes.update(mg_df[col].dropna().str.strip().tolist())
        # Build case mapping: adata_upper → module_case
        module_case = {g.upper(): g for g in module_genes}
        # Remap adata gene names to match module file case
        remapped = [module_case.get(g.upper(), g) for g in adata.var_names]
    else:
        remapped = list(adata.var_names)

    expr_df = pd.DataFrame(X, index=adata.obs_names, columns=remapped)
    input_csv = scfea_input_dir / "scFEA_input.csv"
    expr_df.T.to_csv(input_csv)  # genes×cells (scFEA expects this orientation)
    log.info("  scFEA: Exported expression (%d cells × %d genes)",
             expr_df.shape[0], expr_df.shape[1])

    # Step 2: Run scFEA via subprocess (all paths absolute for subprocess)
    n_epochs    = cfg.get("scfea", {}).get("epochs", 100)
    balance_out = scfea_res_dir / "balance.csv"
    flux_out    = scfea_res_dir / "flux.csv"

    cmd = [
        sys.executable, str(scfea_script),
        "--data_dir",             str(scfea_data),
        "--input_dir",            str(scfea_input_dir),
        "--res_dir",              str(scfea_res_dir),
        "--test_file",            "scFEA_input.csv",
        "--moduleGene_file",      module_gene_file,
        "--stoichiometry_matrix", cm_file,
        "--cName_file",           cname_file,
        "--sc_imputation",        str(cfg.get("scfea", {}).get("magic_imputation", False)),
        "--output_flux_file",     str(flux_out),
        "--output_balance_file",  str(balance_out),
        "--train_epoch",          str(n_epochs),
    ]
    log.info("  scFEA: Running (epochs=%d, cells=%d, genes=%d)...",
             n_epochs, expr_df.shape[0], expr_df.shape[1])

    # Stream scFEA stdout/stderr directly to the user — the progress bars +
    # per-epoch loss are the most informative thing to see while preprocessing.
    # No capture; user sees the same prints scFEA emits natively.
    result = subprocess.run(cmd, text=True)
    # Below kept for the (rare) error-path branches; scFEA output is already
    # on the user's terminal.
    if False and result.stderr:
        for line in result.stderr.strip().split("\n"):
            line_s = line.strip()
            if not line_s or "FutureWarning" in line_s or "deprecated" in line_s.lower():
                continue
            # Keep progress bars and real errors
            if "scFEA:" in line_s or "%" in line_s:
                log.info("  [scFEA] %s", line_s)
            elif "Error" in line_s or "Traceback" in line_s:
                log.error("  [scFEA] %s", line_s)
    if result.returncode != 0:
        raise RuntimeError(f"scFEA failed with exit code {result.returncode}")
    log.info("  scFEA: Completed successfully")

    # Step 3: Copy balance to the config-expected path
    if not balance_out.exists():
        raise FileNotFoundError(f"scFEA did not produce balance file: {balance_out}")
    Path(scfea_balance_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(balance_out, scfea_balance_path)
    log.info("  scFEA: Balance saved → %s", scfea_balance_path)
    return scfea_balance_path


def _step4_metabolite_database(cfg):
    """Load scFEA flux + build module→receptor map from M_R.txt — SEPARATE from LR pairs."""
    import pandas as pd

    def _resolve(p_str):
        """Resolve path: absolute used as-is, relative from root.parent."""
        p = Path(p_str) if p_str else Path("")
        if p.is_absolute():
            return p
        return Path(cfg["paths"]["root"]).parent / p

    mr_path = _resolve(cfg["paths"].get("met_database", ""))
    scfea_bal = cfg["paths"].get("scfea_balance", "")
    if scfea_bal:
        scfea_path = _resolve(scfea_bal)
    else:
        # Default: store in processed dir alongside graph
        scfea_path = cfg["_processed_dir"] / "scfea_balance.csv"

    # Skip metabolites if flag set or no scFEA balance and skip_rerun
    skip_met = cfg.get("scfea", {}).get("skip_metabolites", False)
    skip_rerun = cfg.get("scfea", {}).get("skip_rerun", False)
    no_balance = scfea_path is None or not scfea_path.exists()
    no_mr_db = not mr_path.exists() or not str(cfg["paths"].get("met_database", ""))

    if skip_met or no_mr_db or (no_balance and skip_rerun):
        reason = "no M_R database" if no_mr_db else ("no scFEA balance + skip_rerun" if no_balance else "config flag")
        log.debug("  Metabolite step SKIPPED (%s)", reason)
        return {
            "flux_matrix":         np.zeros((0, 0), dtype=np.float32),
            "module_names":        [],
            "module_receptor_map": {},
        }

    # Auto-run scFEA if balance file doesn't exist
    if scfea_path is not None and not scfea_path.exists():
        adata_path = _resolve(cfg["paths"]["raw_adata"])
        log.info("  scFEA balance not found — running scFEA (%d epochs)...",
                 cfg.get("scfea", {}).get("epochs", 100))
        _run_scfea(cfg, adata_path, scfea_path)

    balance_df   = pd.read_csv(scfea_path, index_col=0)
    flux_matrix  = balance_df.values.astype(np.float32)
    module_names = list(balance_df.columns)

    # Build module→receptor map — handle both human M_R.txt and mouse M_R_mouse.txt formats
    # Human format: "Metabolite name" (col 2), "Receptor symbol" (col 11), 15 columns
    # Mouse format: "standard_metName" (col 2), "Gene_name" (col 4), 8 columns
    mr_df   = pd.read_csv(str(mr_path), sep="\t")
    cols    = mr_df.columns.tolist()
    if "Metabolite name" in cols:
        met_col = "Metabolite name"
    elif "standard_metName" in cols:
        met_col = "standard_metName"
    else:
        met_col = cols[2]
    if "Receptor symbol" in cols:
        rec_col = "Receptor symbol"
    elif "Gene_name" in cols:
        rec_col = "Gene_name"
    else:
        rec_col = cols[4]
    log.debug(f"  M_R format detected: met_col='{met_col}', rec_col='{rec_col}' ({len(mr_df)} pairs)")

    met_to_recs: dict = {}
    for _, row in mr_df.iterrows():
        m = str(row[met_col]).strip().lower()
        r = str(row[rec_col]).strip().upper()
        if r and r != "NAN":
            met_to_recs.setdefault(m, []).append(r)

    module_receptor_map: dict = {}
    for mod in module_names:
        mod_l = mod.strip().lower()
        recs = set(met_to_recs.get(mod_l, []))
        if not recs:
            for m_name, rs in met_to_recs.items():
                if mod_l in m_name or m_name in mod_l:
                    recs.update(rs)
        module_receptor_map[mod] = sorted(recs)

    n_mapped = sum(1 for v in module_receptor_map.values() if v)
    log.info(f"  scFEA: {len(module_names)} modules, {n_mapped} with receptors, flux={flux_matrix.shape}")
    return {
        "flux_matrix":         flux_matrix,
        "module_names":        module_names,
        "module_receptor_map": module_receptor_map,
    }


def _step5_gene_universe(step1, step3, step4):
    """Build unified gene universe (LR genes + metabolite receptor genes)."""
    lr_df = step3["lr_df"]
    lr_genes = set(lr_df["ligand"].str.upper()) | set(lr_df["receptor"].str.upper())
    met_genes = set()
    for recs in step4["module_receptor_map"].values():
        met_genes.update(r.upper() for r in recs)
    dataset_genes = set(g.upper() for g in step1["gene_names"])
    lr_in_dataset  = sorted(lr_genes & dataset_genes)
    met_in_dataset = sorted(met_genes & dataset_genes)
    log.debug(f"  LR genes in dataset: {len(lr_in_dataset)}, metabolite receptor genes: {len(met_in_dataset)}")
    return {
        "lr_genes":  lr_in_dataset,
        "met_genes": met_in_dataset,
        "all_interaction_genes": sorted(set(lr_in_dataset) | set(met_in_dataset)),
    }


def _step6_node_features_full(step1, step5, step4_data, cfg):
    """Build gene and metabolite node feature matrices (auto-computes if missing)."""
    from mosaic.graph.node_features import NodeFeatureBuilder

    paths = cfg["paths"]
    prot_dir = paths.get("protein_emb_dir", "")
    met_dir = paths.get("metabolite_emb_dir", "")

    # Resolve paths: try absolute first, then relative to root
    root = Path(paths["root"]).parent if not Path(prot_dir).is_absolute() else Path(".")
    prot_dir = str(Path(prot_dir) if Path(prot_dir).is_absolute() else root / prot_dir)
    met_dir = str(Path(met_dir) if Path(met_dir).is_absolute() else root / met_dir)

    builder = NodeFeatureBuilder(
        protein_emb_dir=prot_dir,
        metabolite_emb_dir=met_dir,
        gene_dim=cfg["node_features"]["gene_dim"],
        metabolite_dim=cfg["node_features"]["metabolite_dim"],
        organism=cfg.get("organism", "human"),
    )
    gene_names = step5["all_interaction_genes"]
    gene_feats, gene_vocab = builder.build_gene_features(gene_names)   # [G, 1280]

    met_names = step4_data["module_names"]
    met_feats, met_vocab = builder.build_metabolite_features(met_names)  # [M, 600]

    log.debug(f"  Gene nodes: {len(gene_vocab)} × {gene_feats.shape[1]}")
    log.debug(f"  Metabolite nodes: {len(met_vocab)} × {met_feats.shape[1]}")
    return {
        "gene_feats":  gene_feats,    # [G, 1280] float32 tensor
        "gene_vocab":  gene_vocab,    # list[str]  gene name → index
        "met_feats":   met_feats,     # [M, 600] float32 tensor
        "met_vocab":   met_vocab,     # list[str]  module name → index
    }


def _step7_typed_cell_edges(step2, step3, step5, cfg):
    """Build τ₁ contact, τ₁ secreted, τ₂ metabolite cell-cell edges — LR and metabolite separate."""
    from mosaic.graph.edge_builder import EdgeBuilder

    builder = EdgeBuilder(cfg)
    result = builder.build_cell_cell_edges(
        edge_index=step2["edge_index"],
        dist_um=step2["dist_um"],
        lr_df=step3["lr_df"],
        gene_set=set(step5["lr_genes"]),
    )
    # result contains: contact_ei, contact_ea, secreted_ei, secreted_ea
    #                  (metabolite cell-cell handled separately in step8 with flux)
    return result


def _step8_metabolite_cell_edges(step2, step4, step5, cfg):
    """Build τ₂ metabolite-mediated cell-cell edges using scFEA flux (SEPARATE from LR)."""
    # Skip if no metabolite data
    flux = step4["flux_matrix"]
    if flux.size == 0 or len(step4["module_names"]) == 0:
        log.debug("  Metabolite cell-cell edges SKIPPED (no flux data)")
        return {
            "met_cc_ei": np.zeros((2, 0), dtype=np.int64),
            "met_cc_ea": np.zeros((0, 3), dtype=np.float32),
        }

    from mosaic.graph.edge_builder import EdgeBuilder

    builder = EdgeBuilder(cfg)
    result = builder.build_metabolite_cell_edges(
        edge_index=step2["edge_index"],
        dist_um=step2["dist_um"],
        flux_matrix=step4["flux_matrix"],
        module_names=step4["module_names"],
        module_receptor_map=step4["module_receptor_map"],
        met_gene_set=set(step5["met_genes"]),
    )
    return result


def _step9_cell_gene_edges(step1, step5, step6_data, cfg):
    """Build (cell, expresses, gene) edges: each cell → its top-K expressed genes."""
    from mosaic.graph.edge_builder import EdgeBuilder

    builder = EdgeBuilder(cfg)
    result = builder.build_cell_gene_edges(
        expr_matrix=step1["expr_matrix"],        # [N, n_genes] raw/norm counts
        gene_names_dataset=step1["gene_names"],  # all genes in dataset
        gene_vocab=step6_data["gene_vocab"],     # interaction genes only (subset)
        top_k=cfg["cell_gene_edges"]["top_k_per_cell"],
    )
    return result


def _step10_gene_interaction_edges(step3, step6_data):
    """Build (gene, interacts, gene) edges: one edge per LR pair in filtered database."""
    from mosaic.graph.edge_builder import EdgeBuilder

    builder = EdgeBuilder({})
    result = builder.build_gene_interaction_edges(
        lr_df=step3["lr_df"],
        gene_vocab=step6_data["gene_vocab"],
    )
    # result: lr_gene_ei [2, n_lr], lr_gene_ea [n_lr, d]
    # This is the KEY edge type — attention here = direct LR pair CCC score
    log.debug(f"  LR interaction edges: {result['lr_gene_ei'].shape[1]}")
    return result


def _step11_cell_metabolite_edges(step4, step2, step6_data, cfg):
    """Build (cell, flux, metabolite) edges: cell → metabolite module it produces (flux > q50)."""
    flux = step4["flux_matrix"]
    if flux.size == 0 or len(step4["module_names"]) == 0:
        log.debug("  Cell-metabolite edges SKIPPED (no flux data)")
        return {"cell_met_ei": np.zeros((2, 0), dtype=np.int64),
                "cell_met_ea": np.zeros((0, 2), dtype=np.float32)}

    from mosaic.graph.edge_builder import EdgeBuilder
    builder = EdgeBuilder(cfg)
    result = builder.build_cell_metabolite_edges(
        flux_matrix=flux,
        module_names=step4["module_names"],
        met_vocab=step6_data["met_vocab"],
        flux_quantile=cfg["metabolite_database"]["flux_quantile_edge"],
    )
    log.debug(f"  Cell-metabolite edges: {result['cell_met_ei'].shape[1]}")
    return result


def _step11b_metabolite_gene_edges(step4, step6_data):
    """Build (metabolite, sensed_by, gene) edges from M_R database."""
    if len(step4["module_names"]) == 0 or not step4["module_receptor_map"]:
        log.debug("  Metabolite→gene edges SKIPPED (no metabolite data)")
        return {"met_gene_ei": np.zeros((2, 0), dtype=np.int64),
                "met_gene_ea": np.zeros((0, 1), dtype=np.float32)}

    from mosaic.graph.edge_builder import EdgeBuilder
    builder = EdgeBuilder({})
    result = builder.build_metabolite_gene_edges(
        module_names=step4["module_names"],
        module_receptor_map=step4["module_receptor_map"],
        met_vocab=step6_data["met_vocab"],
        gene_vocab=step6_data["gene_vocab"],
    )
    log.debug(f"  Metabolite→gene edges: {result['met_gene_ei'].shape[1]}")
    return result


def _step12_intracellular_edges(step1, step4, step5, cfg):
    """Build τ₃ self-loop intracellular edges: receptor PCA + scFEA flux."""
    from mosaic.graph.edge_builder import EdgeBuilder

    flux = step4["flux_matrix"]
    if flux.size == 0:
        # Create zero flux matrix with correct shape for PCA
        n_cells = step1["expr_matrix"].shape[0]
        flux = np.zeros((n_cells, 0), dtype=np.float32)

    builder = EdgeBuilder(cfg)
    result = builder.build_intracellular_edges(
        expr_matrix=step1["expr_matrix"],
        gene_names=step1["gene_names"],
        receptor_genes=step5["lr_genes"] + step5["met_genes"],
        scfea_flux=flux,
        receptor_pca_dim=32,
    )
    log.debug(f"  Intracellular self-loops: {result['intra_ei'].shape[1]}, attr_dim={result['intra_ea'].shape[1]}")
    return result


def _step13_expression_labels(step1, cfg):
    """Generate y_expr [N, n_targets]: log-normalized expression for top variable genes."""
    import scipy.sparse as sp

    n_targets = int(cfg.get("labels", {}).get("n_target_genes", 200))
    X_raw     = step1["expr_matrix_raw"]   # [N, G] sparse
    gene_names = step1["gene_names"]        # list[str]

    if sp.issparse(X_raw):
        X_dense = X_raw.toarray().astype(np.float32)
    else:
        X_dense = np.array(X_raw, dtype=np.float32)

    # Select top-N most variable genes by variance of raw counts
    gene_var  = X_dense.var(axis=0)   # [G]
    top_idx   = np.argsort(-gene_var)[:n_targets]
    target_genes = [gene_names[i] for i in top_idx]

    # Log-normalize: library-size norm + log1p (non-circular w.r.t. scVI latents)
    lib_sizes = X_dense.sum(axis=1, keepdims=True).clip(min=1.0)
    X_norm    = X_dense / lib_sizes * 1e4
    y_expr    = torch.from_numpy(np.log1p(X_norm[:, top_idx]))   # [N, n_targets]

    log.debug(f"  Expression labels: {y_expr.shape}  target genes (top-5): {target_genes[:5]}")
    return {"y_expr": y_expr, "target_genes": target_genes}


def _step14_lr_labels(step2, step3, step1, cfg):
    """Generate y_lr: LR multilabel on base edges — SEPARATE from metabolite labels.
    NOTE: Not used in training (expression-only). Stored for L2 eval only.
    Returns empty tensor to avoid slow computation; fill in for full L2 eval.
    """
    E = step2["edge_index"].shape[1]
    lr_vocab = [(str(r["ligand"]), str(r["receptor"])) for _, r in step3["lr_df"].iterrows()]
    # Return minimal dict — full computation deferred to L2 evaluator
    log.debug(f"  LR labels: skipped (not used in training), {len(lr_vocab)} pairs in vocab")
    return {
        "edge_lr_multilabel": torch.zeros((E, 0), dtype=torch.float32),
        "lr_vocab": lr_vocab,
    }


def _step15_metabolite_labels(step2, step4, step1, cfg):
    """Generate y_metab: per-edge metabolite multilabel — SEPARATE from LR labels.
    NOTE: Not used in training. Stored for L2 eval.
    """
    E = step2["edge_index"].shape[1]
    log.debug(f"  Metabolite labels: skipped (not used in training)")
    return {
        "y_metab":    torch.zeros((E, 0), dtype=torch.float32),
        "metab_vocab": step4["module_names"],
    }


def _step16_spatial_splits(step2, cfg):
    """Build spatial cross-validation node masks (train/val/test) for cells."""
    from mosaic.data.spatial_cv import make_spatial_splits

    sp_cfg = cfg.get("spatial_splits", {})
    result = make_spatial_splits(
        coords=step2["coords_um"],
        edge_index=step2["edge_index"].numpy(),
        n_clusters=int(sp_cfg.get("n_clusters", 22)),
        val_frac=float(sp_cfg.get("val_frac", 0.14)),
        test_frac=float(sp_cfg.get("test_frac", 0.14)),
        seed=int(sp_cfg.get("seed", 42)),
    )
    # Derive node-level masks from cell_split (0=train, 1=val, 2=test)
    cell_split = result["cell_split"]   # [N] int8
    node_train_mask = torch.from_numpy(cell_split == 0)
    node_val_mask   = torch.from_numpy(cell_split == 1)
    node_test_mask  = torch.from_numpy(cell_split == 2)

    log.debug(f"  Node splits — train: {node_train_mask.sum()}, "
             f"val: {node_val_mask.sum()}, test: {node_test_mask.sum()}")
    return {
        "node_train_mask": node_train_mask,
        "node_val_mask":   node_val_mask,
        "node_test_mask":  node_test_mask,
        "cell_split":      cell_split,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(config_path: str, dataset: str, force: bool = False):
    cfg = load_config(config_path, dataset)
    out_dir = cfg["_processed_dir"]
    log.debug(f"=== MOSAIC preprocessing: {dataset} → {out_dir} ===")

    # Check if final graph already exists
    final_path = out_dir / "hetero_ccc_graph.pt"
    if final_path.exists() and not force:
        log.debug(f"Final graph already exists: {final_path}. Use --force to recompute.")
        return

    cache = load_cache(out_dir)

    # ── Step 1: Load AnnData ───────────────────────────────────────────────
    log.debug("[Step 1] Load AnnData → cell features + spatial coords")
    s1 = cached("anndata", cache, _step1_load_anndata, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 2: Spatial graph (k-NN) ──────────────────────────────────────
    log.info("[Step 2] Spatial graph (k-NN, µm coords)")
    s2 = cached("spatial_graph", cache, _step2_spatial_coords, s1, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 3: LR database (SEPARATE) ────────────────────────────────────
    log.debug("[Step 3] LR database (contact + secreted, separate from metabolites)")
    s3 = cached("lr_database", cache, _step3_lr_database, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 4: Metabolite database + scFEA flux (SEPARATE) ───────────────
    log.info("[Step 4] Metabolite database + scFEA flux (separate from LR)")
    s4 = cached("metabolite_database", cache, _step4_metabolite_database, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 5: Gene universe ─────────────────────────────────────────────
    log.debug("[Step 5] Gene universe (LR genes ∪ metabolite receptor genes)")
    s5 = cached("gene_universe", cache, _step5_gene_universe, s1, s3, s4, force=force)
    save_cache(cache, out_dir)

    # ── Step 6: Node features (gene ESM-2, metabolite ChemBERTa) ──────────
    log.debug("[Step 6] Node features (gene ESM-2 1280-dim, metabolite ChemBERTa 600-dim)")
    s6 = cached("node_features", cache, _step6_node_features_full, s1, s5, s4, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 7: Cell-cell edges: contact + secreted (LR-based) ────────────
    log.debug("[Step 7] Cell-cell LR edges (τ₁ contact, τ₁ secreted)")
    s7 = cached("cell_cell_lr_edges", cache, _step7_typed_cell_edges, s2, s3, s5, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 8: Cell-cell edges: metabolite channel (flux-based) ──────────
    log.debug("[Step 8] Cell-cell metabolite edges (τ₂, flux-based, separate from LR)")
    s8 = cached("cell_cell_met_edges", cache, _step8_metabolite_cell_edges, s2, s4, s5, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 9: Cell→gene expression edges ────────────────────────────────
    log.debug("[Step 9] Cell-gene expression edges (cell expresses gene, top-K per cell)")
    s9 = cached("cell_gene_edges", cache, _step9_cell_gene_edges, s1, s5, s6, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 10: Gene↔gene LR interaction edges ───────────────────────────
    log.info("[Step 10] Gene-interaction edges (gene interacts gene = LR pairs, KEY for CCC)")
    s10 = cached("gene_interaction_edges", cache, _step10_gene_interaction_edges, s3, s6, force=force)
    save_cache(cache, out_dir)

    # ── Step 11: Cell→metabolite flux edges ───────────────────────────────
    log.debug("[Step 11] Cell-metabolite flux edges (cell flux metabolite)")
    s11 = cached("cell_metabolite_edges", cache, _step11_cell_metabolite_edges, s4, s2, s6, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 11b: Metabolite→gene sensed_by edges ─────────────────────────
    log.debug("[Step 11b] Metabolite→gene sensed_by edges (ε₄: M_R database)")
    s11b = cached("metabolite_gene_edges", cache, _step11b_metabolite_gene_edges, s4, s6, force=force)
    save_cache(cache, out_dir)

    # ── Step 12: Intracellular self-loops ─────────────────────────────────
    log.info("[Step 12] Intracellular self-loop edges (τ₃: receptor PCA + scFEA flux)")
    s12 = cached("intracellular_edges", cache, _step12_intracellular_edges, s1, s4, s5, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 13: Expression labels y_expr ─────────────────────────────────
    log.debug("[Step 13] Expression labels y_expr [N, 200]")
    s13 = cached("expr_labels", cache, _step13_expression_labels, s1, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 14: LR labels y_lr (SEPARATE from metabolite) ────────────────
    log.debug("[Step 14] LR edge labels y_lr [E, n_pairs] (separate from metabolite labels)")
    s14 = cached("lr_labels", cache, _step14_lr_labels, s2, s3, s1, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 15: Metabolite labels y_metab (SEPARATE from LR) ─────────────
    log.debug("[Step 15] Metabolite edge labels y_metab [E, n_modules] (separate from LR labels)")
    s15 = cached("met_labels", cache, _step15_metabolite_labels, s2, s4, s1, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 16: Spatial CV splits ─────────────────────────────────────────
    log.debug("[Step 16] Spatial cross-validation splits (train/val/test cell masks)")
    s16 = cached("spatial_splits", cache, _step16_spatial_splits, s2, cfg, force=force)
    save_cache(cache, out_dir)

    # ── Step 17: Assemble HeteroData ──────────────────────────────────────
    log.debug("[Step 17] Assemble HeteroData → hetero_ccc_graph.pt")
    _assemble_and_save(cache, cfg, out_dir)

    log.debug(f"=== Preprocessing complete: {out_dir}/hetero_ccc_graph.pt ===")


def _assemble_and_save(cache: dict, cfg: dict, out_dir: Path):
    from mosaic.graph.assembler import GraphAssembler

    assembler = GraphAssembler(cfg)
    graph, metadata = assembler.assemble(cache)

    # Save final graph
    torch.save({"hetero_graph": graph, "metadata": metadata},
               out_dir / "hetero_ccc_graph.pt")

    # Save metadata as JSON (human-readable)
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    log.debug(f"  Saved hetero_ccc_graph.pt")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOSAIC preprocessing pipeline")
    parser.add_argument("--config",  required=True, help="Path to config YAML")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. breast_new)")
    parser.add_argument("--force",   action="store_true", help="Ignore cache and recompute all steps")
    args = parser.parse_args()
    preprocess_dataset(args.config, args.dataset, force=args.force)
