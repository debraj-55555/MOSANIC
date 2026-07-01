"""
mosanic/cli.py — Unified command-line entry point for MOSANIC.

Usage:
    mosanic preprocess --config configs/breast.yaml [--force]
    mosanic train      --config configs/breast.yaml [--device cuda] [--checkpoint path]
    mosanic evaluate   --config configs/breast.yaml [--device cuda]
    mosanic run        --config configs/breast.yaml [--device cuda]   # full pipeline

Each subcommand loads a YAML config and delegates to the appropriate module.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mosanic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    """Load and return a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        log.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve a device string to a torch.device, falling back to CPU."""
    if device_str == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_paths(cfg: dict, dataset: str):
    """Derive processed_dir, graph_path, and checkpoint_path from config."""
    root = Path(cfg["paths"]["root"]).parent
    processed_dir = root / cfg["paths"]["processed_dir"] / dataset
    graph_path = processed_dir / "hetero_ccc_graph.pt"
    ckpt_dir = Path(
        cfg.get("training", {}).get("checkpoint_dir", "mosanic/checkpoints")
    ) / dataset
    ckpt_path = ckpt_dir / "model_best.pt"
    return processed_dir, graph_path, ckpt_path


def _load_graph(graph_path: Path):
    """Load the hetero CCC graph and metadata from disk."""
    if not graph_path.exists():
        log.error(
            "Graph not found: %s\n"
            "Run: mosanic preprocess --config <config.yaml> first.",
            graph_path,
        )
        sys.exit(1)

    ds = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = ds["hetero_graph"]
    meta = ds["metadata"]
    log.info("Loaded graph: %d cells, %d genes, %d metabolites, %d target genes",
             meta["n_cells"], meta["n_genes"], meta["n_metabolites"], meta["n_expr_genes"])
    return data, meta


# ---------------------------------------------------------------------------
# Subcommand: preprocess
# ---------------------------------------------------------------------------

def cmd_preprocess(args):
    """Run the preprocessing pipeline to build the heterogeneous graph."""
    cfg = _load_config(args.config)
    dataset = cfg["dataset"]

    from mosanic.data import preprocess_dataset

    log.info("=== MOSANIC preprocess: %s ===", dataset)
    preprocess_dataset(args.config, dataset, force=args.force)
    log.info("Preprocessing complete.")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train MOSANIC on a preprocessed graph."""
    cfg = _load_config(args.config)
    dataset = cfg["dataset"]
    device = _resolve_device(args.device)
    processed_dir, graph_path, ckpt_path = _resolve_paths(cfg, dataset)

    # --- Optionally run preprocessing if graph missing --------------------
    if not graph_path.exists():
        log.info("Graph not found — running preprocessing...")
        from mosanic.data import preprocess_dataset
        preprocess_dataset(args.config, dataset, force=getattr(args, 'force_preprocess', False))

    # --- Load graph -------------------------------------------------------
    data, meta = _load_graph(graph_path)

    # --- CLI overrides ----------------------------------------------------
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.patience is not None:
        cfg.setdefault("training", {})["patience"] = args.patience
    if args.lr is not None:
        cfg.setdefault("training", {})["lr"] = args.lr
    if args.n_layers is not None:
        cfg.setdefault("model", {})["n_layers"] = args.n_layers
    if args.lambda_spatial is not None:
        cfg.setdefault("training", {})["lambda_spatial"] = args.lambda_spatial

    # --- Build model ------------------------------------------------------
    from mosanic.models import build_model

    model = build_model(cfg, n_expr_genes=meta["n_expr_genes"], graph_metadata=meta)

    # --- Optional warm-start from checkpoint ------------------------------
    if args.checkpoint:
        ckpt_file = Path(args.checkpoint)
        if not ckpt_file.exists():
            log.error("Checkpoint not found: %s", ckpt_file)
            sys.exit(1)
        log.info("Warm-starting from: %s", ckpt_file)
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        log.info("  Checkpoint loaded successfully.")

    param_counts = model.count_parameters()
    log.info("Model: %s (%d params)", model.__class__.__name__, param_counts["total"])

    # --- Train ------------------------------------------------------------
    from mosanic.training import MOSANICTrainer

    trainer = MOSANICTrainer(
        model=model,
        data=data,
        config=cfg,
        device=device,
        dataset=dataset,
    )

    n_epochs = cfg.get("training", {}).get("epochs", 500)
    log.info("=== MOSANIC train: %s on %s for %d epochs ===", dataset, device, n_epochs)
    history = trainer.train(n_epochs)

    log.info("Training complete.")
    log.info("Best val R² = %.4f", max(history["val_r2"]))

    # Load best checkpoint and return MOSANICResult
    from mosanic.api import MOSANICResult

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)

    result = MOSANICResult(
        model=model, data=data, meta=meta,
        config=cfg, dataset=dataset, device=device,
    )
    log.info("Returning MOSANICResult — use result.lr_pairs(), result.plot_spatial(), etc.")
    return result


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    """Evaluate a trained MOSANIC model on CCC metrics."""
    cfg = _load_config(args.config)
    dataset = cfg["dataset"]

    from mosanic.evaluation import evaluate_model

    log.info("=== MOSANIC evaluate: %s ===", dataset)
    results = evaluate_model(
        cfg,
        dataset,
        device=args.device,
        channel=args.channel,
    )
    log.info("Evaluation complete.")
    return results


# ---------------------------------------------------------------------------
# Subcommand: run  (preprocess + train + evaluate)
# ---------------------------------------------------------------------------

def cmd_run(args):
    """Execute the full MOSANIC pipeline: preprocess -> train -> evaluate."""
    cfg = _load_config(args.config)
    dataset = cfg["dataset"]
    log.info("=== MOSANIC full pipeline: %s ===", dataset)

    # Step 1: preprocess
    log.info("[1/3] Preprocessing...")
    from mosanic.data import preprocess_dataset
    preprocess_dataset(args.config, dataset, force=args.force_preprocess)

    # Step 2: train (returns MOSANICResult)
    log.info("[2/3] Training...")
    result = cmd_train(args)

    # Step 3: evaluate
    log.info("[3/3] Evaluating...")
    cmd_evaluate(args)

    log.info("=== MOSANIC pipeline complete for %s ===", dataset)
    return result


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mosanic",
        description=(
            "MOSANIC: Multi-mOdal Self-Attention Network for Intercellular Communication.\n"
            "A heterogeneous graph transformer for cell-cell communication inference\n"
            "from spatial transcriptomics data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  mosanic preprocess --config configs/breast.yaml\n"
            "  mosanic train      --config configs/breast.yaml --device cuda\n"
            "  mosanic evaluate   --config configs/breast.yaml --device cuda\n"
            "  mosanic run        --config configs/breast.yaml --device cuda\n"
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available pipeline stages",
    )

    # --- preprocess -------------------------------------------------------
    pp = subparsers.add_parser(
        "preprocess",
        help="Build the heterogeneous CCC graph from raw AnnData + databases",
        description=(
            "Load raw AnnData, compute scVI embeddings, build spatial graph, "
            "resolve LR/metabolite edges, and save hetero_ccc_graph.pt."
        ),
    )
    pp.add_argument(
        "--config", required=True,
        help="Path to dataset YAML config file",
    )
    pp.add_argument(
        "--force", action="store_true",
        help="Recompute all steps even if cached graph exists",
    )
    pp.set_defaults(func=cmd_preprocess)

    # --- train ------------------------------------------------------------
    tr = subparsers.add_parser(
        "train",
        help="Train MOSANIC on a preprocessed heterogeneous graph",
        description=(
            "Load hetero_ccc_graph.pt, build the MOSANIC transformer, "
            "train with expression prediction objective, and save the best checkpoint."
        ),
    )
    tr.add_argument(
        "--config", required=True,
        help="Path to dataset YAML config file",
    )
    tr.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    tr.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to checkpoint for warm-starting (fine-tuning)",
    )
    tr.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs from config",
    )
    tr.add_argument(
        "--patience", type=int, default=None,
        help="Override early stopping patience from config",
    )
    tr.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate from config",
    )
    tr.add_argument(
        "--n-layers", type=int, default=None, dest="n_layers",
        help="Override number of encoder layers from config",
    )
    tr.add_argument(
        "--lambda-spatial", type=float, default=None, dest="lambda_spatial",
        help="Override spatial attention regularization weight",
    )
    tr.add_argument(
        "--force-preprocess", action="store_true", dest="force_preprocess",
        help="Re-run preprocessing even if graph exists",
    )
    tr.set_defaults(func=cmd_train)

    # --- evaluate ---------------------------------------------------------
    ev = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained MOSANIC model on CCC metrics",
        description=(
            "Load a trained checkpoint, extract attention-based LR scores, "
            "and compute DES, AUROC, ARI, and NMI metrics."
        ),
    )
    ev.add_argument(
        "--config", required=True,
        help="Path to dataset YAML config file",
    )
    ev.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    ev.add_argument(
        "--channel", default="all",
        choices=["all", "contact", "secreted", "metabolite"],
        help="Cell-cell channel for DES evaluation (default: all)",
    )
    ev.set_defaults(func=cmd_evaluate)

    # --- run (full pipeline) ----------------------------------------------
    rn = subparsers.add_parser(
        "run",
        help="Run the full pipeline: preprocess -> train -> evaluate",
        description=(
            "Execute all three stages sequentially. Equivalent to running "
            "preprocess, train, and evaluate in order."
        ),
    )
    rn.add_argument(
        "--config", required=True,
        help="Path to dataset YAML config file",
    )
    rn.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    rn.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to checkpoint for warm-starting (fine-tuning)",
    )
    rn.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs from config",
    )
    rn.add_argument(
        "--patience", type=int, default=None,
        help="Override early stopping patience from config",
    )
    rn.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate from config",
    )
    rn.add_argument(
        "--n-layers", type=int, default=None, dest="n_layers",
        help="Override number of encoder layers from config",
    )
    rn.add_argument(
        "--lambda-spatial", type=float, default=None, dest="lambda_spatial",
        help="Override spatial attention regularization weight",
    )
    rn.add_argument(
        "--force-preprocess", action="store_true", dest="force_preprocess",
        help="Re-run preprocessing even if graph exists",
    )
    rn.add_argument(
        "--channel", default="all",
        choices=["all", "contact", "secreted", "metabolite"],
        help="Cell-cell channel for DES evaluation (default: all)",
    )
    rn.set_defaults(func=cmd_run)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Main CLI entry point for MOSANIC."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
