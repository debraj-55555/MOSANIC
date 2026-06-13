"""
mosaic/analysis/relay.py

Relay network analysis from MOSAIC per-edge attention.
Extracts per-cell-pair attention, builds relay graph, finds connected components.
Computes metrics comparable to CellNEST relay analysis.

Usage:
    python -m mosaic.analysis.relay --dataset breast_new --device cuda:0
"""
import argparse, json, logging, sys, time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml
import networkx as nx
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_model_and_data(dataset: str, device: str, config_dir: str = None, data_dir: str = None):
    """Load trained MOSAIC model + hetero graph."""
    from mosaic.models import build_model

    if config_dir is None:
        config_dir = "mosaic/configs"
    if data_dir is None:
        data_dir = "mosaic/data/processed"

    cfg_path = Path(config_dir) / f"{dataset.replace('_new','')}_config.yaml"
    if not cfg_path.exists():
        cfg_path = Path(config_dir) / "breast_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    graph_data = torch.load(Path(data_dir) / dataset / "hetero_ccc_graph.pt",
                            map_location="cpu", weights_only=False)
    data = graph_data['hetero_graph']
    metadata = graph_data['metadata']

    n_expr_genes = int(metadata.get('n_expr_genes', metadata.get('n_target_genes', 200)))
    model = build_model(cfg, n_expr_genes, graph_metadata=metadata)
    ckpt_path = Path(cfg.get("training", {}).get("checkpoint_dir", "mosaic/checkpoints")) / dataset / "model_best.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model = model.to(device)
    data = data.to(device)
    model.eval()

    return model, data, cfg


def extract_per_edge_attention(model, data, device):
    """Extract per-cell-pair attention for all edge types."""
    from mosaic.evaluation.ccc_extractor import CCCExtractor

    extractor = CCCExtractor(model, data, device=device)
    extraction = extractor.extract()

    result = {
        'edge_scores': {},       # {edge_type_str: np.array}
        'edge_indices': {},      # {edge_type_str: np.array [2, E]}
        'cell_cell_scores': {},  # {relation: np.array [E_cc]}
        'lr_pair_scores': extraction['lr_pair_edge_scores'],
        'lr_pair_names': extraction.get('lr_pair_names', None),
    }

    # Per-edge attention for ALL edge types
    for et, scores in extraction['edge_scores'].items():
        et_str = "__".join(et) if isinstance(et, tuple) else str(et)
        result['edge_scores'][et_str] = scores
        ei = data[et].edge_index.cpu().numpy()
        result['edge_indices'][et_str] = ei

    # Cell-cell specific
    for relation, scores in extraction['cell_cell_edge_scores'].items():
        result['cell_cell_scores'][relation] = scores

    return result


def build_relay_graph(edge_index, attention_scores, top_pct=20):
    """Build relay graph from top-K% attention edges.

    CellNEST approach: keep top_pct% of edges by attention score,
    find connected components = relay networks.
    """
    n_edges = len(attention_scores)
    threshold_idx = int(n_edges * (1 - top_pct / 100))
    sorted_idx = np.argsort(attention_scores)
    keep_mask = np.zeros(n_edges, dtype=bool)
    keep_mask[sorted_idx[threshold_idx:]] = True

    threshold = attention_scores[sorted_idx[threshold_idx]]

    # Build networkx graph
    G = nx.Graph()
    src = edge_index[0, keep_mask]
    dst = edge_index[1, keep_mask]
    weights = attention_scores[keep_mask]

    for s, d, w in zip(src, dst, weights):
        G.add_edge(int(s), int(d), weight=float(w))

    # Connected components
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)

    return {
        'graph': G,
        'components': components,
        'n_components': len(components),
        'threshold': float(threshold),
        'n_edges_kept': int(keep_mask.sum()),
        'n_edges_total': n_edges,
        'component_sizes': [len(c) for c in components],
    }


def analyze_relay_biology(relay_result, cell_types, ct_mapping, coords=None):
    """Analyze biological composition of relay networks."""
    components = relay_result['components']

    relay_info = []
    for comp_id, comp in enumerate(components[:20]):  # top 20
        cells = list(comp)
        n_cells = len(cells)
        if n_cells < 5:
            continue

        # Cell type composition
        ct_in_comp = [cell_types[c] for c in cells if c < len(cell_types)]
        ct_counts = Counter(ct_in_comp)
        total = sum(ct_counts.values())
        dominant_ct, dominant_n = ct_counts.most_common(1)[0]
        dominant_pct = dominant_n / total
        n_types = len(ct_counts)
        is_interface = dominant_pct < 0.70 and n_types >= 2

        # Spatial extent (if coords available)
        spatial_extent = None
        if coords is not None:
            comp_coords = coords[cells]
            spatial_extent = np.sqrt(np.sum((comp_coords.max(0) - comp_coords.min(0))**2))

        info = {
            'comp_id': comp_id,
            'n_cells': n_cells,
            'n_types': n_types,
            'dominant_ct': int(dominant_ct),
            'dominant_ct_name': ct_mapping.get(int(dominant_ct), f'CT{dominant_ct}'),
            'dominant_pct': float(dominant_pct),
            'is_interface': is_interface,
            'ct_composition': {ct_mapping.get(int(k), f'CT{k}'): int(v) for k, v in ct_counts.items()},
            'spatial_extent': float(spatial_extent) if spatial_extent is not None else None,
        }
        relay_info.append(info)

    return relay_info


def compare_with_cellnest(our_relay, cellnest_scores_path=None):
    """Compute metrics for quantitative comparison with CellNEST.

    CellNEST metrics (from their paper):
    1. Number of connected components
    2. Size distribution of components
    3. Fraction of interface (mixed cell-type) communities
    4. Biological coherence: known LR pairs enriched in community edges
    """
    components = our_relay['components']
    sizes = [len(c) for c in components]

    metrics = {
        'n_components': len(components),
        'largest_component': sizes[0] if sizes else 0,
        'median_component': float(np.median(sizes)) if sizes else 0,
        'mean_component': float(np.mean(sizes)) if sizes else 0,
        'n_singleton': sum(1 for s in sizes if s == 1),
        'n_large': sum(1 for s in sizes if s >= 10),
        'component_sizes': sizes[:50],  # top 50
    }

    return metrics


def detect_relays(model, data, cell_types, ct_mapping, coords=None,
                  device="cpu", top_pct_values=None):
    """
    High-level API: detect relay networks from a trained MOSAIC model.

    Args:
        model: trained MOSAIC model
        data: HeteroData graph
        cell_types: [N] integer cell type labels
        ct_mapping: {int: str} cell type ID to name mapping
        coords: [N, 2] spatial coordinates (optional)
        device: torch device string
        top_pct_values: list of percentile thresholds (default: [5, 10, 15, 20, 30, 50])

    Returns:
        dict with relay analysis results for each threshold
    """
    if top_pct_values is None:
        top_pct_values = [5, 10, 15, 20, 30, 50]

    # Extract per-edge attention
    log.info("Extracting per-edge attention...")
    t0 = time.time()
    extraction = extract_per_edge_attention(model, data, device)
    log.info("Extraction done in %.1fs", time.time() - t0)

    # Log available edge types
    for et_str, scores in extraction['edge_scores'].items():
        log.info("  %s: %d edges, mean=%.4f, max=%.4f",
                 et_str, len(scores), scores.mean(), scores.max())

    # Find secreted edge type
    sec_key = None
    for k in extraction['cell_cell_scores']:
        if 'secreted' in k.lower():
            sec_key = k
            break

    if sec_key is None:
        for k in extraction['edge_scores']:
            if 'secreted' in k.lower():
                sec_key = k
                break

    if sec_key is None:
        log.error("No secreted edge type found!")
        log.info("Available: %s", list(extraction['edge_scores'].keys()))
        log.info("Cell-cell: %s", list(extraction['cell_cell_scores'].keys()))
        return {}

    # Get edge index for secreted
    sec_scores = extraction['cell_cell_scores'].get(sec_key,
                  extraction['edge_scores'].get(sec_key))

    # Find matching edge index
    sec_ei = None
    for et_str, ei in extraction['edge_indices'].items():
        if 'secreted' in et_str.lower():
            sec_ei = ei
            break

    if sec_ei is None:
        log.error("No secreted edge index found")
        return {}

    log.info("Secreted edges: %d, edge_index: %s", len(sec_scores), sec_ei.shape)

    # Build relay at multiple thresholds
    all_results = {}
    for pct in top_pct_values:
        relay = build_relay_graph(sec_ei, sec_scores, top_pct=pct)
        relay_bio = analyze_relay_biology(relay, cell_types, ct_mapping, coords)
        metrics = compare_with_cellnest(relay)

        n_interface = sum(1 for r in relay_bio if r['is_interface'])

        log.info("  top-%d%%: %d components, largest=%d, interface=%d/%d",
                 pct, relay['n_components'],
                 relay['component_sizes'][0] if relay['component_sizes'] else 0,
                 n_interface, len(relay_bio))

        all_results[f'top_{pct}pct'] = {
            'relay_metrics': metrics,
            'relay_biology': relay_bio,
            'threshold': relay['threshold'],
            'n_edges_kept': relay['n_edges_kept'],
        }

    # -- Also extract metabolite channel attention -----------------------
    met_key = None
    for k in list(extraction['cell_cell_scores'].keys()) + list(extraction['edge_scores'].keys()):
        if 'metabolite' in k.lower() and 'sensed' not in k.lower():
            met_key = k
            break

    met_scores = None
    met_ei = None
    if met_key:
        met_scores = extraction['cell_cell_scores'].get(met_key,
                      extraction['edge_scores'].get(met_key))
        for et_str, ei_arr in extraction['edge_indices'].items():
            if 'metabolite' in et_str.lower() and 'sensed' not in et_str.lower():
                met_ei = ei_arr
                break
        if met_scores is not None:
            log.info("Metabolite edges: %d", len(met_scores))

    # -- 2-HOP RELAY ANALYSIS ------------------------------------------
    log.info("Computing 2-hop relay chains...")

    def compute_2hop_relays(ei, attn, cell_types_arr, ct_map, coord_arr, top_pct_val=10):
        """Find A->B->C relay chains where BOTH edges have strong attention."""
        threshold = np.percentile(attn, 100 - top_pct_val)
        strong = attn >= threshold
        s_src, s_dst, s_attn = ei[0, strong], ei[1, strong], attn[strong]

        # Build neighbor lookup
        edge_lookup = defaultdict(list)
        for i in range(len(s_src)):
            edge_lookup[int(s_src[i])].append((int(s_dst[i]), float(s_attn[i])))

        relays = []
        for a in edge_lookup:
            for b, ab_attn in edge_lookup[a]:
                if b not in edge_lookup:
                    continue
                for c, bc_attn in edge_lookup[b]:
                    if c == a:
                        continue
                    product = ab_attn * bc_attn
                    dist_ab = float(np.sqrt(((coord_arr[a] - coord_arr[b])**2).sum())) if coord_arr is not None else 0.0
                    dist_bc = float(np.sqrt(((coord_arr[b] - coord_arr[c])**2).sum())) if coord_arr is not None else 0.0
                    relays.append({
                        'a': int(a), 'b': int(b), 'c': int(c),
                        'ab_attn': float(ab_attn), 'bc_attn': float(bc_attn),
                        'product': float(product),
                        'ct_a': ct_map.get(int(cell_types_arr[a]), f'CT{cell_types_arr[a]}'),
                        'ct_b': ct_map.get(int(cell_types_arr[b]), f'CT{cell_types_arr[b]}'),
                        'ct_c': ct_map.get(int(cell_types_arr[c]), f'CT{cell_types_arr[c]}'),
                        'dist_ab': dist_ab, 'dist_bc': dist_bc,
                    })

        relays.sort(key=lambda x: -x['product'])
        return relays

    sec_2hop = compute_2hop_relays(sec_ei, sec_scores, cell_types, ct_mapping, coords, top_pct_val=10)
    log.info("Secreted 2-hop relays (top-10%%): %d", len(sec_2hop))
    for r in sec_2hop[:3]:
        log.info("  %s->%s->%s  prod=%.4f  dist=%.0f+%.0fpx",
                 r['ct_a'][:15], r['ct_b'][:15], r['ct_c'][:15],
                 r['product'], r['dist_ab'], r['dist_bc'])

    met_2hop = []
    if met_scores is not None and met_ei is not None:
        met_2hop = compute_2hop_relays(met_ei, met_scores, cell_types, ct_mapping, coords, top_pct_val=10)
        log.info("Metabolite 2-hop relays (top-10%%): %d", len(met_2hop))

    # Cross-channel 2-hop: A->B via secreted (top-10%), B->C via metabolite (top-10%)
    cross_2hop = []
    if met_scores is not None and met_ei is not None:
        sec_thresh = np.percentile(sec_scores, 90)
        met_thresh = np.percentile(met_scores, 90)
        sec_strong = sec_scores >= sec_thresh
        met_strong = met_scores >= met_thresh

        sec_lookup = defaultdict(list)
        for i in range(len(sec_scores)):
            if sec_strong[i]:
                sec_lookup[int(sec_ei[0, i])].append((int(sec_ei[1, i]), float(sec_scores[i])))

        met_lookup = defaultdict(list)
        for i in range(len(met_scores)):
            if met_strong[i]:
                met_lookup[int(met_ei[0, i])].append((int(met_ei[1, i]), float(met_scores[i])))

        for a in sec_lookup:
            for b, ab_attn in sec_lookup[a]:
                if b not in met_lookup:
                    continue
                for c, bc_attn in met_lookup[b]:
                    if c == a:
                        continue
                    product = ab_attn * bc_attn
                    cross_2hop.append({
                        'a': int(a), 'b': int(b), 'c': int(c),
                        'ab_attn': float(ab_attn), 'bc_attn': float(bc_attn),
                        'product': float(product),
                        'ct_a': ct_mapping.get(int(cell_types[a]), f'CT{cell_types[a]}'),
                        'ct_b': ct_mapping.get(int(cell_types[b]), f'CT{cell_types[b]}'),
                        'ct_c': ct_mapping.get(int(cell_types[c]), f'CT{cell_types[c]}'),
                        'channel_ab': 'secreted', 'channel_bc': 'metabolite',
                    })
        cross_2hop.sort(key=lambda x: -x['product'])
        log.info("Cross-channel 2-hop relays: %d", len(cross_2hop))

    all_results['2hop_secreted'] = sec_2hop[:100]
    all_results['2hop_metabolite'] = met_2hop[:100]
    all_results['2hop_cross_channel'] = cross_2hop[:100]

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='breast_new')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--top_pct', type=float, default=20, help='Top %% edges to keep')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--config_dir', default=None)
    parser.add_argument('--data_dir', default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results_figure/relay/{args.dataset}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log.info("Dataset: %s, Device: %s", args.dataset, args.device)

    # Load model
    log.info("Loading model and data...")
    model, data, cfg = load_model_and_data(args.dataset, args.device,
                                            config_dir=args.config_dir,
                                            data_dir=args.data_dir)

    # Load cell types and coords from preprocessing cache
    data_dir = args.data_dir or "mosaic/data/processed"
    cache = torch.load(Path(data_dir) / args.dataset / "preprocessing_cache.pt",
                       map_location="cpu", weights_only=False)
    cell_types = np.array(cache['anndata']['cell_types'])
    coords = np.array(cache['anndata']['coords_px'])

    # Build CT mapping
    ct_mapping = {i: f'CT{i}' for i in range(len(np.unique(cell_types)))}

    # Run relay detection
    all_results = detect_relays(
        model, data, cell_types, ct_mapping, coords,
        device=args.device,
    )

    # Save results
    out_path = Path(args.output_dir) / 'relay_analysis.json'
    save_data = {
        'dataset': args.dataset,
        'relay_results': {k: v for k, v in all_results.items()
                         if not k.startswith('2hop')},
        '2hop_secreted': all_results.get('2hop_secreted', []),
        '2hop_metabolite': all_results.get('2hop_metabolite', []),
        '2hop_cross_channel': all_results.get('2hop_cross_channel', []),
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info("Saved relay analysis: %s", out_path)

    # Print summary
    best_key = 'top_20pct'
    if best_key in all_results:
        best = all_results[best_key]
        print(f"\n{'='*60}")
        print(f"  RELAY ANALYSIS SUMMARY ({args.dataset})")
        print(f"{'='*60}")
        print(f"  Top-20% relay: {best['relay_metrics']['n_components']} components")
        print(f"  Largest: {best['relay_metrics']['largest_component']} cells")
        print(f"  Interface communities: {sum(1 for r in best['relay_biology'] if r['is_interface'])}")
        print(f"  2-hop secreted relays: {len(all_results.get('2hop_secreted', []))}")
        print(f"  2-hop metabolite relays: {len(all_results.get('2hop_metabolite', []))}")
        print(f"  2-hop cross-channel relays: {len(all_results.get('2hop_cross_channel', []))}")


if __name__ == '__main__':
    main()
