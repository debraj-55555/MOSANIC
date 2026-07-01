"""
mosanic/evaluation/ccc_extractor.py

CCC Extractor for the MOSANIC model.

Key feature:
  - Extracts attention on (gene, interacts, gene) edges -> DIRECT LR pair scores.
    No post-hoc gene-weighting correction needed.
  - score(LR_pair_k) = mean attention over edges (ligand_k -> receptor_k)
    across heads and layers.

Additionally extracts:
  - Cell-cell channel attention (contact, secreted, metabolite, intracellular)
  - Gate weights per destination node type

Usage:
    extractor = CCCExtractor(model, data, device)
    result = extractor.extract()
    lr_scores = extractor.get_lr_pair_scores(result, lr_pair_vocab)
    comm_matrix = extractor.get_cell_communication_matrix(result, cell_types)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class CCCExtractor:
    """
    Extract CCC signals from a trained MOSANIC model.

    The model's attention on (gene, interacts, gene) edges gives
    direct LR pair importance scores -- no post-hoc correction needed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        device: torch.device = torch.device("cpu"),
    ):
        self.model  = model
        self.data   = data
        self.device = device

    # -----------------------------------------------------------------
    # Full extraction
    # -----------------------------------------------------------------

    @torch.no_grad()
    def extract(self) -> Dict:
        """
        Run forward pass with attention extraction.

        Returns dict:
            'edge_scores':            {edge_type: [E_t] float}   per-edge attention scores
            'gate_weights':           {dst_node_type: {edge_type_rel: float}}  avg gate weights
            'attention_raw':          {edge_type: [E_t, n_heads, n_layers]}
            'lr_pair_edge_scores':    [n_lr_pairs]   direct LR pair attention (KEY)
            'cell_cell_edge_scores':  {relation: [E_cc]}   cell-cell attention only
            'edge_types':             list of edge type tuples
        """
        self.model.eval()
        data = self.data.to(self.device)

        result        = self.model(data, return_attention=True)
        attention_info = result["attention_info"]

        per_layer    = attention_info["per_layer"]    # list of {edge_type: [E_t, heads]}
        gate_weights = attention_info["gate_weights"] # list of {dst_type: [n_et] tensor}

        n_layers   = len(per_layer)
        edge_types = list(per_layer[0].keys())

        # -- Aggregate attention per edge type -------------------------
        edge_scores   = {}
        attention_raw = {}

        for et in edge_types:
            layer_attns = []
            for layer_attn in per_layer:
                if et in layer_attn:
                    alpha = layer_attn[et].detach().cpu()   # [E_t, n_heads]
                    layer_attns.append(alpha)

            if not layer_attns:
                continue

            # Pad to same E_t (in case of minor differences across layers)
            min_e = min(a.shape[0] for a in layer_attns)
            trimmed = [a[:min_e] for a in layer_attns]
            stacked = torch.stack(trimmed, dim=-1)   # [E_t, n_heads, n_layers]

            attention_raw[et] = stacked
            edge_scores[et]   = stacked.mean(dim=(1, 2)).numpy()  # [E_t]

        # -- Aggregate gate weights across layers ----------------------
        gate_accum: Dict[str, Dict[str, List[float]]] = {}
        for layer_gates in gate_weights:
            for dst_type, gate_vec in layer_gates.items():
                if dst_type not in gate_accum:
                    gate_accum[dst_type] = {}
                # gate_vec is a 1-D tensor of size n_edge_types_for_dst
                for i, g in enumerate(gate_vec.numpy()):
                    k = f"et_{i}"
                    gate_accum[dst_type].setdefault(k, []).append(float(g))

        avg_gates = {
            dst: {k: float(np.mean(v)) for k, v in ets.items()}
            for dst, ets in gate_accum.items()
        }

        # -- LR pair scores from (gene, interacts, gene) attention -----
        lr_gene_et = ("gene", "interacts", "gene")
        lr_pair_edge_scores = None
        if lr_gene_et in edge_scores:
            lr_pair_edge_scores = edge_scores[lr_gene_et]
            log.debug("  LR pair edge scores: %d edges, mean=%.4f, std=%.4f",
                      len(lr_pair_edge_scores),
                      lr_pair_edge_scores.mean(),
                      lr_pair_edge_scores.std())
        else:
            log.warning("  (gene, interacts, gene) edges not found in attention output")

        # -- Cell-cell channel attention -------------------------------
        cc_edge_types = [et for et in edge_types if et[0] == "cell" and et[2] == "cell"]
        cell_cell_edge_scores = {et[1]: edge_scores[et]
                                  for et in cc_edge_types if et in edge_scores}

        # -- Summary logging -------------------------------------------
        log.debug("CCC Extraction: %d edge types, %d layers", len(edge_scores), n_layers)

        return {
            "edge_scores":            edge_scores,
            "gate_weights":           avg_gates,
            "attention_raw":          attention_raw,
            "lr_pair_edge_scores":    lr_pair_edge_scores,
            "cell_cell_edge_scores":  cell_cell_edge_scores,
            "edge_types":             edge_types,
        }

    # -----------------------------------------------------------------
    # LR pair scoring  (direct from gene-gene attention)
    # -----------------------------------------------------------------

    def get_lr_pair_scores(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
    ) -> Dict[Tuple[str, str], float]:
        """
        Map per-edge attention scores to LR pair scores.

        The (gene, interacts, gene) edge order matches lr_pair_vocab
        (same order as built by EdgeBuilder.build_gene_interaction_edges).

        score(LR_k) = attention on edge k  (mean across heads and layers)

        Args:
            extraction:     output of extract()
            lr_pair_vocab:  list of (ligand, receptor) tuples in edge order

        Returns:
            {(ligand, receptor): score}  sorted descending
        """
        scores = extraction.get("lr_pair_edge_scores")
        if scores is None:
            log.warning("No LR pair edge scores available")
            return {}

        n = min(len(scores), len(lr_pair_vocab))
        lr_scores = {lr_pair_vocab[i]: float(scores[i]) for i in range(n)}

        # Sort descending
        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))

        log.debug("LR pair scores: %d pairs, top-3: %s",
                 len(lr_scores),
                 list(lr_scores.items())[:3])

        return lr_scores

    # -----------------------------------------------------------------
    # Cell-type communication matrix  (from cell-cell attention)
    # -----------------------------------------------------------------

    def get_cell_communication_matrix(
        self,
        extraction: Dict,
        cell_types: torch.Tensor,
        cell_type_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Build cell-type x cell-type communication matrix per channel.

        Args:
            extraction:       output of extract()
            cell_types:       [N] integer cell type labels
            cell_type_names:  optional list of string names for types

        Returns:
            {channel_name: [n_types, n_types] numpy array}
        """
        ct   = cell_types.cpu().numpy()
        n_types = int(ct.max()) + 1

        matrices = {}
        for relation, scores in extraction["cell_cell_edge_scores"].items():
            # Find edge index for this edge type
            for et in extraction["edge_types"]:
                if et[1] == relation and et[0] == "cell" and et[2] == "cell":
                    ei = self.data[et].edge_index.cpu().numpy()
                    break
            else:
                continue

            src_types = ct[ei[0]]
            dst_types = ct[ei[1]]

            mat   = np.zeros((n_types, n_types))
            count = np.zeros((n_types, n_types))

            for s, d, score in zip(src_types, dst_types, scores):
                mat[s, d]   += score
                count[s, d] += 1

            count = np.maximum(count, 1)
            matrices[relation] = mat / count

        return matrices

    def get_lr_pair_scores_raw_filtered(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        filter_homodimers: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        """
        Raw attention scores with optional homodimer filtering + expression tie-breaking.

        Homodimers (lig == rec) always receive cosine_sim=1.0 and often attn=1.0
        because the embedding of X is identical to itself and local softmax saturates
        for degree-1 receptor nodes. Filtering them removes known artifacts.

        Tie-breaking: within tied attention groups (e.g. all at 1.0), pairs are
        re-ranked by sqrt(mean_expr_lig x mean_expr_rec) from cell-gene expression
        edges. Unexpressed gene pairs get lower effective score.

        Args:
            extraction:         output of extract()
            lr_pair_vocab:      list of (ligand, receptor) tuples in edge order
            filter_homodimers:  if True, remove pairs where ligand == receptor

        Returns:
            {(ligand, receptor): score} sorted descending
        """
        raw_scores = extraction.get("lr_pair_edge_scores")
        if raw_scores is None:
            log.warning("No LR pair edge scores available")
            return {}

        n = min(len(raw_scores), len(lr_pair_vocab))

        # -- Gene expression from (cell, expresses, gene) edges --------
        cell_gene_et = ("cell", "expresses", "gene")
        n_gene_nodes = int(self.data["gene"].x.shape[0])
        mean_gene_expr = np.zeros(n_gene_nodes, dtype=np.float32)
        try:
            cg_ei   = self.data[cell_gene_et].edge_index.cpu().numpy()
            cg_attr = self.data[cell_gene_et].edge_attr.cpu().numpy()
            expr_col = cg_attr[:, 0]
            gene_sum   = np.zeros(n_gene_nodes, dtype=np.float64)
            gene_count = np.zeros(n_gene_nodes, dtype=np.float64)
            np.add.at(gene_sum,   cg_ei[1], expr_col)
            np.add.at(gene_count, cg_ei[1], 1.0)
            mask = gene_count > 0
            mean_gene_expr[mask] = (gene_sum[mask] / gene_count[mask]).astype(np.float32)
        except Exception as e:
            log.warning("Could not compute gene expression for tie-breaking: %s", e)

        lr_gene_et = ("gene", "interacts", "gene")
        gene_gene_ei = self.data[lr_gene_et].edge_index.cpu()
        lig_node_idx = gene_gene_ei[0, :n].numpy()
        rec_node_idx = gene_gene_ei[1, :n].numpy()

        # -- Assemble scores -------------------------------------------
        pairs_with_scores = []
        n_homodimer = 0
        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                n_homodimer += 1
                continue
            attn = float(raw_scores[i])
            expr_prod = float(np.sqrt(max(mean_gene_expr[lig_node_idx[i]] *
                                         mean_gene_expr[rec_node_idx[i]], 0.0)))
            # Combined score: attn as primary, expr as tie-breaker
            # Encoding: primary key = round(attn,6), secondary = expr_prod
            combined_key = attn * 1e6 + expr_prod  # ensures primary ordering preserved
            pairs_with_scores.append(((lig_u, rec_u), attn, expr_prod, combined_key))

        log.debug("raw_filtered: %d pairs (removed %d homodimers)", len(pairs_with_scores), n_homodimer)

        if not pairs_with_scores:
            return {}

        pairs_with_scores.sort(key=lambda x: -x[3])
        max_key = pairs_with_scores[0][3]

        lr_scores = {}
        for (pair, attn, expr_prod, key) in pairs_with_scores:
            lr_scores[pair] = key / max_key if max_key > 0 else 0.0

        return lr_scores

    # -----------------------------------------------------------------
    # Enhanced LR pair scoring  (fixes score saturation)
    # -----------------------------------------------------------------

    def get_lr_pair_scores_enhanced(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        mode: str = "combined",
    ) -> Dict[Tuple[str, str], float]:
        """
        Enhanced LR pair scoring to fix score saturation caused by degree-1 receptor nodes.

        Root cause: TransformerConv softmax is local -- if a receptor gene node has only one
        incoming LR edge, that edge always gets attention=1.0 regardless of biology.
        This means 11% of pairs are tied at 1.0, destroying ranking.

        Fix strategies:
          "degree":     score = attn x log1p(receptor_in_degree)
                        Downweights unique (degree-1) receptors; rare pairs should score less.
          "expr":       score = attn x sqrt(mean_expr_lig x mean_expr_rec)
                        Pairs involving unexpressed genes are penalized.
          "combined":   score = attn x log1p(receptor_in_degree) x sqrt(expr_lig x expr_rec)
                        Both corrections combined (recommended).

        Gene expression is estimated from (cell, expresses, gene) edge attributes (col 0 = expr_norm).
        Degree is computed from (gene, interacts, gene) edge_index (in-degree of receptor node).

        Args:
            extraction:    output of extract()
            lr_pair_vocab: list of (ligand, receptor) tuples in edge order
            mode:          "degree" | "expr" | "combined"

        Returns:
            {(ligand, receptor): enhanced_score}  sorted descending
        """
        raw_scores = extraction.get("lr_pair_edge_scores")
        if raw_scores is None:
            log.warning("No LR pair edge scores available")
            return {}

        n = min(len(raw_scores), len(lr_pair_vocab))

        # -- 1. Receptor in-degree in LR graph -------------------------
        lr_gene_et = ("gene", "interacts", "gene")
        gene_gene_ei = self.data[lr_gene_et].edge_index.cpu()  # [2, E_gg]
        receptor_nodes = gene_gene_ei[1].numpy()               # destination = receptor
        n_gene_nodes = int(self.data["gene"].x.shape[0])

        in_degree = np.zeros(n_gene_nodes, dtype=np.float32)
        for idx in receptor_nodes:
            in_degree[idx] += 1.0

        # Map: for each LR edge i, get receptor node index -> in-degree
        ligand_node_idx   = gene_gene_ei[0, :n].numpy()
        receptor_node_idx = gene_gene_ei[1, :n].numpy()

        log.debug("In-degree stats: min=%.0f  max=%.0f  mean=%.2f  (degree-1 nodes: %d/%.0f)",
                 in_degree[in_degree > 0].min(), in_degree.max(), in_degree[in_degree > 0].mean(),
                 int((in_degree[receptor_node_idx] == 1).sum()), float(n))

        # -- 2. Gene expression from (cell, expresses, gene) edges -----
        cell_gene_et = ("cell", "expresses", "gene")
        mean_gene_expr = np.zeros(n_gene_nodes, dtype=np.float32)

        try:
            cg_ei   = self.data[cell_gene_et].edge_index.cpu().numpy()  # [2, E_cg]
            cg_attr = self.data[cell_gene_et].edge_attr.cpu().numpy()   # [E_cg, >=1]
            expr_col = cg_attr[:, 0]  # expr_norm

            # Accumulate per gene
            gene_sum   = np.zeros(n_gene_nodes, dtype=np.float64)
            gene_count = np.zeros(n_gene_nodes, dtype=np.float64)
            np.add.at(gene_sum,   cg_ei[1], expr_col)
            np.add.at(gene_count, cg_ei[1], 1.0)
            mask = gene_count > 0
            mean_gene_expr[mask] = (gene_sum[mask] / gene_count[mask]).astype(np.float32)
            log.debug("Gene expr from cell->gene edges: %d genes with expression",
                     int(mask.sum()))
        except Exception as e:
            log.warning("Could not compute gene expression weights: %s", e)

        # -- 3. Compute enhanced scores --------------------------------
        raw = raw_scores[:n]
        rec_degree = in_degree[receptor_node_idx]   # [n]
        expr_lig   = mean_gene_expr[ligand_node_idx]
        expr_rec   = mean_gene_expr[receptor_node_idx]
        expr_prod  = np.sqrt(np.maximum(expr_lig * expr_rec, 0.0))

        if mode == "degree":
            enhanced = raw * np.log1p(rec_degree)
        elif mode == "linear_degree":
            # Exact de-normalization: attn x in_degree ~ exp(logit) / mean_exp_logit
            # More precise than log1p -- directly undoes the local softmax normalization.
            # A degree-1 receptor gets score=attn*1; a degree-20 receptor's top pair
            # gets score~0.3*20=6.0, which correctly promotes it over trivial degree-1 pairs.
            enhanced = raw * rec_degree
        elif mode == "log_degree":
            # Log-space degree correction: log(attn) + log(degree) ~ pre-softmax logit
            # This is the theoretically correct approximation of the attention logit before
            # the local softmax normalization. Converts softmax weights back to logit space
            # where cross-receptor comparison is meaningful.
            log_attn = np.log(np.maximum(raw, 1e-8))
            log_deg  = np.log(np.maximum(rec_degree, 1.0))
            enhanced = log_attn + log_deg  # This is the logit approximation
            # Shift to positive range: add the minimum to make all values >= 0
            enhanced = enhanced - enhanced.min()
        elif mode == "expr":
            enhanced = raw * expr_prod
        elif mode == "combined":
            enhanced = raw * np.log1p(rec_degree) * expr_prod
        elif mode == "linear_combined":
            # linear_degree + expr: strongest correction
            enhanced = raw * rec_degree * expr_prod
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'degree', 'linear_degree', 'log_degree', 'expr', 'combined', 'linear_combined'.")

        # -- 4. Normalize to [0, 1] ------------------------------------
        emax = enhanced.max()
        if emax > 0:
            enhanced = enhanced / emax

        # -- 5. Assemble result ----------------------------------------
        lr_scores = {lr_pair_vocab[i]: float(enhanced[i]) for i in range(n)}
        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))

        # Saturation report
        tied_top = sum(1 for v in lr_scores.values() if v >= 0.999)
        log.debug("Enhanced (%s) LR scores: %d pairs, tied@top: %d  (was %d raw)",
                 mode, n,
                 tied_top,
                 int((raw_scores[:n] >= 0.999).sum()))

        log.info("Top-5 enhanced pairs: %s", list(lr_scores.items())[:5])
        return lr_scores

    # -----------------------------------------------------------------
    # Cosine-similarity LR scoring (fixes local-softmax bias)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def get_lr_pair_scores_embedding(
        self,
        lr_pair_vocab: List[Tuple[str, str]],
        gene_name_to_idx: Optional[Dict[str, int]] = None,
    ) -> Dict[Tuple[str, str], float]:
        """
        Score LR pairs by cosine similarity of GNN-learned gene embeddings.

        Fixes the local-softmax bias in attention-based scoring where
        high-degree hub receptors (CD44, ITGB1 etc.) are systematically
        under-scored because softmax normalizes per receptor.

        Cosine similarity uses the FULL gene embedding learned by the GNN
        after message passing through all edge types -- this captures
        which genes co-appear in similar spatial contexts.

        Args:
            lr_pair_vocab:      list of (ligand, receptor) string pairs
            gene_name_to_idx:   optional {gene_name: gene_node_idx} map;
                                loaded from graph metadata if not provided

        Returns:
            {(ligand_upper, receptor_upper): cosine_similarity_score}
            Scores in [-1, 1]; higher = more similar embeddings.
            Shifted to [0, 1] by (score + 1) / 2 for compatibility.
        """
        self.model.eval()
        data = self.data.to(self.device)

        # Build gene name -> node index map
        if gene_name_to_idx is None:
            # Try graph metadata (stored as gene_vocab list)
            meta_gene_vocab = getattr(data.get("gene", data), "gene_vocab", None)
            if meta_gene_vocab is None:
                # Fall back: gene names not available in graph -- use index order
                n_genes = data["gene"].x.shape[0]
                gene_name_to_idx = {}   # empty -> will skip unknown genes
                log.warning("gene_name_to_idx not provided and not in graph; "
                            "pass gene_name_to_idx explicitly")
            else:
                gene_name_to_idx = {g.upper(): i for i, g in enumerate(meta_gene_vocab)}

        # Extract graph inputs
        from mosanic.models import MOSANIC
        x_dict, edge_index_dict, edge_attr_dict = self.model._extract_graph_data(data)

        # Run forward_all_embeddings to get gene node embeddings
        emb_dict = self.model.encoder.forward_all_embeddings(
            x_dict, edge_index_dict, edge_attr_dict
        )
        gene_emb = emb_dict["gene"].cpu()  # [G, hidden_dim]

        # L2-normalize for cosine similarity
        gene_emb_norm = torch.nn.functional.normalize(gene_emb, dim=-1)

        # Score each LR pair
        lr_scores: Dict[Tuple[str, str], float] = {}
        missing = 0
        for lig, rec in lr_pair_vocab:
            lig_u = str(lig).upper()
            rec_u = str(rec).upper()
            i = gene_name_to_idx.get(lig_u)
            j = gene_name_to_idx.get(rec_u)
            if i is None or j is None:
                missing += 1
                continue
            cos_sim = float(torch.dot(gene_emb_norm[i], gene_emb_norm[j]))
            # Shift [-1,1] -> [0,1]
            lr_scores[(lig_u, rec_u)] = (cos_sim + 1.0) / 2.0

        log.info("Embedding cosine scores: %d pairs scored, %d missing gene index",
                 len(lr_scores), missing)
        if lr_scores:
            vals = list(lr_scores.values())
            log.info("  score range [%.4f, %.4f]  mean=%.4f",
                     min(vals), max(vals), sum(vals) / len(vals))

        return lr_scores

    def get_lr_pair_scores_last_layer(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        layer_idx: int = -1,
        filter_homodimers: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        """
        LR pair scores using attention from a specific layer (default: last layer).

        Motivation: the last GNN layer may be more discriminative because it captures
        the full multi-hop context (gene signal propagated through cell->gene->gene->cell).
        The mean-over-layers aggregation in get_lr_pair_scores() dilutes this refined signal
        with the noisier first-layer attention.

        Args:
            extraction:         output of extract()
            lr_pair_vocab:      list of (ligand, receptor) tuples in edge order
            layer_idx:          which layer to use (-1 = last, 0 = first)
            filter_homodimers:  remove pairs where ligand == receptor

        Returns:
            {(ligand, receptor): score} sorted descending
        """
        lr_gene_et = ("gene", "interacts", "gene")
        attn_raw = extraction.get("attention_raw", {})
        if lr_gene_et not in attn_raw:
            log.warning("(gene,interacts,gene) not in attention_raw -- falling back to mean")
            return self.get_lr_pair_scores(extraction, lr_pair_vocab)

        stacked = attn_raw[lr_gene_et]  # [E_t, n_heads, n_layers]
        n_layers = stacked.shape[-1]
        idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        layer_attn = stacked[:, :, idx].mean(dim=-1).numpy()  # [E_t] mean over heads

        n = min(len(layer_attn), len(lr_pair_vocab))
        lr_scores = {}
        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                continue
            lr_scores[(lig_u, rec_u)] = float(layer_attn[i])

        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))
        log.info("last_layer (layer=%d): %d pairs scored", idx, len(lr_scores))
        return lr_scores

    def get_lr_pair_scores_last_layer_degree(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        layer_idx: int = -1,
        degree_mode: str = "linear",
        filter_homodimers: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        """
        Last-layer attention x degree correction.

        Combines the most discriminative signal (last-layer attention) with
        exact de-normalization of the local softmax (linear degree correction).

        This is the recommended scoring variant for publication:
          score(L->R) = attn_last_layer(L->R) x in_degree(R)
          Normalise to [0, 1].

        Motivation:
          - Last-layer attention is most discriminative
          - Linear degree x attn ~ pre-softmax logit: corrects degree-1 receptor bias
          - Together: best absolute AUPR expected

        Args:
            extraction:         output of extract()
            lr_pair_vocab:      list of (ligand, receptor) tuples in edge order
            layer_idx:          -1 = last, 0 = first layer
            degree_mode:        "linear" = attn*in_degree | "log" = attn*log1p(in_degree)
            filter_homodimers:  remove lig==rec pairs

        Returns:
            {(ligand, receptor): score} sorted descending
        """
        lr_gene_et = ("gene", "interacts", "gene")
        attn_raw = extraction.get("attention_raw", {})
        if lr_gene_et not in attn_raw:
            log.warning("(gene,interacts,gene) not in attention_raw")
            return self.get_lr_pair_scores(extraction, lr_pair_vocab)

        stacked = attn_raw[lr_gene_et]  # [E_t, n_heads, n_layers]
        n_layers = stacked.shape[-1]
        idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
        layer_attn = stacked[:, :, idx].mean(dim=-1).numpy()  # [E_t]

        # -- Receptor in-degree ----------------------------------------
        gene_gene_ei = self.data[lr_gene_et].edge_index.cpu()
        receptor_nodes = gene_gene_ei[1].numpy()
        n_gene_nodes = int(self.data["gene"].x.shape[0])
        in_degree = np.zeros(n_gene_nodes, dtype=np.float32)
        for r_idx in receptor_nodes:
            in_degree[r_idx] += 1.0

        n = min(len(layer_attn), len(lr_pair_vocab))
        rec_node_idx = gene_gene_ei[1, :n].numpy()
        rec_degree = in_degree[rec_node_idx]

        if degree_mode == "linear":
            enhanced = layer_attn[:n] * rec_degree
        else:  # log
            enhanced = layer_attn[:n] * np.log1p(rec_degree)

        emax = enhanced.max()
        if emax > 0:
            enhanced = enhanced / emax

        lr_scores = {}
        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                continue
            lr_scores[(lig_u, rec_u)] = float(enhanced[i])

        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))

        tied_top = sum(1 for v in lr_scores.values() if v >= 0.999)
        log.info("last_layer_degree (layer=%d, mode=%s): %d pairs, tied@top=%d",
                 idx, degree_mode, len(lr_scores), tied_top)
        return lr_scores

    def get_lr_pair_scores_expressed(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        min_expr_frac: float = 0.05,
        scoring_mode: str = "raw",
        filter_homodimers: bool = True,
    ) -> Dict[Tuple[str, str], float]:
        """
        LR pair scores restricted to pairs where BOTH genes are expressed.

        Filters out unexpressed gene pairs to reduce false positives.
        Among expressed pairs, ranks by raw attention.

        Args:
            extraction:         output of extract()
            lr_pair_vocab:      list of (ligand, receptor) tuples in edge order
            min_expr_frac:      minimum fraction of cells that must express both genes
            scoring_mode:       "raw" | "raw_filtered" | "last_layer"
            filter_homodimers:  remove lig==rec pairs

        Returns:
            {(ligand, receptor): score} -- only expressed pairs
        """
        # Get expression fractions from (cell, expresses, gene) edges
        cell_gene_et = ("cell", "expresses", "gene")
        n_gene_nodes = int(self.data["gene"].x.shape[0])
        n_cells = int(self.data["cell"].x.shape[0])
        gene_expr_frac = np.zeros(n_gene_nodes, dtype=np.float32)

        try:
            cg_ei = self.data[cell_gene_et].edge_index.cpu().numpy()
            # Each cell->gene edge means that cell expresses that gene
            gene_cell_count = np.zeros(n_gene_nodes, dtype=np.float64)
            np.add.at(gene_cell_count, cg_ei[1], 1.0)
            gene_expr_frac = (gene_cell_count / max(n_cells, 1)).astype(np.float32)
            log.info("Expressed genes (frac>=%.2f): %d / %d",
                     min_expr_frac, int((gene_expr_frac >= min_expr_frac).sum()), n_gene_nodes)
        except Exception as e:
            log.warning("Could not compute expression fractions: %s", e)
            return self.get_lr_pair_scores(extraction, lr_pair_vocab)

        # Get base scores
        raw_scores = extraction.get("lr_pair_edge_scores")
        if raw_scores is None:
            return {}

        lr_gene_ei = self.data[("gene", "interacts", "gene")].edge_index.cpu()
        n = min(len(raw_scores), len(lr_pair_vocab))
        lig_node_idx = lr_gene_ei[0, :n].numpy()
        rec_node_idx = lr_gene_ei[1, :n].numpy()

        lr_scores = {}
        n_filtered = 0
        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                continue
            # Check expression threshold
            frac_lig = gene_expr_frac[lig_node_idx[i]]
            frac_rec = gene_expr_frac[rec_node_idx[i]]
            if frac_lig < min_expr_frac or frac_rec < min_expr_frac:
                n_filtered += 1
                continue
            lr_scores[(lig_u, rec_u)] = float(raw_scores[i])

        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))
        log.info("expressed_raw: %d pairs after filtering %d unexpressed", len(lr_scores), n_filtered)
        return lr_scores

    # -----------------------------------------------------------------
    # Spatial enrichment (for DES metric)
    # -----------------------------------------------------------------

    def get_spatial_attention_scores(
        self,
        extraction: Dict,
        coords_um: np.ndarray,
        channel: str = "all",
        exclude_self_loops: bool = True,
        exclude_channels: Optional[List[str]] = None,
    ) -> Dict:
        """
        Return per cell-pair (distance, attention) tuples for DES evaluation.

        IMPORTANT: By default, self-loops (intracellular edges with src==dst, dist=0)
        are EXCLUDED. Including them inflates DES artificially: self-loops always
        have distance=0 and receive moderately high attention, creating a spurious
        correlation between attention and proximity.

        Args:
            extraction:         output of extract()
            coords_um:          [N, 2] spatial coordinates in um
            channel:            which cell-cell channel ('all' = sum of spatial channels)
            exclude_self_loops: if True (default), exclude edges where src == dst
            exclude_channels:   list of channel names to exclude (default: ['intracellular'])

        Returns:
            {'cell_pairs': (src_idx, dst_idx), 'distances': [E], 'scores': [E]}
        """
        if exclude_channels is None:
            exclude_channels = ["intracellular"]   # exclude by default -- see note above

        cell_cell_scores = extraction["cell_cell_edge_scores"]

        if channel == "all":
            # Sum across spatial cell-cell channels only (skip intracellular by default)
            all_scores: Dict[Tuple[int, int], float] = {}
            all_dist:   Dict[Tuple[int, int], float] = {}

            for relation in cell_cell_scores:
                if relation in exclude_channels:
                    continue
                for et in extraction["edge_types"]:
                    if et[1] == relation and et[0] == "cell" and et[2] == "cell":
                        ei = self.data[et].edge_index.cpu().numpy()
                        sc = cell_cell_scores[relation]
                        for i in range(len(sc)):
                            s_i, d_i = int(ei[0, i]), int(ei[1, i])
                            if exclude_self_loops and s_i == d_i:
                                continue
                            d_um = float(np.linalg.norm(coords_um[s_i] - coords_um[d_i]))
                            pair = (s_i, d_i)
                            all_scores[pair] = all_scores.get(pair, 0.0) + float(sc[i])
                            all_dist[pair] = d_um
                        break
        else:
            all_scores = {}
            all_dist   = {}
            if channel in cell_cell_scores:
                sc = cell_cell_scores[channel]
                for et in extraction["edge_types"]:
                    if et[1] == channel and et[0] == "cell" and et[2] == "cell":
                        ei = self.data[et].edge_index.cpu().numpy()
                        for i in range(len(sc)):
                            s_i, d_i = int(ei[0, i]), int(ei[1, i])
                            if exclude_self_loops and s_i == d_i:
                                continue
                            d_um = float(np.linalg.norm(coords_um[s_i] - coords_um[d_i]))
                            all_scores[(s_i, d_i)] = float(sc[i])
                            all_dist[(s_i, d_i)]   = d_um
                        break

        pairs     = list(all_scores.keys())
        scores_np = np.array([all_scores[p] for p in pairs], dtype=np.float32)
        dists_np  = np.array([all_dist[p]   for p in pairs], dtype=np.float32)

        return {
            "cell_pairs": pairs,
            "distances":  dists_np,
            "scores":     scores_np,
        }

    # -----------------------------------------------------------------
    # Cell-spatial LIGREC scoring
    # -----------------------------------------------------------------

    def get_lr_pair_scores_cell_spatial(
        self,
        extraction: Dict,
        lr_pair_vocab: List[Tuple[str, str]],
        gene_name_to_idx: Optional[Dict[str, int]] = None,
        channels: List[str] = ("contact", "secreted"),
        filter_homodimers: bool = True,
        alpha: float = 0.5,
    ) -> Dict[Tuple[str, str], float]:
        """
        Cell-spatial LIGREC score: combines gene-gene attention with cell-level
        spatial expression evidence.

        Motivation: gene-gene attention is discriminative WITHIN a receptor's ligand
        pool but cannot compare ACROSS receptors with different in-degrees. This score
        uses a complementary signal:

          ligrec(L, R) = sum_{(i,j) in cell_pairs} cell_attn(i,j) x expr(i,L) x expr(j,R)

        Where:
          - cell_attn(i,j) = summed attention over contact + secreted cell-cell edges
          - expr(i, L) = expression of ligand L in cell i (from cell-gene edges)
          - expr(j, R) = expression of receptor R in cell j (from cell-gene edges)

        This captures: if cells expressing L tend to be spatially close AND highly attended
        to cells expressing R -> (L, R) is an active CCC pair.

        Final score: alpha x gene_gene_attn + (1-alpha) x ligrec_normalized
        (alpha=0.5 by default; alpha=0.0 = pure ligrec; alpha=1.0 = pure attn)

        Complexity: O(n_pairs x N) where N = n_cells -- moderate, cached per call.

        Args:
            extraction:       output of extract()
            lr_pair_vocab:    list of (ligand, receptor) tuples in edge order
            gene_name_to_idx: {gene_name_upper: gene_node_idx}; loaded from graph if None
            channels:         cell-cell channels to sum for cell_attn
            filter_homodimers: remove lig==rec pairs
            alpha:            weight for gene-gene attention (1-alpha for ligrec)

        Returns:
            {(ligand, receptor): combined_score} sorted descending
        """
        lr_gene_et = ("gene", "interacts", "gene")
        raw_scores = extraction.get("lr_pair_edge_scores")
        if raw_scores is None:
            log.warning("No LR pair edge scores -- returning empty")
            return {}

        n = min(len(raw_scores), len(lr_pair_vocab))
        n_cells = int(self.data["cell"].x.shape[0])
        n_gene_nodes = int(self.data["gene"].x.shape[0])

        # -- 1. Build cell-pair attention matrix (sparse, per channel sum)
        # cell_attn[i] = array of (j, attn_sum) for each neighbor j of cell i
        # Represented as two arrays: src, dst, score
        cc_src_list, cc_dst_list, cc_score_list = [], [], []
        for ch in channels:
            for et in extraction["edge_types"]:
                if et[0] == "cell" and et[1] == ch and et[2] == "cell":
                    ei = self.data[et].edge_index.cpu().numpy()  # [2, E]
                    sc = extraction["cell_cell_edge_scores"].get(ch)
                    if sc is not None and len(sc) > 0:
                        cc_src_list.append(ei[0])
                        cc_dst_list.append(ei[1])
                        cc_score_list.append(sc.astype(np.float32))
                    break

        if not cc_src_list:
            log.warning("No cell-cell edge scores for channels %s -- using raw only", channels)
            return self.get_lr_pair_scores(extraction, lr_pair_vocab)

        cc_src   = np.concatenate(cc_src_list)
        cc_dst   = np.concatenate(cc_dst_list)
        cc_score = np.concatenate(cc_score_list)
        # Remove self-loops
        mask = cc_src != cc_dst
        cc_src, cc_dst, cc_score = cc_src[mask], cc_dst[mask], cc_score[mask]
        log.info("Cell-pair attention: %d edges (channels: %s)", len(cc_src), list(channels))

        # -- 2. Build per-cell gene expression arrays -------------------
        # cell_gene_expr[c, g] = mean expr_norm from (cell, expresses, gene) edges
        # Stored sparse: gene -> (cell_ids array, expr_vals array)
        cell_gene_et = ("cell", "expresses", "gene")
        gene_to_cells: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        try:
            cg_ei   = self.data[cell_gene_et].edge_index.cpu().numpy()  # [2, E_cg]
            cg_attr = self.data[cell_gene_et].edge_attr.cpu().numpy()   # [E_cg, >=1]
            expr_col = cg_attr[:, 0].astype(np.float32)  # expr_norm
            # Group by gene
            from collections import defaultdict
            _cell_ids_by_gene = defaultdict(list)
            _expr_by_gene = defaultdict(list)
            for edge_idx in range(len(cg_ei[0])):
                c_id = int(cg_ei[0, edge_idx])
                g_id = int(cg_ei[1, edge_idx])
                _cell_ids_by_gene[g_id].append(c_id)
                _expr_by_gene[g_id].append(expr_col[edge_idx])
            for g_id in _cell_ids_by_gene:
                gene_to_cells[g_id] = (
                    np.array(_cell_ids_by_gene[g_id], dtype=np.int32),
                    np.array(_expr_by_gene[g_id], dtype=np.float32),
                )
            log.info("Cell-gene expression: %d genes with expression data", len(gene_to_cells))
        except Exception as e:
            log.warning("Cannot build cell-gene expression map: %s", e)
            return self.get_lr_pair_scores(extraction, lr_pair_vocab)

        # -- 3. Build gene name -> node index map -----------------------
        if gene_name_to_idx is None:
            meta_gene_vocab = getattr(self.data.get("gene", self.data), "gene_vocab", None)
            if meta_gene_vocab is not None:
                gene_name_to_idx = {g.upper(): i for i, g in enumerate(meta_gene_vocab)}
            else:
                gene_name_to_idx = {}

        # For each LR pair, also get gene node indices from the edge
        gene_gene_ei = self.data[lr_gene_et].edge_index.cpu().numpy()  # [2, E_gg]

        # -- 4. Build cell-pair src->dst sparse index for fast lookup ---
        # For each source cell i, store which destination cells j have an edge, and score
        from collections import defaultdict
        src_to_neighbors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        _nbr_dst = defaultdict(list)
        _nbr_sc  = defaultdict(list)
        for ei_idx in range(len(cc_src)):
            s = int(cc_src[ei_idx])
            _nbr_dst[s].append(int(cc_dst[ei_idx]))
            _nbr_sc[s].append(float(cc_score[ei_idx]))
        for s in _nbr_dst:
            src_to_neighbors[s] = (
                np.array(_nbr_dst[s], dtype=np.int32),
                np.array(_nbr_sc[s],  dtype=np.float32),
            )

        # For fast lookup: cell expression by gene -> cell_expr[cell_id] (dict)
        gene_cell_expr: Dict[int, Dict[int, float]] = {}
        for g_id, (cell_ids, exprs) in gene_to_cells.items():
            gene_cell_expr[g_id] = {int(c): float(e) for c, e in zip(cell_ids, exprs)}

        # -- 5. Compute LIGREC scores for each LR pair -----------------
        ligrec_scores = np.zeros(n, dtype=np.float32)
        n_homodimers = 0
        log.info("Computing cell-spatial LIGREC for %d LR pairs ...", n)

        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                n_homodimers += 1
                continue

            # Gene node indices from edge
            lig_g_idx = int(gene_gene_ei[0, i])
            rec_g_idx = int(gene_gene_ei[1, i])

            lig_cell_expr = gene_cell_expr.get(lig_g_idx, {})
            rec_cell_expr = gene_cell_expr.get(rec_g_idx, {})

            if not lig_cell_expr or not rec_cell_expr:
                continue  # No expression data -> score stays 0

            # Iterate over cells expressing ligand
            total = 0.0
            for c_lig, e_lig in lig_cell_expr.items():
                if c_lig not in src_to_neighbors:
                    continue
                nbr_dst, nbr_sc = src_to_neighbors[c_lig]
                # For each neighbor, check if it expresses receptor
                for j_idx in range(len(nbr_dst)):
                    c_rec = int(nbr_dst[j_idx])
                    e_rec = rec_cell_expr.get(c_rec, 0.0)
                    if e_rec > 0.0:
                        total += float(nbr_sc[j_idx]) * float(e_lig) * float(e_rec)

            ligrec_scores[i] = total

        log.info("LIGREC computed. Non-zero pairs: %d / %d (homodimers: %d)",
                 int((ligrec_scores > 0).sum()), n, n_homodimers)

        # -- 6. Normalize each signal to [0, 1] and combine ------------
        attn_arr = np.array(raw_scores[:n], dtype=np.float32)
        attn_max = attn_arr.max()
        if attn_max > 0:
            attn_norm = attn_arr / attn_max
        else:
            attn_norm = attn_arr

        ligrec_max = ligrec_scores.max()
        if ligrec_max > 0:
            ligrec_norm = ligrec_scores / ligrec_max
        else:
            ligrec_norm = ligrec_scores

        combined = alpha * attn_norm + (1.0 - alpha) * ligrec_norm

        # -- 7. Assemble result ----------------------------------------
        lr_scores: Dict[Tuple[str, str], float] = {}
        for i in range(n):
            lig, rec = lr_pair_vocab[i]
            lig_u, rec_u = str(lig).upper(), str(rec).upper()
            if filter_homodimers and lig_u == rec_u:
                continue
            lr_scores[(lig_u, rec_u)] = float(combined[i])

        lr_scores = dict(sorted(lr_scores.items(), key=lambda x: -x[1]))

        # Log top pairs
        log.info("cell_spatial (alpha=%.2f): %d pairs, top-5: %s",
                 alpha, len(lr_scores), list(lr_scores.items())[:5])
        return lr_scores
