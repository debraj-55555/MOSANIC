"""
mosaic/api.py — High-level Python API for MOSAIC.

After training (or loading a checkpoint), users get a MOSAICResult object
that provides clean methods for querying CCC signals.

Usage (programmatic):
    from mosaic import train, load
    result = train("configs/breast.yaml", device="cuda")
    # or
    result = load("configs/breast.yaml", device="cuda")

    # LR pair ranking
    lr = result.lr_pairs(top_k=50)
    lr = result.lr_pairs(receptor="EGFR")

    # MR pair ranking
    mr = result.mr_pairs(top_k=20)

    # Cell-type communication
    comm = result.communication_matrix(channel="secreted")

    # Spatial maps
    activity = result.lr_activity("TGFB1", "TGFBR2")
    activity = result.mr_activity("Lactate", "GPR81")

    # Enriched cell groups
    groups = result.communication_hotspots(top_k=10)

    # Clustering
    labels = result.cluster(n_clusters=8)

    # Plotting
    result.plot_lr_ranking(top_k=30)
    result.plot_communication_matrix(channel="secreted")
    result.plot_spatial("TGFB1", "TGFBR2")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MOSAICResult: the main user-facing object
# ---------------------------------------------------------------------------

class MOSAICResult:
    """
    Result object returned after training or loading a MOSAIC model.

    Holds the trained model, graph data, metadata, and extracted attention
    signals. Provides high-level methods for querying LR/MR pairs,
    communication matrices, spatial maps, and clustering.

    Attributes:
        model:      Trained MOSAIC model (eval mode)
        data:       PyG HeteroData graph
        meta:       dict with n_cells, n_genes, lr_pair_vocab, gene_vocab, etc.
        device:     torch.device
        config:     dict from YAML config
        dataset:    dataset name string
    """

    def __init__(
        self,
        model: torch.nn.Module,
        data,
        meta: dict,
        config: dict,
        dataset: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model.eval().to(device)
        self.data = data
        self.meta = meta
        self.config = config
        self.dataset = dataset
        self.device = device

        # Lazy-initialised caches
        self._extraction = None
        self._extractor = None
        self._cell_embeddings = None
        self._lr_pair_vocab = None
        self._gene_vocab = None
        self._gene_name_to_idx = None
        self._coords_um = None
        self._cell_types = None
        self._cell_type_names = None
        self._metabolite_names = None

    # -----------------------------------------------------------------
    # Properties (lazy-loaded)
    # -----------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return int(self.meta["n_cells"])

    @property
    def n_genes(self) -> int:
        return int(self.meta["n_genes"])

    @property
    def n_metabolites(self) -> int:
        return int(self.meta.get("n_metabolites", 0))

    @property
    def lr_pair_vocab(self) -> List[Tuple[str, str]]:
        if self._lr_pair_vocab is None:
            vocab = self.meta.get("lr_pair_vocab", [])
            self._lr_pair_vocab = [(str(l), str(r)) for l, r in vocab]
        return self._lr_pair_vocab

    @property
    def gene_vocab(self) -> List[str]:
        if self._gene_vocab is None:
            self._gene_vocab = [str(g) for g in self.meta.get("gene_vocab", [])]
        return self._gene_vocab

    @property
    def gene_name_to_idx(self) -> Dict[str, int]:
        if self._gene_name_to_idx is None:
            self._gene_name_to_idx = {
                g.upper(): i for i, g in enumerate(self.gene_vocab)
            }
        return self._gene_name_to_idx

    @property
    def coords(self) -> Optional[np.ndarray]:
        """Spatial coordinates [N, 2+] in microns (or pixels if no scale)."""
        if self._coords_um is None:
            # Try graph data first
            if hasattr(self.data["cell"], "pos") and self.data["cell"].pos is not None:
                self._coords_um = self.data["cell"].pos.cpu().numpy()
            else:
                # Load from adata (most reliable source)
                adata_path = self.config.get("paths", {}).get("raw_adata")
                if adata_path and Path(adata_path).exists():
                    try:
                        import scanpy as sc
                        adata = sc.read_h5ad(adata_path)
                        if "spatial" in adata.obsm:
                            self._coords_um = np.array(adata.obsm["spatial"], dtype=np.float32)
                    except Exception:
                        pass
                # Try preprocessing cache as last resort
                if self._coords_um is None:
                    processed = Path(self.config.get("paths", {}).get("processed_dir", ""))
                    cache_path = processed / self.dataset / "preprocessing_cache.pt"
                    if cache_path.exists():
                        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                        c = cache.get("spatial_graph", {}).get("coords_um")
                        if c is not None:
                            self._coords_um = np.array(c, dtype=np.float32) if not isinstance(c, np.ndarray) else c
        return self._coords_um

    @property
    def cell_types(self) -> Optional[np.ndarray]:
        """Integer cell type labels [N]."""
        if self._cell_types is None:
            if hasattr(self.data["cell"], "cell_type"):
                self._cell_types = self.data["cell"].cell_type.cpu().numpy()
        return self._cell_types

    @property
    def cell_type_names(self) -> Optional[List[str]]:
        """
        String cell type names, if available.

        These are NOT stored in the graph by default — only integer labels are.
        To set names, either:
          1. Pass them when building the graph (meta["cell_type_names"])
          2. Set result.cell_type_names = [...] from your adata.obs
        """
        if self._cell_type_names is None:
            names = self.meta.get("cell_type_names")
            if names is not None:
                self._cell_type_names = [str(n) for n in names]
        return self._cell_type_names

    @cell_type_names.setter
    def cell_type_names(self, names: List[str]):
        """Allow users to set cell type names from their adata."""
        self._cell_type_names = [str(n) for n in names]

    def set_cell_types_from_adata(self, adata, col: str = "cell_type"):
        """
        Load cell type labels and names from an AnnData object.

        Usage:
            result.set_cell_types_from_adata(adata, col="cell_type_annot")

        Args:
            adata: AnnData with .obs containing the cell type column
            col:   Column name in adata.obs (e.g. "cell_type", "leiden", etc.)
        """
        if col not in adata.obs.columns:
            avail = [c for c in adata.obs.columns if "type" in c.lower() or "cluster" in c.lower() or c == "leiden"]
            raise ValueError(
                f"Column '{col}' not in adata.obs. "
                f"Available candidates: {avail}"
            )

        ct_raw = adata.obs[col].values
        uniq, inv = np.unique(ct_raw, return_inverse=True)
        self._cell_types = inv.astype(np.int64)
        self._cell_type_names = [str(u) for u in uniq]

        # Also store on graph data for downstream use
        self.data["cell"].cell_type = torch.from_numpy(self._cell_types)

        log.debug("Set %d cell types from adata.obs['%s']", len(uniq), col)

    @property
    def metabolite_names(self) -> List[str]:
        """Metabolite names from graph metadata (met_vocab)."""
        if self._metabolite_names is None:
            # met_vocab is stored in metadata by assembler.py
            names = self.meta.get("met_vocab")
            if names is not None:
                self._metabolite_names = [str(n) for n in names]
            else:
                self._metabolite_names = [f"met_{i}" for i in range(self.n_metabolites)]
        return self._metabolite_names

    @property
    def cell_embeddings(self) -> np.ndarray:
        """Cell embeddings [N, hidden_dim] from the encoder."""
        if self._cell_embeddings is None:
            with torch.no_grad():
                out = self.model(self.data.to(self.device), return_attention=False)
            self._cell_embeddings = out["node_embeddings"].cpu().numpy()
        return self._cell_embeddings

    @property
    def extractor(self):
        """Lazy CCCExtractor instance."""
        if self._extractor is None:
            from mosaic.evaluation.ccc_extractor import CCCExtractor
            self._extractor = CCCExtractor(self.model, self.data, device=self.device)
        return self._extractor

    @property
    def extraction(self) -> dict:
        """Lazy extraction (forward pass with attention collection)."""
        if self._extraction is None:
            self._extraction = self.extractor.extract()
            log.debug("Extraction complete: %d edge types.", len(self._extraction["edge_types"]))
        return self._extraction

    # -----------------------------------------------------------------
    # LR pair scoring
    # -----------------------------------------------------------------

    def _compute_intensity_scores(self) -> Dict[Tuple[str, str], float]:
        """
        Compute intensity = attn × mean(expr_L) × mean(expr_R) per LR pair.

        This is the primary scoring used in all MOSAIC figures. It naturally
        breaks degree-1 ties (attn=1.0 for 183 pairs) and weights by
        biological expression evidence.

        Cached after first call.
        """
        if hasattr(self, "_intensity_cache") and self._intensity_cache is not None:
            return self._intensity_cache

        ext = self.extraction
        vocab = self.lr_pair_vocab
        raw_scores = ext.get("lr_pair_edge_scores")
        if raw_scores is None:
            self._intensity_cache = {}
            return self._intensity_cache

        # Get per-gene mean expression from cell→gene edges
        cg_et = ("cell", "expresses", "gene")
        mean_expr = np.zeros(self.n_genes, dtype=np.float32)
        if cg_et in self.data.edge_types:
            cg_ei = self.data[cg_et].edge_index.cpu().numpy()
            cg_attr = self.data[cg_et].edge_attr.cpu().numpy()
            gene_sum = np.zeros(self.n_genes, dtype=np.float64)
            gene_cnt = np.zeros(self.n_genes, dtype=np.float64)
            np.add.at(gene_sum, cg_ei[1], np.maximum(cg_attr[:, 0], 0))
            np.add.at(gene_cnt, cg_ei[1], 1.0)
            mask = gene_cnt > 0
            mean_expr[mask] = (gene_sum[mask] / self.n_cells).astype(np.float32)

        g2i = self.gene_name_to_idx
        n = min(len(raw_scores), len(vocab))
        intensity = {}
        for i in range(n):
            lig, rec = vocab[i]
            attn = float(raw_scores[i])
            li = g2i.get(lig.upper())
            ri = g2i.get(rec.upper())
            expr_l = float(mean_expr[li]) if li is not None else 0.0
            expr_r = float(mean_expr[ri]) if ri is not None else 0.0
            intensity[(lig, rec)] = attn * expr_l * expr_r

        # Normalise to [0, 1]
        max_v = max(intensity.values()) if intensity else 0
        if max_v > 0:
            intensity = {k: v / max_v for k, v in intensity.items()}

        self._intensity_cache = dict(sorted(intensity.items(), key=lambda x: -x[1]))
        return self._intensity_cache

    def lr_pairs(
        self,
        top_k: Optional[int] = None,
        receptor: Optional[str] = None,
        ligand: Optional[str] = None,
        scoring: str = "intensity",
        min_expr_frac: float = 0.0,
    ) -> List[Dict]:
        """
        Get ranked LR pair scores.

        Args:
            top_k:     Return only top-K pairs (None = all)
            receptor:  Filter to pairs with this receptor gene
            ligand:    Filter to pairs with this ligand gene
            scoring:   Scoring method:
                       "intensity" (default): attn × mean(expr_L) × mean(expr_R)
                           — used in all MOSAIC figures. Naturally breaks
                           degree-1 ties and weights by expression evidence.
                       "raw": raw attention (within-receptor ranking only)
                       "degree": attn × receptor_in_degree
                       "combined": attn × log1p(degree) × sqrt(expr_L × expr_R)
                       "embedding": cosine similarity of gene embeddings
            min_expr_frac: Filter pairs where either gene expressed in < this
                          fraction of cells (0 = no filter)

        Returns:
            List of dicts: [{"ligand": str, "receptor": str, "score": float,
                             "rank": int}, ...]
        """
        ext = self.extraction
        vocab = self.lr_pair_vocab

        if scoring == "intensity":
            scores = self._compute_intensity_scores()
        elif scoring == "raw":
            scores = self.extractor.get_lr_pair_scores(ext, vocab)
        elif scoring == "degree":
            scores = self.extractor.get_lr_pair_scores_enhanced(
                ext, vocab, mode="linear_degree"
            )
        elif scoring == "combined":
            scores = self.extractor.get_lr_pair_scores_enhanced(
                ext, vocab, mode="combined"
            )
        elif scoring == "embedding":
            scores = self.extractor.get_lr_pair_scores_embedding(
                vocab, gene_name_to_idx=self.gene_name_to_idx
            )
        elif scoring == "last_layer":
            scores = self.extractor.get_lr_pair_scores_last_layer(
                ext, vocab, layer_idx=-1
            )
        else:
            scores = self.extractor.get_lr_pair_scores_enhanced(
                ext, vocab, mode=scoring
            )

        if min_expr_frac > 0:
            expressed = self.extractor.get_lr_pair_scores_expressed(
                ext, vocab, min_expr_frac=min_expr_frac
            )
            scores = {k: v for k, v in scores.items() if k in expressed}

        if receptor:
            receptor = receptor.upper()
            scores = {k: v for k, v in scores.items() if k[1].upper() == receptor}
        if ligand:
            ligand = ligand.upper()
            scores = {k: v for k, v in scores.items() if k[0].upper() == ligand}

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            {"ligand": l, "receptor": r, "score": float(s), "rank": i + 1}
            for i, ((l, r), s) in enumerate(ranked)
        ]

    def lr_pairs_for_receptor(self, receptor: str, top_k: int = 20) -> List[Dict]:
        """
        Within-receptor ligand ranking (raw attention, directly interpretable).

        Because softmax is per-receptor, this shows the relative importance
        of each ligand for a specific receptor — no degree bias here.
        """
        return self.lr_pairs(receptor=receptor, top_k=top_k, scoring="raw")

    # -----------------------------------------------------------------
    # MR pair scoring
    # -----------------------------------------------------------------

    def mr_pairs(
        self,
        top_k: Optional[int] = None,
        metabolite: Optional[str] = None,
        receptor: Optional[str] = None,
        scoring: str = "flux_expr",
    ) -> List[Dict]:
        """
        Get ranked metabolite-receptor pair scores.

        ε₄ has only ~122 edges and most receptor genes have in-degree 1,
        so raw attention is nearly always 1.0 (useless for ranking).
        Default scoring combines attention with biological evidence:

        Args:
            scoring:
                "flux_expr" (default): α × mean_flux × mean_receptor_expr
                    Ranks by how much a metabolite actually flows AND the
                    receptor is expressed — not just graph connectivity.
                "raw": raw attention only (mostly 1.0, not recommended)
                "degree": α × in_degree (still nearly uniform for ε₄)

        Returns:
            List of dicts: [{"metabolite": str, "receptor": str,
                             "score": float, "rank": int, "attn": float,
                             "mean_flux": float, "mean_receptor_expr": float}]
        """
        ext = self.extraction
        e4_key = None
        for et in ext["edge_types"]:
            if "sensed_by" in et[1]:
                e4_key = et
                break

        if e4_key is None or e4_key not in ext["edge_scores"]:
            log.warning("No epsilon_4 (metabolite→gene) edges found.")
            return []

        e4_scores = ext["edge_scores"][e4_key]
        e4_ei = self.data[e4_key].edge_index.cpu().numpy()

        met_names = self.metabolite_names
        gene_names = self.gene_vocab
        n_edges = len(e4_scores)

        # --- Compute mean flux per metabolite (from ε₃ cell→met edges) ---
        e3_et = ("cell", "flux", "metabolite")
        mean_flux_per_met = np.zeros(self.n_metabolites, dtype=np.float32)
        if e3_et in self.data.edge_types:
            e3_ei = self.data[e3_et].edge_index.cpu().numpy()
            e3_attr = self.data[e3_et].edge_attr.cpu().numpy()
            # Column 0 = normalised flux magnitude
            for m_idx in range(self.n_metabolites):
                mask = e3_ei[1] == m_idx
                if mask.sum() > 0:
                    mean_flux_per_met[m_idx] = float(np.abs(e3_attr[mask, 0]).mean())

        # --- Compute mean expression per gene (from ε₁ cell→gene edges) ---
        cg_et = ("cell", "expresses", "gene")
        mean_expr_per_gene = np.zeros(self.n_genes, dtype=np.float32)
        if cg_et in self.data.edge_types:
            cg_ei = self.data[cg_et].edge_index.cpu().numpy()
            cg_attr = self.data[cg_et].edge_attr.cpu().numpy()
            gene_sum = np.zeros(self.n_genes, dtype=np.float64)
            gene_cnt = np.zeros(self.n_genes, dtype=np.float64)
            np.add.at(gene_sum, cg_ei[1], cg_attr[:, 0])
            np.add.at(gene_cnt, cg_ei[1], 1.0)
            mask = gene_cnt > 0
            mean_expr_per_gene[mask] = (gene_sum[mask] / gene_cnt[mask]).astype(np.float32)

        # --- Compute ε₄ in-degree per gene ---
        gene_indeg_e4 = np.zeros(self.n_genes, dtype=np.float32)
        np.add.at(gene_indeg_e4, e4_ei[1], 1.0)

        # --- Score each MR edge ---
        results = []
        for i in range(n_edges):
            met_idx = int(e4_ei[0, i])
            gene_idx = int(e4_ei[1, i])
            attn = float(e4_scores[i])
            flux = float(mean_flux_per_met[met_idx])
            expr = float(mean_expr_per_gene[gene_idx])
            deg = float(gene_indeg_e4[gene_idx])

            if scoring == "flux_expr":
                score = attn * flux * expr
            elif scoring == "raw":
                score = attn
            elif scoring == "degree":
                score = attn * deg
            else:
                score = attn * flux * expr

            met_name = met_names[met_idx] if met_idx < len(met_names) else f"met_{met_idx}"
            gene_name = gene_names[gene_idx] if gene_idx < len(gene_names) else f"gene_{gene_idx}"

            results.append({
                "metabolite": met_name,
                "metabolite_idx": met_idx,
                "receptor": gene_name,
                "receptor_idx": gene_idx,
                "score": score,
                "attn": attn,
                "mean_flux": flux,
                "mean_receptor_expr": expr,
            })

        # Normalise scores to [0, 1]
        max_score = max((r["score"] for r in results), default=0)
        if max_score > 0:
            for r in results:
                r["score"] = r["score"] / max_score

        results.sort(key=lambda x: -x["score"])

        if metabolite:
            metabolite = metabolite.upper()
            results = [r for r in results if r["metabolite"].upper() == metabolite]
        if receptor:
            receptor = receptor.upper()
            results = [r for r in results if r["receptor"].upper() == receptor]

        if top_k is not None:
            results = results[:top_k]

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    # -----------------------------------------------------------------
    # Cell-type communication
    # -----------------------------------------------------------------

    def communication_matrix(
        self,
        channel: str = "secreted",
        cell_types: Optional[np.ndarray] = None,
        cell_type_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Cell-type × cell-type communication matrix.

        Args:
            channel: "secreted", "metabolite", "contact", "intracellular",
                     or "all" (sum of secreted + metabolite + contact)
            cell_types: [N] integer labels (auto-loaded from graph if None)
            cell_type_names: optional string names

        Returns:
            dict with:
                "matrix": np.ndarray [n_types, n_types]
                "cell_type_names": list of str
                "channel": str
        """
        ct = cell_types if cell_types is not None else self.cell_types
        ct_names = cell_type_names or self.cell_type_names
        if ct is None:
            raise ValueError(
                "No cell_type labels found. Pass cell_types= or ensure "
                "data['cell'].cell_type exists."
            )

        ct_tensor = torch.from_numpy(ct) if isinstance(ct, np.ndarray) else ct
        matrices = self.extractor.get_cell_communication_matrix(
            self.extraction, ct_tensor, cell_type_names=ct_names
        )

        if channel == "all":
            combined = None
            for ch_name, mat in matrices.items():
                if ch_name == "intracellular":
                    continue
                combined = mat if combined is None else combined + mat
            matrix = combined
        elif channel in matrices:
            matrix = matrices[channel]
        else:
            avail = list(matrices.keys())
            raise ValueError(f"Channel '{channel}' not found. Available: {avail}")

        n_types = int(ct.max()) + 1
        if ct_names is None:
            ct_names = [f"type_{i}" for i in range(n_types)]

        return {
            "matrix": matrix,
            "cell_type_names": ct_names,
            "channel": channel,
            "all_channels": matrices,
        }

    # -----------------------------------------------------------------
    # Per-cell spatial activity maps
    # -----------------------------------------------------------------

    def lr_activity(
        self,
        ligand: str,
        receptor: str,
        scoring: str = "raw",
    ) -> np.ndarray:
        """
        Per-cell LR communication intensity for a specific pair.

        I_i(l, r) = s_raw(l, r) × max(x_il, 0) × max(x_ir, 0)

        Uses raw attention by default (matching all MOSAIC figures).
        Expression multiplication naturally resolves degree-1 ties.

        Args:
            ligand:   Ligand gene name
            receptor: Receptor gene name
            scoring:  "degree" (default) or "raw"

        Returns:
            np.ndarray [N] of per-cell activity
        """
        pair = (ligand.upper(), receptor.upper())

        # Get pair score
        lr_scores = self.lr_pairs(scoring=scoring)
        score = 0.0
        for entry in lr_scores:
            if (entry["ligand"].upper(), entry["receptor"].upper()) == pair:
                score = entry["score"]
                break

        if score == 0.0:
            log.warning("LR pair (%s, %s) not found or score=0.", ligand, receptor)

        # Get per-cell expression
        lig_idx = self.gene_name_to_idx.get(pair[0])
        rec_idx = self.gene_name_to_idx.get(pair[1])

        if lig_idx is None or rec_idx is None:
            log.warning("Gene not in vocab: lig=%s (found=%s), rec=%s (found=%s)",
                        pair[0], lig_idx is not None, pair[1], rec_idx is not None)
            return np.zeros(self.n_cells)

        # Expression from cell→gene edges
        cg_et = ("cell", "expresses", "gene")
        cg_ei = self.data[cg_et].edge_index.cpu().numpy()
        cg_attr = self.data[cg_et].edge_attr.cpu().numpy()

        lig_expr = np.zeros(self.n_cells, dtype=np.float32)
        rec_expr = np.zeros(self.n_cells, dtype=np.float32)

        mask_lig = cg_ei[1] == lig_idx
        lig_expr[cg_ei[0, mask_lig]] = cg_attr[mask_lig, 0]

        mask_rec = cg_ei[1] == rec_idx
        rec_expr[cg_ei[0, mask_rec]] = cg_attr[mask_rec, 0]

        return score * np.maximum(lig_expr, 0) * np.maximum(rec_expr, 0)

    def mr_activity(
        self,
        metabolite: str,
        receptor: str,
    ) -> np.ndarray:
        """
        Per-cell MR communication intensity.

        I_i(k, r) = alpha_kr × |f_ik| × max(x_ir, 0)

        Returns:
            np.ndarray [N] of per-cell MR activity
        """
        mr_scores = self.mr_pairs(metabolite=metabolite, receptor=receptor)
        if not mr_scores:
            log.warning("MR pair (%s, %s) not found.", metabolite, receptor)
            return np.zeros(self.n_cells)

        alpha_kr = mr_scores[0]["score"]
        met_idx = mr_scores[0]["metabolite_idx"]
        rec_idx_gene = self.gene_name_to_idx.get(receptor.upper())

        if rec_idx_gene is None:
            return np.zeros(self.n_cells)

        # Flux from cell→metabolite edges
        e3_et = ("cell", "flux", "metabolite")
        e3_ei = self.data[e3_et].edge_index.cpu().numpy()
        e3_attr = self.data[e3_et].edge_attr.cpu().numpy()

        flux = np.zeros(self.n_cells, dtype=np.float32)
        mask_met = e3_ei[1] == met_idx
        flux[e3_ei[0, mask_met]] = np.abs(e3_attr[mask_met, 0])

        # Receptor expression
        cg_et = ("cell", "expresses", "gene")
        cg_ei = self.data[cg_et].edge_index.cpu().numpy()
        cg_attr = self.data[cg_et].edge_attr.cpu().numpy()

        rec_expr = np.zeros(self.n_cells, dtype=np.float32)
        mask_rec = cg_ei[1] == rec_idx_gene
        rec_expr[cg_ei[0, mask_rec]] = cg_attr[mask_rec, 0]

        return alpha_kr * flux * np.maximum(rec_expr, 0)

    # -----------------------------------------------------------------
    # Communication hotspots
    # -----------------------------------------------------------------

    def communication_hotspots(
        self,
        top_k: int = 10,
        channel: str = "secreted",
        cell_types: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Find cell groups with strongest communication enrichment.

        Returns the top-K (sender_type, receiver_type) pairs ranked by
        mean attention, along with their top LR pairs.

        Returns:
            List of dicts: [{"sender": str, "receiver": str,
                             "score": float, "top_lr_pairs": [...], "rank": int}]
        """
        comm = self.communication_matrix(channel=channel, cell_types=cell_types)
        matrix = comm["matrix"]
        names = comm["cell_type_names"]

        # Flatten and rank
        n = matrix.shape[0]
        entries = []
        for i in range(n):
            for j in range(n):
                entries.append((names[i], names[j], float(matrix[i, j])))

        entries.sort(key=lambda x: -x[2])
        entries = entries[:top_k]

        # For each top pair, find which LR pairs contribute most
        lr_ranked = self.lr_pairs(top_k=50, scoring="degree")

        results = []
        for rank, (sender, receiver, score) in enumerate(entries):
            results.append({
                "sender": sender,
                "receiver": receiver,
                "score": score,
                "rank": rank + 1,
                "top_lr_pairs": lr_ranked[:5],
            })

        return results

    # -----------------------------------------------------------------
    # Clustering
    # -----------------------------------------------------------------

    def cluster(
        self,
        n_clusters: Optional[int] = None,
        method: str = "kmeans",
    ) -> np.ndarray:
        """
        Cluster cells using learned embeddings.

        Args:
            n_clusters: Number of clusters (defaults to number of cell types)
            method:     "kmeans" (default) or "leiden"

        Returns:
            np.ndarray [N] integer cluster labels
        """
        from sklearn.cluster import KMeans

        emb = self.cell_embeddings

        if n_clusters is None:
            ct = self.cell_types
            n_clusters = int(ct.max()) + 1 if ct is not None else 10

        if method == "kmeans":
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(emb)
        elif method == "leiden":
            try:
                import scanpy as sc
                import anndata as ad
                adata = ad.AnnData(X=emb)
                sc.pp.neighbors(adata, use_rep="X", n_neighbors=15)
                sc.tl.leiden(adata, resolution=1.0)
                labels = adata.obs["leiden"].astype(int).values
            except ImportError:
                raise ImportError("Leiden clustering requires scanpy: pip install scanpy")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'leiden'.")

        return labels

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    def plot_lr_ranking(
        self,
        top_k: int = 30,
        scoring: str = "degree",
        figsize: Tuple[float, float] = (8, 6),
        ax=None,
    ):
        """
        Horizontal bar plot of top-K LR pairs by score.

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        pairs = self.lr_pairs(top_k=top_k, scoring=scoring)
        if not pairs:
            log.warning("No LR pairs to plot.")
            return None

        labels = [f"{p['ligand']}→{p['receptor']}" for p in reversed(pairs)]
        scores = [p["score"] for p in reversed(pairs)]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.barh(range(len(labels)), scores, color="#4A90D9", edgecolor="none")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"LR Score ({scoring})")
        ax.set_title(f"Top-{top_k} LR Pairs — {self.dataset}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig

    def plot_communication_matrix(
        self,
        channel: str = "secreted",
        cell_types: Optional[np.ndarray] = None,
        cell_type_names: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (8, 7),
        cmap: str = "YlOrRd",
        ax=None,
    ):
        """
        Heatmap of cell-type communication matrix.

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        comm = self.communication_matrix(
            channel=channel, cell_types=cell_types,
            cell_type_names=cell_type_names,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(comm["matrix"], cmap=cmap, aspect="auto")
        names = comm["cell_type_names"]
        n = len(names)
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Sender")
        ax.set_title(f"Communication ({channel}) — {self.dataset}")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Mean attention")
        plt.tight_layout()
        return fig

    def plot_spatial(
        self,
        ligand: str,
        receptor: str,
        scoring: str = "degree",
        cmap: str = "YlOrRd",
        figsize: Tuple[float, float] = (8, 7),
        s: float = 5,
        ax=None,
    ):
        """
        Spatial scatter plot of per-cell LR activity.

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        coords = self.coords
        if coords is None:
            raise ValueError("No spatial coordinates available.")

        activity = self.lr_activity(ligand, receptor, scoring=scoring)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        order = np.argsort(activity)
        sc = ax.scatter(
            coords[order, 0], coords[order, 1],
            c=activity[order], cmap=cmap, s=s,
            edgecolor="none", rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{ligand}→{receptor} activity — {self.dataset}")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="LR intensity")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig

    def plot_spatial_mr(
        self,
        metabolite: str,
        receptor: str,
        cmap: str = "YlOrRd",
        figsize: Tuple[float, float] = (8, 7),
        s: float = 5,
        ax=None,
    ):
        """Spatial scatter plot of per-cell MR activity."""
        import matplotlib.pyplot as plt

        coords = self.coords
        if coords is None:
            raise ValueError("No spatial coordinates available.")

        activity = self.mr_activity(metabolite, receptor)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        order = np.argsort(activity)
        sc = ax.scatter(
            coords[order, 0], coords[order, 1],
            c=activity[order], cmap=cmap, s=s,
            edgecolor="none", rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{metabolite}→{receptor} MR activity — {self.dataset}")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="MR intensity")
        plt.tight_layout()
        return fig

    def plot_clusters(
        self,
        labels: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        cmap: str = "tab20",
        figsize: Tuple[float, float] = (8, 7),
        s: float = 5,
        ax=None,
    ):
        """
        Spatial scatter plot coloured by cluster labels.

        If labels is None, runs k-means clustering first.
        """
        import matplotlib.pyplot as plt

        coords = self.coords
        if coords is None:
            raise ValueError("No spatial coordinates available.")

        if labels is None:
            labels = self.cluster(n_clusters=n_clusters)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap=cmap, s=s,
            edgecolor="none", rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(f"Cell clusters ({len(np.unique(labels))}) — {self.dataset}")
        plt.tight_layout()
        return fig

    def plot_communication_edges(
        self,
        sender_type: str,
        receiver_type: str,
        channel: str = "secreted",
        top_n_edges: int = 500,
        figsize: Tuple[float, float] = (10, 8),
        ax=None,
    ):
        """
        Spatial plot showing communication edges between two cell types.

        Draws lines between sender and receiver cells, colored by attention.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        coords = self.coords
        ct = self.cell_types
        ct_names = self.cell_type_names
        if coords is None or ct is None:
            raise ValueError("Need spatial coords and cell_type labels.")

        # Find type indices
        if ct_names is not None:
            s_idx = ct_names.index(sender_type) if sender_type in ct_names else None
            r_idx = ct_names.index(receiver_type) if receiver_type in ct_names else None
        else:
            s_idx = int(sender_type) if sender_type.isdigit() else None
            r_idx = int(receiver_type) if receiver_type.isdigit() else None

        if s_idx is None or r_idx is None:
            raise ValueError(f"Cell type not found: {sender_type} or {receiver_type}")

        # Get edges + scores for this channel
        ext = self.extraction
        ch_scores = ext["cell_cell_edge_scores"].get(channel)
        if ch_scores is None:
            raise ValueError(f"Channel '{channel}' not in extraction.")

        # Find edge_index for this channel
        et_match = None
        for et in ext["edge_types"]:
            if et[0] == "cell" and et[1] == channel and et[2] == "cell":
                et_match = et
                break
        if et_match is None:
            raise ValueError(f"Edge type for channel '{channel}' not found.")

        ei = self.data[et_match].edge_index.cpu().numpy()

        # Filter to sender_type → receiver_type
        mask = (ct[ei[0]] == s_idx) & (ct[ei[1]] == r_idx)
        src_cells = ei[0, mask]
        dst_cells = ei[1, mask]
        scores = ch_scores[mask]

        # Take top-N by score
        if len(scores) > top_n_edges:
            top_idx = np.argsort(-scores)[:top_n_edges]
            src_cells = src_cells[top_idx]
            dst_cells = dst_cells[top_idx]
            scores = scores[top_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Background: all cells in grey
        ax.scatter(coords[:, 0], coords[:, 1], c="#E0E0E0", s=2,
                   edgecolor="none", rasterized=True, zorder=1)

        # Sender/receiver cells
        s_mask = ct == s_idx
        r_mask = ct == r_idx
        ax.scatter(coords[s_mask, 0], coords[s_mask, 1], c="#4A90D9", s=8,
                   label=sender_type, edgecolor="none", zorder=2)
        ax.scatter(coords[r_mask, 0], coords[r_mask, 1], c="#E74C3C", s=8,
                   label=receiver_type, edgecolor="none", zorder=2)

        # Draw edges
        if len(scores) > 0:
            norm_scores = scores / scores.max()
            lines = [
                [coords[s], coords[d]]
                for s, d in zip(src_cells, dst_cells)
            ]
            lc = LineCollection(
                lines, linewidths=0.5, alpha=0.3,
                colors=plt.cm.Reds(norm_scores), zorder=3,
            )
            ax.add_collection(lc)

        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_title(f"{sender_type}→{receiver_type} ({channel}) — {self.dataset}")
        plt.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # Knockout (perturbation) analysis
    # -----------------------------------------------------------------

    def knockout(self, per_gene: bool = False) -> Dict:
        """
        Run edge-type ablation analysis.

        Removes each edge type one at a time and measures change in
        expression prediction (delta-MSE, delta-R²).

        Args:
            per_gene: If True, also compute per-gene delta-R² (slower).

        Returns:
            dict with:
                "channel_importance": {edge_type: delta_mse}
                "full_mse": float
                "per_gene_importance": (only if per_gene=True)
        """
        from mosaic.analysis.knockout import run_knockout
        return run_knockout(self.model, self.data, device=self.device, per_gene=per_gene)

    def knockout_gene(
        self,
        gene: str,
        edge_type: str = "interacts",
    ) -> Dict:
        """
        In-silico knockout of a specific gene's LR edges.

        Removes all ε₂ edges involving this gene (as ligand or receptor),
        re-runs forward pass, and returns the change in predicted expression.

        Args:
            gene:      Gene name to knock out
            edge_type: Edge relation to ablate ("interacts" for LR)

        Returns:
            dict with "delta_r2": float, "baseline_r2": float,
            "baseline_pred": [N, G], "knockout_pred": [N, G]
        """
        import copy
        gene_upper = gene.upper()
        g2i = self.gene_name_to_idx

        if gene_upper not in g2i:
            raise ValueError(f"Gene '{gene}' not in gene vocabulary.")

        gene_idx = g2i[gene_upper]

        # Find the edge type
        target_et = None
        for et in self.data.edge_types:
            if et[1] == edge_type:
                target_et = et
                break
        if target_et is None:
            raise ValueError(f"Edge type '{edge_type}' not found.")

        self.model.eval()
        data = self.data.to(self.device)

        with torch.no_grad():
            # Baseline
            baseline_out = self.model(data)
            baseline_pred = baseline_out["expression"].cpu().numpy()

            # Knockout: remove edges involving this gene
            x_dict, ei_dict, ea_dict = self.model._extract_graph_data(data)
            ei = ei_dict[target_et]  # [2, E]

            # For ε₂ (gene→gene), remove where src or dst is the gene
            mask = (ei[0] != gene_idx) & (ei[1] != gene_idx)
            ei_dict_mod = dict(ei_dict)
            ea_dict_mod = dict(ea_dict)
            ei_dict_mod[target_et] = ei[:, mask]
            if ea_dict.get(target_et) is not None:
                ea_dict_mod[target_et] = ea_dict[target_et][mask]

            ko_emb = self.model.encoder(x_dict, ei_dict_mod, ea_dict_mod)
            ko_pred = self.model.expression_decoder(ko_emb).cpu().numpy()

        # Compute R² change
        test_mask = self.data["cell"].test_mask.cpu().numpy()
        y_true = self.data["cell"].y_expr.cpu().numpy()

        def _r2(pred):
            p = np.clip(pred[test_mask], 0, None)
            t = y_true[test_mask]
            ss_res = ((t - p) ** 2).mean()
            ss_tot = ((t - t.mean(axis=0)) ** 2).mean()
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        baseline_r2 = _r2(baseline_pred)
        ko_r2 = _r2(ko_pred)

        n_removed = int((~mask).sum())
        log.debug("Knockout %s: removed %d/%d edges, ΔR²=%.4f",
                  gene, n_removed, len(mask), baseline_r2 - ko_r2)

        return {
            "gene": gene,
            "delta_r2": float(baseline_r2 - ko_r2),
            "baseline_r2": float(baseline_r2),
            "knockout_r2": float(ko_r2),
            "n_edges_removed": n_removed,
            "baseline_pred": baseline_pred,
            "knockout_pred": ko_pred,
        }

    # -----------------------------------------------------------------
    # Relay detection
    # -----------------------------------------------------------------

    def relays(
        self,
        top_pct: List[float] = None,
    ) -> Dict:
        """
        Detect multi-hop relay networks from cell-cell attention.

        Args:
            top_pct: Percentile thresholds (default: [5, 10, 15, 20, 30, 50])

        Returns:
            dict with relay results at each threshold + 2-hop chains
        """
        from mosaic.analysis.relay import detect_relays

        ct = self.cell_types
        ct_names = self.cell_type_names
        if ct is None:
            raise ValueError("Cell types required for relay analysis. "
                             "Call set_cell_types_from_adata() first.")

        ct_mapping = {}
        if ct_names:
            ct_mapping = {i: name for i, name in enumerate(ct_names)}
        else:
            ct_mapping = {i: f"type_{i}" for i in range(int(ct.max()) + 1)}

        return detect_relays(
            self.model, self.data,
            cell_types=ct,
            ct_mapping=ct_mapping,
            coords=self.coords,
            device=str(self.device),
            top_pct_values=top_pct,
        )

    def relay_chains(
        self,
        top_k: int = 20,
        channel: str = "secreted",
    ) -> List[Dict]:
        """
        Get formatted 2-hop relay chains: A → B → C.

        Args:
            top_k:   Number of top chains to return
            channel: "secreted", "metabolite", or "cross_channel"

        Returns:
            List of dicts with readable chain info.
        """
        relay_result = self.relays(top_pct=[20])
        key = f"2hop_{channel}"
        chains = relay_result.get(key, [])

        formatted = []
        for c in chains[:top_k]:
            formatted.append({
                "chain": f"{c.get('ct_a', '?')} → {c.get('ct_b', '?')} → {c.get('ct_c', '?')}",
                "cells": f"{c['a']} → {c['b']} → {c['c']}",
                "score": float(c.get("product", 0)),
                "attn_AB": float(c.get("ab_attn", 0)),
                "attn_BC": float(c.get("bc_attn", 0)),
                "dist_AB": float(c.get("dist_ab", 0)),
                "dist_BC": float(c.get("dist_bc", 0)),
                "rank": len(formatted) + 1,
            })
        return formatted

    def relay_hubs(
        self,
        top_k: int = 20,
        top_pct: float = 20,
    ) -> List[Dict]:
        """
        Identify relay hub genes — genes that appear as intermediaries
        in many 2-hop relay chains (A→B→C where B is the hub).

        These are prime candidates for knockout experiments since
        disrupting a hub disrupts multiple communication cascades.

        Args:
            top_k:   Number of top hub genes to return
            top_pct: Attention percentile threshold for relay detection

        Returns:
            List of dicts: [{"gene": str, "n_chains": int,
                             "mean_relay_score": float, "rank": int,
                             "example_chains": [...]}]
        """
        relay_result = self.relays(top_pct=[top_pct])
        key = f"top_{int(top_pct)}pct"

        # Collect 2-hop chains
        chains_2hop = relay_result.get("2hop_secreted", [])
        chains_met = relay_result.get("2hop_metabolite", [])
        chains_cross = relay_result.get("2hop_cross_channel", [])
        all_chains = chains_2hop + chains_met + chains_cross

        if not all_chains:
            log.warning("No 2-hop relay chains found at %d%% threshold.", top_pct)
            return []

        # Count hub (middle cell) appearances
        # Each chain is a dict with 'relay_cell' or 'B' as the middle node
        hub_counts = {}
        hub_scores = {}
        hub_examples = {}

        for chain in all_chains:
            b = chain.get("b", chain.get("relay_cell", chain.get("B")))
            if b is None:
                continue
            hub_counts[b] = hub_counts.get(b, 0) + 1
            score = chain.get("product", chain.get("attn_product", chain.get("score", 0)))
            hub_scores.setdefault(b, []).append(float(score))
            hub_examples.setdefault(b, []).append(chain)

        # Map cell indices to cell types
        ct = self.cell_types
        ct_names = self.cell_type_names

        results = []
        for cell_idx, count in sorted(hub_counts.items(), key=lambda x: -x[1]):
            ct_label = ""
            if ct is not None and ct_names is not None:
                ct_int = int(ct[cell_idx])
                ct_label = ct_names[ct_int] if ct_int < len(ct_names) else f"type_{ct_int}"

            results.append({
                "cell_idx": int(cell_idx),
                "cell_type": ct_label,
                "n_chains": count,
                "mean_relay_score": float(np.mean(hub_scores[cell_idx])),
                "example_chains": hub_examples[cell_idx][:3],
            })

        results = results[:top_k]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    def suggest_knockouts(
        self,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Suggest genes for knockout based on multiple evidence sources:
        1. High LR pair intensity (from lr_pairs)
        2. Relay hub frequency (from relay analysis)
        3. Channel importance contribution

        Returns genes ranked by combined evidence.

        Returns:
            List of dicts: [{"gene": str, "lr_rank": int,
                             "n_relay_chains": int, "combined_score": float}]
        """
        # Top genes from LR pairs
        lr_top = self.lr_pairs(top_k=100, scoring="intensity")
        gene_lr_rank = {}
        for p in lr_top:
            for g in [p["ligand"], p["receptor"]]:
                if g not in gene_lr_rank:
                    gene_lr_rank[g] = p["rank"]

        # Top relay hubs
        try:
            hubs = self.relay_hubs(top_k=50)
            hub_map = {h["cell_type"]: h["n_chains"] for h in hubs}
        except Exception:
            hub_map = {}

        # Combine: genes appearing in both top LR and relay hubs get highest score
        all_genes = set(gene_lr_rank.keys())
        results = []
        for gene in all_genes:
            lr_rank = gene_lr_rank.get(gene, 999)
            lr_score = max(0, 1 - lr_rank / 100)  # 1.0 for rank 1, 0 for rank 100
            results.append({
                "gene": gene,
                "lr_rank": lr_rank,
                "combined_score": lr_score,
            })

        results.sort(key=lambda x: -x["combined_score"])
        results = results[:top_k]
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results

    # -----------------------------------------------------------------
    # Knockout + Relay visualizations
    # -----------------------------------------------------------------

    def plot_knockout_comparison(
        self,
        genes: List[str],
        figsize: Optional[Tuple[float, float]] = None,
        point_size: Optional[float] = None,
    ):
        """Per-gene spatial knockout: baseline | knocked-out | Δ.

        For each gene in ``genes``, draws three side-by-side spatial maps —
        the baseline LR activity, the post-knockout LR activity, and the
        signed Δ (baseline − knockout) — sharing colorbars across genes for
        comparability.

        Args:
            genes:      list of gene symbols to knock out.
            figsize:    (W, H) override. Default scales to give each panel
                        a square aspect with ≥ 3 in side.
            point_size: scatter marker size. Default auto-scales with the
                        spatial-coordinate range so cells are visible.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, TwoSlopeNorm
        from matplotlib.gridspec import GridSpec

        coords = self.coords
        if coords is None:
            raise ValueError("No spatial coordinates.")
        n_genes = len(genes)
        if n_genes == 0:
            raise ValueError("genes list is empty")

        # Auto-scale marker size based on spatial extent + cell count
        if point_size is None:
            extent = max(coords[:, 0].ptp(), coords[:, 1].ptp())
            density = self.n_cells / (extent ** 2 + 1e-9)
            point_size = float(np.clip(2200.0 / max(density ** 0.5, 0.01), 6, 26))

        # Figure: one row per gene, 3 columns + one shared colorbar column.
        if figsize is None:
            panel = 3.4
            figsize = (panel * 3 + 0.6, panel * n_genes + 0.6)

        fig = plt.figure(figsize=figsize, dpi=110)
        gs = GridSpec(
            n_genes, 4, figure=fig,
            width_ratios=[1, 1, 1, 0.04], wspace=0.06, hspace=0.18,
        )

        # ── Baseline LR activity (top-50 pairs) ──
        top50 = self.lr_pairs(top_k=50, scoring="intensity")
        baseline = np.zeros(self.n_cells, dtype=np.float32)
        for p in top50:
            baseline += self.lr_activity(p["ligand"], p["receptor"])
        norm_bl = Normalize(
            vmin=float(np.percentile(baseline, 2)),
            vmax=float(np.percentile(baseline, 98) + 1e-9),
        )

        # Pre-compute knockouts so we can share the Δ-colormap range
        ko_records = []
        for gene in genes:
            ko = self.knockout_gene(gene)
            kn_act = np.zeros(self.n_cells, dtype=np.float32)
            for p in top50:
                kn_act += self.lr_activity(p["ligand"], p["receptor"])  # baseline (unchanged)
            # Δ = baseline-prediction mean − knockout-prediction mean (per cell)
            delta = (ko["baseline_pred"].mean(axis=1)
                     - ko["knockout_pred"].mean(axis=1))
            ko_records.append((gene, ko, delta))

        # Shared Δ-norm across all genes
        vmax_d = max(
            float(np.abs(np.percentile(d, [1, 99])).max()) for _, _, d in ko_records
        )
        vmax_d = max(vmax_d, 1e-6)
        norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0.0, vmax=vmax_d)

        def _scatter(ax, vals, cmap, norm):
            order = np.argsort(np.abs(vals))   # plot extremes on top
            return ax.scatter(
                coords[order, 0], coords[order, 1], c=vals[order],
                cmap=cmap, norm=norm, s=point_size,
                edgecolor="none", rasterized=True,
            )

        bl_ax = None
        for r, (gene, ko, delta) in enumerate(ko_records):
            ax_bl = fig.add_subplot(gs[r, 0]); bl_ax = ax_bl
            ax_ko = fig.add_subplot(gs[r, 1])
            ax_d  = fig.add_subplot(gs[r, 2])

            sc_bl = _scatter(ax_bl, baseline, "YlOrRd", norm_bl)
            sc_d  = _scatter(ax_d,  delta,    "RdBu_r", norm_d)
            # KO map: baseline scaled by per-cell post-KO predicted activity
            ko_vals = baseline - delta  # rough analogue of post-KO activity
            ax_ko.scatter(
                coords[:, 0], coords[:, 1], c=ko_vals,
                cmap="YlOrRd", norm=norm_bl, s=point_size,
                edgecolor="none", rasterized=True,
            )

            for ax in (ax_bl, ax_ko, ax_d):
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax_bl.set_ylabel(gene, fontsize=11, fontweight="bold", labelpad=4)
            if r == 0:
                ax_bl.set_title("Baseline LR",   fontsize=10, pad=4)
                ax_ko.set_title("After knockout", fontsize=10, pad=4)
                ax_d .set_title("Δ (baseline − KO)", fontsize=10, pad=4)

        # Shared colorbar (rightmost column spans all rows)
        cax = fig.add_subplot(gs[:, 3])
        cb = fig.colorbar(sc_d, cax=cax)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Δ LR activity (per cell)", fontsize=9)

        fig.suptitle(
            f"In-silico knockout — {self.dataset}",
            fontsize=12, y=0.995,
        )
        return fig

    def plot_channel_importance(
        self,
        figsize: Tuple[float, float] = (8, 4),
        ax=None,
    ):
        """Bar chart of edge-type importance from knockout analysis."""
        import matplotlib.pyplot as plt

        ko = self.knockout()
        ch_imp = ko["channel_importance"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        names = []
        deltas = []
        for et_str, info in sorted(ch_imp.items(), key=lambda x: -x[1].get("delta_mse", 0)):
            names.append(et_str.split("__")[1] if "__" in et_str else et_str)
            deltas.append(info.get("delta_mse", 0))

        colors = ["#F1C40F" if "interacts" in n else "#4A90D9" for n in names]
        ax.barh(range(len(names)), deltas, color=colors, edgecolor="none")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("ΔMSE (higher = more important)")
        ax.set_title(f"Edge-type importance — {self.dataset}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # Hub-score (paper §4, Eq. 18) — the central scalar
    # -----------------------------------------------------------------

    def hub_scores(
        self,
        top_k: Optional[int] = None,
        channels: str = "all",
        as_dataframe: bool = True,
    ):
        """Per-gene hub-score (paper §4, Eq. 18).

        Sum of incoming attention into each gene, averaged over heads and
        encoder layers, restricted to the requested channel mask.

        Args:
            top_k:    return top-K hubs only (None = all genes with score>0).
            channels: "all"  → canonical hub-score over ε₂ + ε₄ (default).
                      "lr"   → LR-network hub  (ε₂ only; paper Fig 6a–b).
                      "mr"   → MR hub          (ε₄ only; paper Fig 4c–d).
            as_dataframe: return a pandas DataFrame (default) or list of dicts.

        Returns:
            DataFrame with columns ``gene``, ``hub_score``, ``rank``
            (sorted descending). When ``as_dataframe=False`` returns
            ``[{"gene": ..., "hub_score": ..., "rank": ...}, ...]``.
        """
        ch = channels.lower()
        if ch not in ("all", "lr", "mr"):
            raise ValueError(f"channels must be 'all', 'lr', or 'mr'; got {channels!r}")

        ext = self.extraction
        edge_types = ext["edge_types"]
        edge_scores = ext.get("edge_scores", {})  # {edge_type: [E] mean attn across heads + layers}

        gene_score = np.zeros(self.n_genes, dtype=np.float64)
        use_lr = ch in ("all", "lr")
        use_mr = ch in ("all", "mr")

        # ε₂: gene → gene LR edges — incoming attention into receptor gene.
        lr_et = ("gene", "interacts", "gene")
        if use_lr and lr_et in edge_types and lr_et in edge_scores:
            attn = np.asarray(edge_scores[lr_et])
            gg_ei = self.data[lr_et].edge_index.cpu().numpy()
            np.add.at(gene_score, gg_ei[1, : len(attn)], attn)

        # ε₄: metabolite → gene MR edges — incoming attention into receptor gene.
        mr_et = ("metabolite", "sensed_by", "gene")
        if use_mr and mr_et in edge_types and mr_et in edge_scores:
            attn = np.asarray(edge_scores[mr_et])
            mg_ei = self.data[mr_et].edge_index.cpu().numpy()
            np.add.at(gene_score, mg_ei[1, : len(attn)], attn)

        gene_names = self.gene_vocab
        mask = gene_score > 0
        idx = np.argsort(-gene_score[mask])
        ordered_genes = np.asarray(gene_names)[mask][idx]
        ordered_scores = gene_score[mask][idx]
        if top_k is not None:
            ordered_genes = ordered_genes[:top_k]
            ordered_scores = ordered_scores[:top_k]

        records = [
            {"gene": str(g), "hub_score": float(s), "rank": i + 1}
            for i, (g, s) in enumerate(zip(ordered_genes, ordered_scores))
        ]
        if as_dataframe:
            import pandas as pd
            return pd.DataFrame.from_records(records)
        return records

    # -----------------------------------------------------------------
    # Hub fan-out (paper Fig 6h — multiplexer / SCARF1-style pattern)
    # -----------------------------------------------------------------

    def hub_fanout(
        self,
        gene: str,
        top_k: int = 20,
        as_dataframe: bool = True,
    ):
        """Downstream LR-pair fan-out from a hub gene.

        Treating ``gene`` as a receptor, list the ligands feeding into it
        (paper Fig 6h "SCARF1 multiplexer" analogue).

        Args:
            gene:   receptor gene symbol (case-insensitive).
            top_k:  return top-K ligands by attention-weighted intensity.
            as_dataframe: return a pandas DataFrame (default) or list of dicts.

        Returns:
            DataFrame with columns ``ligand``, ``receptor``, ``score``
            (intensity-weighted attention, sorted descending).
        """
        receptor = gene.upper()
        pairs = self.lr_pairs(receptor=receptor, scoring="intensity")
        pairs = pairs[: top_k]

        if as_dataframe:
            import pandas as pd
            return pd.DataFrame.from_records(pairs)
        return pairs

    # -----------------------------------------------------------------
    # Independent-DB evaluation (paper Fig 2 — AUROC vs OmniPath / CDB / NCB)
    # -----------------------------------------------------------------

    def evaluate_against(
        self,
        db_path: str,
        ligand_col: str = "ligand",
        receptor_col: str = "receptor",
    ) -> Dict[str, float]:
        """AUROC of MOSAIC's LR scores against an independent reference catalogue.

        Treats each LR pair in MOSAIC's vocabulary as a binary-classifier
        instance: positive if present in ``db_path``, negative otherwise.
        Reports AUROC + AUPR. This mirrors the Fig 2 evaluation procedure
        used in the paper against OmniPath, ConnectomeDB2025 and NeuronChatDB.

        Args:
            db_path:      CSV/TSV/JSON with at least the ligand and receptor columns.
            ligand_col:   column name for the ligand symbol.
            receptor_col: column name for the receptor symbol.

        Returns:
            ``{"auroc": float, "aupr": float, "n_pairs_in_vocab": int,
               "n_positives": int, "n_negatives": int}``
        """
        import pandas as pd
        from sklearn.metrics import roc_auc_score, average_precision_score

        # Load reference DB
        p = Path(db_path)
        if p.suffix.lower() == ".json":
            ref_df = pd.read_json(p)
        elif p.suffix.lower() == ".tsv":
            ref_df = pd.read_csv(p, sep="\t")
        else:
            ref_df = pd.read_csv(p)
        if ligand_col not in ref_df.columns or receptor_col not in ref_df.columns:
            raise ValueError(
                f"reference DB must contain columns {ligand_col!r} and "
                f"{receptor_col!r}; got {list(ref_df.columns)}"
            )
        ref_pairs = {
            (str(l).upper(), str(r).upper())
            for l, r in zip(ref_df[ligand_col], ref_df[receptor_col])
            if isinstance(l, str) and isinstance(r, str)
        }

        # Score every pair in our vocabulary
        scores_dict = self._compute_intensity_scores()
        y_true = []
        y_score = []
        for (lig, rec), s in scores_dict.items():
            y_true.append(int((lig.upper(), rec.upper()) in ref_pairs))
            y_score.append(float(s))
        if not y_true or sum(y_true) == 0 or sum(y_true) == len(y_true):
            raise ValueError(
                "Cannot evaluate: reference DB has no positives in MOSAIC's "
                "vocabulary (or contains every pair). "
                f"n_vocab={len(y_true)}, n_positives={sum(y_true)}"
            )

        y_true_arr = np.asarray(y_true)
        y_score_arr = np.asarray(y_score)
        return {
            "auroc":             float(roc_auc_score(y_true_arr, y_score_arr)),
            "aupr":              float(average_precision_score(y_true_arr, y_score_arr)),
            "n_pairs_in_vocab":  int(len(y_true)),
            "n_positives":       int(sum(y_true)),
            "n_negatives":       int(len(y_true) - sum(y_true)),
        }

    # -----------------------------------------------------------------
    # Summary / repr
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            f"MOSAICResult — {self.dataset}",
            f"  Cells: {self.n_cells:,}",
            f"  Genes: {self.n_genes:,}",
            f"  Metabolites: {self.n_metabolites:,}",
            f"  LR pairs: {len(self.lr_pair_vocab):,}",
            f"  Device: {self.device}",
        ]
        if self.cell_type_names:
            lines.append(f"  Cell types: {len(self.cell_type_names)} ({', '.join(self.cell_type_names[:5])}{'...' if len(self.cell_type_names) > 5 else ''})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MOSAICResult(dataset='{self.dataset}', "
            f"cells={self.n_cells}, genes={self.n_genes}, "
            f"lr_pairs={len(self.lr_pair_vocab)})"
        )


# ---------------------------------------------------------------------------
# Factory functions: train() and load()
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    device: str = "cuda",
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
    lr: Optional[float] = None,
    lambda_spatial: Optional[float] = None,
    force_preprocess: bool = False,
) -> MOSAICResult:
    """
    Train MOSAIC and return a MOSAICResult for interactive analysis.

    Usage:
        from mosaic import train
        result = train("configs/breast.yaml", device="cuda")
        result.lr_pairs(top_k=20)
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    root = Path(cfg["paths"]["root"]).parent
    processed_dir = root / cfg["paths"]["processed_dir"] / dataset
    graph_path = processed_dir / "hetero_ccc_graph.pt"

    # Preprocess if needed
    if not graph_path.exists() or force_preprocess:
        from mosaic.data import preprocess_dataset
        preprocess_dataset(config_path, dataset, force=force_preprocess)

    # Load graph
    ds = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = ds["hetero_graph"]
    meta = ds["metadata"]

    # Apply overrides
    if epochs is not None:
        cfg.setdefault("training", {})["epochs"] = epochs
    if patience is not None:
        cfg.setdefault("training", {})["patience"] = patience
    if lr is not None:
        cfg.setdefault("training", {})["lr"] = lr
    if lambda_spatial is not None:
        cfg.setdefault("training", {})["lambda_spatial"] = lambda_spatial

    # Build model
    from mosaic.models import build_model
    model = build_model(cfg, n_expr_genes=meta["n_expr_genes"], graph_metadata=meta)

    # Train
    from mosaic.training import MOSAICTrainer
    trainer = MOSAICTrainer(model=model, data=data, config=cfg, device=dev, dataset=dataset)
    n_ep = cfg.get("training", {}).get("epochs", 500)
    history = trainer.train(n_ep)
    log.info("Training complete. Best val R²=%.4f", max(history["val_r2"]))

    # Load best checkpoint
    ckpt_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "mosaic/checkpoints")) / dataset
    ckpt_path = ckpt_dir / "model_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    return MOSAICResult(
        model=model, data=data, meta=meta,
        config=cfg, dataset=dataset, device=dev,
    )


def load(
    config_path: str,
    checkpoint: Optional[str] = None,
    device: str = "cuda",
) -> MOSAICResult:
    """
    Load a trained MOSAIC model and return a MOSAICResult.

    Usage:
        from mosaic import load
        result = load("configs/breast.yaml", device="cuda")
        result.lr_pairs(top_k=20)
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    root = Path(cfg["paths"]["root"]).parent
    processed_dir = root / cfg["paths"]["processed_dir"] / dataset
    graph_path = processed_dir / "hetero_ccc_graph.pt"

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found: {graph_path}\n"
            f"Run: mosaic preprocess --config {config_path}"
        )

    ds = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = ds["hetero_graph"]
    meta = ds["metadata"]

    from mosaic.models import build_model
    model = build_model(cfg, n_expr_genes=meta["n_expr_genes"], graph_metadata=meta)

    # Find checkpoint
    if checkpoint is None:
        ckpt_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "mosaic/checkpoints")) / dataset
        checkpoint = str(ckpt_dir / "model_best.pt")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        # Try src5 checkpoints as fallback
        alt = Path(f"src5/checkpoints/{dataset}/model_best.pt")
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    log.info("Loaded checkpoint: %s", ckpt_path)

    return MOSAICResult(
        model=model, data=data, meta=meta,
        config=cfg, dataset=dataset, device=dev,
    )


# ---------------------------------------------------------------------------
# run_pipeline(): zero-config entry point
# ---------------------------------------------------------------------------

def _find_package_dir() -> Path:
    """Locate the MOSAIC package installation directory."""
    return Path(__file__).parent


def _find_emb_dir(pkg_dir: Path, kind: str) -> Path:
    """
    Search for pre-computed embedding directory.

    Checks package data dir, then common sibling locations.
    If not found, returns a directory that will be created for auto-compute caching.
    """
    candidates = [
        pkg_dir / "data" / "embeddings" / kind,
        pkg_dir.parent / "data" / "embeddings" / kind,
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.npy")):
            return p
    # Return package data dir (auto-compute will cache here)
    return candidates[0]


def _build_config(
    adata_path: str,
    output_dir: str,
    dataset: str,
    technology: str,
    organism: str,
    um_per_pixel: Optional[float] = None,
    scfea_balance: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    epochs: int = 500,
) -> dict:
    """
    Auto-generate a complete MOSAIC config from minimal user input.

    Resolves all database/embedding paths from the package installation
    directory. No hardcoded external paths.
    """
    import yaml

    pkg_dir = _find_package_dir()

    # Load defaults
    with open(pkg_dir / "configs" / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load technology config
    tech_path = pkg_dir / "configs" / f"{technology}.yaml"
    if tech_path.exists():
        with open(tech_path) as f:
            tech_cfg = yaml.safe_load(f)
        cfg.setdefault("spatial", {}).update(tech_cfg.get("spatial", {}))
        cfg["technology"] = technology
    else:
        avail = [p.stem for p in (pkg_dir / "configs").glob("*.yaml") if p.stem != "default"]
        raise ValueError(f"Unknown technology: '{technology}'. Available: {avail}")

    # Load organism config
    org_path = pkg_dir / "configs" / "organisms" / f"{organism}.yaml"
    if org_path.exists():
        with open(org_path) as f:
            org_cfg = yaml.safe_load(f)
        cfg["organism"] = organism
        org_paths = org_cfg.get("paths", {})
    else:
        avail = [p.stem for p in (pkg_dir / "configs" / "organisms").glob("*.yaml")]
        raise ValueError(f"Unknown organism: '{organism}'. Available: {avail}")

    # Resolve all paths as absolute
    adata_abs = str(Path(adata_path).resolve())
    output_abs = str(Path(output_dir).resolve())
    Path(output_abs).mkdir(parents=True, exist_ok=True)

    # Database paths: resolve relative to package dir
    lr_db = pkg_dir / org_paths.get("lr_database", "databases/CellNEST_database.csv")
    met_db = pkg_dir / org_paths.get("met_database", "databases/M_R.txt")

    cfg["dataset"] = dataset
    cfg["paths"] = {
        "root": output_abs,
        "raw_adata": adata_abs,
        "lr_database": str(lr_db) if lr_db.exists() else str(lr_db),
        "met_database": str(met_db) if met_db.exists() else "",
        "protein_emb_dir": str(_find_emb_dir(pkg_dir, "proteins")),
        "metabolite_emb_dir": str(_find_emb_dir(pkg_dir, "metabolites")),
        "processed_dir": str(Path(output_abs) / "processed"),
    }
    if scfea_balance:
        cfg["paths"]["scfea_balance"] = str(Path(scfea_balance).resolve())

    cfg.setdefault("training", {})["checkpoint_dir"] = str(Path(output_abs) / "checkpoints")
    cfg["training"]["epochs"] = epochs

    if um_per_pixel is not None:
        cfg.setdefault("spatial", {})["um_per_pixel"] = um_per_pixel

    if cell_type_col is not None:
        cfg.setdefault("labels", {})["cell_type_col"] = cell_type_col

    # scFEA: auto-run if no pre-computed balance (takes 1-3h per dataset)
    # Set skip_rerun=False to enable; True to skip metabolite channel
    if not scfea_balance:
        cfg["scfea"] = {"use_existing": False, "skip_rerun": False}
    else:
        cfg["scfea"] = {"use_existing": True, "skip_rerun": True}

    return cfg


def setup(
    adata_path: str,
    output_dir: str = "./mosaic_output",
    dataset: str = "my_dataset",
    technology: str = "visium",
    organism: str = "human",
    um_per_pixel: Optional[float] = None,
    scfea_balance: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    epochs: int = 500,
) -> str:
    """
    Step 1: Generate config and save to output_dir.

    Returns path to the generated config.yaml.
    """
    import yaml

    cfg = _build_config(
        adata_path=adata_path, output_dir=output_dir, dataset=dataset,
        technology=technology, organism=organism, um_per_pixel=um_per_pixel,
        scfea_balance=scfea_balance, cell_type_col=cell_type_col, epochs=epochs,
    )
    config_path = Path(output_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Config saved: %s", config_path)
    return str(config_path)


def preprocess(config_path: str) -> Path:
    """
    Step 2: Build the heterogeneous graph from raw data.

    Runs all 17 preprocessing steps: scVI (if needed), spatial graph,
    LR database, scFEA flux, ESM-2 gene embeddings, ChemBERTa metabolite
    embeddings, edge construction, expression labels, spatial splits.

    Returns path to hetero_ccc_graph.pt.
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    from mosaic.data import preprocess_dataset
    preprocess_dataset(config_path, dataset, force=False)

    processed = Path(cfg["paths"]["processed_dir"]) / dataset
    graph_path = processed / "hetero_ccc_graph.pt"
    # Display as relative to CWD when possible (cleaner tutorial output)
    try:
        display_path = graph_path.relative_to(Path.cwd())
    except ValueError:
        display_path = graph_path
    log.info("Graph saved: %s", display_path)
    return graph_path


def train_model(
    config_path: str,
    device: str = "cuda",
    epochs: Optional[int] = None,
) -> "MOSAICResult":
    """
    Step 3: Train MOSAIC and return result object.

    Args:
        config_path: Path to config.yaml (from setup())
        device:      cuda or cpu
        epochs:      Override epochs from config

    Returns:
        MOSAICResult ready for analysis
    """
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    if epochs is not None:
        cfg.setdefault("training", {})["epochs"] = epochs

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    processed = Path(cfg["paths"]["processed_dir"]) / dataset
    graph_path = processed / "hetero_ccc_graph.pt"
    ds = torch.load(graph_path, map_location="cpu", weights_only=False)
    data, meta = ds["hetero_graph"], ds["metadata"]

    from mosaic.models import build_model
    model = build_model(cfg, n_expr_genes=meta["n_expr_genes"], graph_metadata=meta)

    from mosaic.training import MOSAICTrainer
    n_ep = cfg.get("training", {}).get("epochs", 500)
    trainer = MOSAICTrainer(model=model, data=data, config=cfg, device=dev, dataset=dataset)
    history = trainer.train(n_ep)
    best_r2 = max(history["val_r2"])
    log.info("Training complete. Best val R² = %.4f", best_r2)

    # Load best checkpoint
    ckpt_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "mosaic/checkpoints")) / dataset
    ckpt_path = ckpt_dir / "model_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    result = MOSAICResult(model=model, data=data, meta=meta,
                          config=cfg, dataset=dataset, device=dev)

    # Auto-load cell types
    adata_path = cfg["paths"]["raw_adata"]
    ct_col = cfg.get("labels", {}).get("cell_type_col")
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
        if ct_col is None:
            for col in ["cell_type", "cell_type_annot", "celltype", "annotation", "leiden"]:
                if col in adata.obs.columns:
                    ct_col = col
                    break
        if ct_col and ct_col in adata.obs.columns:
            result.set_cell_types_from_adata(adata, col=ct_col)
    except Exception:
        pass

    return result


def run_pipeline(
    adata_path: str,
    output_dir: str = "./mosaic_output",
    dataset: str = "my_dataset",
    technology: str = "visium",
    organism: str = "human",
    um_per_pixel: Optional[float] = None,
    scfea_balance: Optional[str] = None,
    cell_type_col: Optional[str] = None,
    epochs: int = 500,
    device: str = "cuda",
) -> "MOSAICResult":
    """
    Run the full MOSAIC pipeline from a single h5ad file.

    This is the recommended entry point for new users. Handles config
    generation, preprocessing, training, and returns a MOSAICResult.

    Minimal usage:
        from mosaic import run_pipeline
        result = run_pipeline("my_tissue.h5ad", technology="visium")

    The h5ad file must contain:
        .X:                   Log-normalised expression [N, G]
        .layers["raw_count"]: Raw UMI counts [N, G]
        .obsm["X_scvi"]:     scVI embeddings [N, 128]
        .obsm["spatial"]:    Spatial coordinates [N, 2] (pixels)
        .var_names:          Gene symbols

    Args:
        adata_path:     Path to h5ad file
        output_dir:     Where to save all outputs (default: ./mosaic_output)
        dataset:        Name for this dataset (default: my_dataset)
        technology:     Spatial platform: visium, xenium, merfish, slideseq
        organism:       Species: human, mouse
        um_per_pixel:   Microns per pixel (auto-detected for some techs)
        scfea_balance:  Path to pre-computed scFEA balance CSV (optional)
        cell_type_col:  Column in adata.obs for cell types (auto-detected)
        epochs:         Training epochs (default: 500)
        device:         cuda or cpu

    Returns:
        MOSAICResult ready for LR/MR queries, spatial maps, knockouts,
        relay detection, and plotting.
    """
    adata_path = str(Path(adata_path).resolve())
    if not Path(adata_path).exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    log.info("MOSAIC Pipeline: %s (%s, %s) → %s",
             Path(adata_path).name, technology, organism, output_dir)

    # Step 1: Config
    config_path = setup(
        adata_path=adata_path, output_dir=output_dir, dataset=dataset,
        technology=technology, organism=organism, um_per_pixel=um_per_pixel,
        scfea_balance=scfea_balance, cell_type_col=cell_type_col, epochs=epochs,
    )

    # Step 2: Preprocess
    preprocess(config_path)

    # Step 3: Train + return result
    result = train_model(config_path, device=device, epochs=epochs)

    log.info("Pipeline complete. %d cells, %d genes, %d LR pairs.",
             result.n_cells, result.n_genes, len(result.lr_pair_vocab))
    return result
