"""
mosanic/graph/node_features.py

Builds node feature tensors for all three node types:
  cell        [N, 128]   — scVI latent embeddings (loaded in preprocess step 1)
  gene        [G, 1280]  — ESM-2 protein embeddings
  metabolite  [M, 600]   — ChemBERTa SMILES embeddings

Embeddings are loaded from cached .npy files. If missing, they are auto-computed:
  - ESM-2: requires `fair-esm` package (auto-installs via pip)
  - ChemBERTa: requires `transformers` package
If the model cannot be loaded, zero vectors are used as fallback.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class NodeFeatureBuilder:
    """
    Builds node feature matrices from cached or auto-computed embeddings.

    Tries cached .npy files first. If missing, auto-computes using:
      - ESM-2 (esm2_t33_650M_UR50D) for gene embeddings
      - ChemBERTa (DeepChem/ChemBERTa-77M-MTR) for metabolite embeddings

    Args:
        protein_emb_dir:    directory for cached/computed gene .npy files
        metabolite_emb_dir: directory for cached/computed metabolite .npy files
        gene_dim:           embedding dimension for genes (default 1280)
        metabolite_dim:     embedding dimension for metabolites (default 600)
        device:             torch device for model inference (default: auto)
        smiles_csv:         path to metabolite_list.csv with SMILES (optional)
    """

    def __init__(
        self,
        protein_emb_dir: str,
        metabolite_emb_dir: str,
        gene_dim: int = 1280,
        metabolite_dim: int = 600,
        device: str = "auto",
        smiles_csv: Optional[str] = None,
        organism: str = "human",
    ):
        self.protein_emb_dir    = Path(protein_emb_dir)
        self.metabolite_emb_dir = Path(metabolite_emb_dir)
        self.gene_dim           = gene_dim
        self.metabolite_dim     = metabolite_dim
        self.device             = device
        self.smiles_csv         = smiles_csv
        self.organism           = organism

        # Lazy-loaded models
        self._esm_model = None
        self._esm_alphabet = None
        self._chemberta_model = None
        self._chemberta_tokenizer = None

        # Ensure cache dirs exist
        self.protein_emb_dir.mkdir(parents=True, exist_ok=True)
        self.metabolite_emb_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Gene node features  (ESM-2)
    # ─────────────────────────────────────────────────────────────────────

    def build_gene_features(
        self, gene_names: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Build gene node feature matrix [G, gene_dim].

        1. Tries cached {GENE_NAME}.npy files
        2. If any missing, auto-computes with ESM-2 and caches
        3. Falls back to zero vectors only if ESM-2 unavailable

        Returns:
            features: [G, gene_dim] float32 tensor
            vocab:    ordered list of gene names (same order as rows)
        """
        features = []
        missing = []

        for gene in gene_names:
            vec = self._load_protein_emb(gene)
            if vec is None:
                missing.append(gene)
            features.append(vec)

        # Auto-compute missing with ESM-2
        if missing:
            log.debug(f"  {len(missing)}/{len(gene_names)} gene embeddings missing — computing with ESM-2...")
            computed = self._compute_esm2_embeddings(missing)
            # Fill in computed vectors
            miss_idx = 0
            for i, vec in enumerate(features):
                if vec is None:
                    gene = gene_names[i]
                    features[i] = computed.get(gene.upper(), np.zeros(self.gene_dim, dtype=np.float32))
                    miss_idx += 1

            still_missing = sum(1 for v in features if v is None or np.all(v == 0))
            if still_missing > 0 and still_missing < len(gene_names):
                log.debug(f"  ESM-2 computed {len(missing) - still_missing}/{len(missing)}, "
                         f"{still_missing} zero-filled")

        # Replace any remaining None with zeros
        for i, vec in enumerate(features):
            if vec is None:
                features[i] = np.zeros(self.gene_dim, dtype=np.float32)

        feat_tensor = torch.from_numpy(np.stack(features, axis=0))
        log.debug(f"  Gene node features: {feat_tensor.shape}")
        return feat_tensor, list(gene_names)

    def _compute_esm2_embeddings(self, gene_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute ESM-2 embeddings for a list of gene symbols.

        Uses the esm package to generate protein embeddings from sequences.
        Results are cached as .npy files for future use.
        """
        try:
            import esm
        except ImportError:
            log.warning("  ESM-2 auto-compute requires 'fair-esm'. Install: pip install fair-esm")
            log.warning("  Falling back to zero vectors for %d genes.", len(gene_list))
            return {}

        if self._esm_model is None:
            log.debug("  Loading ESM-2 model (esm2_t33_650M_UR50D)...")
            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._esm_model, self._esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self._esm_model = self._esm_model.to(dev).eval()
            self._esm_batch_converter = self._esm_alphabet.get_batch_converter()
            self._esm_device = torch.device(dev)
            log.debug("  ESM-2 loaded on %s", dev)

        # We need gene → protein sequence mapping.
        # Try to get sequences from UniProt via simple heuristic:
        # use the gene symbol as a pseudo-sequence query.
        # For production: provide uniprot_tsv; for now, try biopython or skip.
        embeddings = {}
        batch_size = 4
        max_length = 1022  # ESM-2 max

        # Try fetching sequences from UniProt API
        sequences = self._fetch_sequences(gene_list)
        if not sequences:
            log.warning("  Could not fetch protein sequences. Zero-filling %d genes.", len(gene_list))
            return {}

        gene_seq_pairs = [(g, sequences[g]) for g in gene_list if g.upper() in sequences]
        if not gene_seq_pairs:
            return {}

        log.debug("  Computing ESM-2 embeddings for %d/%d genes...", len(gene_seq_pairs), len(gene_list))

        for i in range(0, len(gene_seq_pairs), batch_size):
            batch = gene_seq_pairs[i:i + batch_size]
            # Truncate sequences
            data = [(g, seq[:max_length]) for g, seq in batch]
            _, _, tokens = self._esm_batch_converter(data)
            tokens = tokens.to(self._esm_device)

            with torch.no_grad():
                results = self._esm_model(tokens, repr_layers=[33], return_contacts=False)

            reps = results["representations"][33]  # [B, seq_len, 1280]

            for j, (gene, _) in enumerate(data):
                # Mean-pool over sequence (excluding BOS/EOS)
                seq_len = min(len(batch[j][1]), max_length)
                vec = reps[j, 1:seq_len + 1].mean(dim=0).cpu().numpy().astype(np.float32)
                embeddings[gene.upper()] = vec

                # Cache
                cache_path = self.protein_emb_dir / f"{gene.upper()}.npy"
                np.save(cache_path, vec)

        log.debug("  ESM-2: computed and cached %d embeddings", len(embeddings))
        return embeddings

    def _fetch_sequences(self, gene_list: List[str]) -> Dict[str, str]:
        """
        Fetch protein sequences for genes.

        Tries in order:
        1. Local UniProt TSV if available
        2. Ensembl/UniProt REST API (batch query)
        3. Returns empty dict if both fail
        """
        # Check for local UniProt TSV
        pkg_dir = Path(__file__).parent.parent
        uniprot_candidates = [
            pkg_dir / "databases" / "uniprot_human.tsv",
            pkg_dir / "databases" / "uniprot_human.tsv.gz",
            pkg_dir.parent / "Databases" / "uniprot_human.tsv.gz",
        ]
        for tsv_path in uniprot_candidates:
            if tsv_path.exists():
                return self._load_uniprot_tsv(tsv_path, gene_list)

        # Try REST API
        try:
            return self._fetch_sequences_api(gene_list, organism=self.organism)
        except Exception as e:
            log.warning("  Could not fetch sequences from API: %s", e)
            return {}

    def _load_uniprot_tsv(self, tsv_path: Path, gene_list: List[str]) -> Dict[str, str]:
        """Load sequences from a local UniProt TSV file."""
        import pandas as pd
        log.debug("  Loading UniProt sequences from %s", tsv_path)
        compression = "gzip" if str(tsv_path).endswith(".gz") else None
        df = pd.read_csv(tsv_path, sep="\t", compression=compression)

        gene_set = {g.upper() for g in gene_list}
        sequences = {}

        if "Gene Names" in df.columns and "Sequence" in df.columns:
            reviewed = df[df.get("Reviewed", "reviewed") == "reviewed"] if "Reviewed" in df.columns else df
            for _, row in reviewed.iterrows():
                names = str(row.get("Gene Names", "")).split()
                seq = str(row.get("Sequence", ""))
                if not seq or seq == "nan":
                    continue
                for name in names:
                    if name.upper() in gene_set and name.upper() not in sequences:
                        sequences[name.upper()] = seq

        log.debug("  UniProt: found sequences for %d/%d genes", len(sequences), len(gene_list))
        return sequences

    def _fetch_sequences_api(self, gene_list: List[str], organism: str = "human") -> Dict[str, str]:
        """Fetch sequences from UniProt REST API in batches."""
        import urllib.request
        import json

        org_id = "10090" if organism == "mouse" else "9606"
        sequences = {}
        batch_size = 50
        gene_set = set(g.upper() for g in gene_list)

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i + batch_size]
            query = "+OR+".join(f"gene_exact:{g}" for g in batch)
            url = (f"https://rest.uniprot.org/uniprotkb/search?"
                   f"query=({query})+AND+(organism_id:{org_id})+AND+(reviewed:true)"
                   f"&fields=gene_names,sequence&format=json&size=500")
            try:
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                for entry in data.get("results", []):
                    names = []
                    for gn in entry.get("genes", []):
                        name = gn.get("geneName", {}).get("value", "")
                        if name:
                            names.append(name.upper())
                    seq = entry.get("sequence", {}).get("value", "")
                    if seq:
                        for name in names:
                            if name in gene_set and name not in sequences:
                                sequences[name] = seq
            except Exception:
                continue  # Skip batch on error

        log.debug("  UniProt API: fetched sequences for %d/%d genes", len(sequences), len(gene_list))
        return sequences

    def _load_protein_emb(self, gene: str) -> np.ndarray | None:
        """Try loading {gene}.npy with case variants."""
        for name in [gene, gene.upper(), gene.lower()]:
            path = self.protein_emb_dir / f"{name}.npy"
            if path.exists():
                vec = np.load(path).astype(np.float32)
                if vec.ndim > 1:
                    vec = vec.mean(axis=0)   # some ESM-2 outputs are [seq_len, dim]
                if len(vec) == self.gene_dim:
                    return vec
                # Dimension mismatch — try to truncate/pad
                if len(vec) > self.gene_dim:
                    return vec[:self.gene_dim]
                padded = np.zeros(self.gene_dim, dtype=np.float32)
                padded[:len(vec)] = vec
                return padded
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Metabolite node features  (ChemBERTa)
    # ─────────────────────────────────────────────────────────────────────

    def build_metabolite_features(
        self, module_names: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Build metabolite node feature matrix [M, metabolite_dim].

        1. Tries cached .npy files
        2. Auto-computes with ChemBERTa if SMILES available
        3. Falls back to zero vectors

        Returns:
            features: [M, metabolite_dim] float32 tensor
            vocab:    ordered list of module names (same order as rows)
        """
        if not module_names:
            log.debug("  No metabolite features (empty module list)")
            return torch.zeros(0, self.metabolite_dim), []

        features = []
        missing = []

        for mod in module_names:
            vec = self._load_metabolite_emb(mod)
            if vec is None:
                missing.append(mod)
            features.append(vec)

        # Auto-compute missing with ChemBERTa
        if missing:
            log.debug(f"  {len(missing)}/{len(module_names)} metabolite embeddings missing "
                     f"— computing with ChemBERTa...")
            computed = self._compute_chemberta_embeddings(missing)
            for i, vec in enumerate(features):
                if vec is None:
                    mod = module_names[i]
                    features[i] = computed.get(mod.lower(),
                                  computed.get(mod, np.zeros(self.metabolite_dim, dtype=np.float32)))

        # Replace remaining None
        for i, vec in enumerate(features):
            if vec is None:
                features[i] = np.zeros(self.metabolite_dim, dtype=np.float32)

        feat_tensor = torch.from_numpy(np.stack(features, axis=0))
        log.debug(f"  Metabolite node features: {feat_tensor.shape}")
        return feat_tensor, list(module_names)

    def _compute_chemberta_embeddings(self, module_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute ChemBERTa embeddings for metabolites using their SMILES strings.

        Looks up SMILES from metabolite_list.csv (bundled with package).
        """
        # Load SMILES mapping
        smiles_map = self._load_smiles_map()
        if not smiles_map:
            log.warning("  No SMILES data available. Zero-filling %d metabolites.", len(module_names))
            return {}

        # Find SMILES for missing metabolites
        to_compute = {}
        for mod in module_names:
            smiles = smiles_map.get(mod.lower()) or smiles_map.get(mod)
            if smiles:
                to_compute[mod] = smiles

        if not to_compute:
            log.warning("  No SMILES found for any missing metabolites.")
            return {}

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError:
            log.warning("  ChemBERTa requires 'transformers'. Install: pip install transformers")
            return {}

        if self._chemberta_model is None:
            log.debug("  Loading ChemBERTa model...")
            dev = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self._chemberta_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
            self._chemberta_model = AutoModelForMaskedLM.from_pretrained(
                "DeepChem/ChemBERTa-77M-MTR", use_safetensors=True
            ).to(dev).eval()
            self._chemberta_device = torch.device(dev)

        embeddings = {}
        for mod, smiles in to_compute.items():
            try:
                inputs = self._chemberta_tokenizer(smiles, return_tensors="pt",
                                                    truncation=True, max_length=512,
                                                    padding=True)
                inputs = {k: v.to(self._chemberta_device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._chemberta_model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]  # last layer
                    vec = hidden[0].mean(dim=0).cpu().numpy().astype(np.float32)

                # Truncate/pad to expected dim
                if len(vec) > self.metabolite_dim:
                    vec = vec[:self.metabolite_dim]
                elif len(vec) < self.metabolite_dim:
                    padded = np.zeros(self.metabolite_dim, dtype=np.float32)
                    padded[:len(vec)] = vec
                    vec = padded

                embeddings[mod] = vec
                # Cache
                cache_path = self.metabolite_emb_dir / f"{mod}.npy"
                np.save(cache_path, vec)
            except Exception as e:
                log.debug("  ChemBERTa failed for %s: %s", mod, e)

        log.debug("  ChemBERTa: computed %d/%d metabolite embeddings", len(embeddings), len(to_compute))
        return embeddings

    def _load_smiles_map(self) -> Dict[str, str]:
        """
        Load metabolite name → SMILES mapping from the bundled scFEA module SMILES file.

        The file scfea_module_smiles.csv has pre-curated SMILES for all 70 scFEA
        metabolic modules (66/70 have SMILES; 4 glycan structures excluded).
        """
        import pandas as pd

        pkg_dir = Path(__file__).parent.parent

        # Primary: dedicated module SMILES file (complete for all 70 modules)
        for p in [
            pkg_dir / "databases" / "scfea_module_smiles.csv",
            self.smiles_csv if self.smiles_csv else "",
        ]:
            if p and Path(p).exists():
                df = pd.read_csv(Path(p))
                smiles_map = {}
                for _, row in df.iterrows():
                    name = str(row.get("module_name", "")).strip()
                    smiles = str(row.get("smiles", "")).strip()
                    if name and smiles and smiles != "nan":
                        smiles_map[name] = smiles
                        smiles_map[name.lower()] = smiles
                if smiles_map:
                    log.debug("SMILES map: %d metabolites", len(smiles_map) // 2)
                    return smiles_map

        return {}

    def _load_metabolite_emb(self, module_name: str) -> np.ndarray | None:
        """Try loading {module_name}.npy with normalised filename variants."""
        import re
        def _sanitize(s: str) -> str:
            return re.sub(r'[/\\:*?"<>|]', '_', s)
        candidates = [
            module_name,
            module_name.replace(" ", "_"),
            module_name.replace("-", "_"),
            module_name.lower(),
            module_name.lower().replace(" ", "_"),
            _sanitize(module_name),
            _sanitize(module_name.lower()),
            _sanitize(module_name).replace(" ", "_"),
            _sanitize(module_name.lower()).replace(" ", "_"),
        ]
        for name in candidates:
            path = self.metabolite_emb_dir / f"{name}.npy"
            if path.exists():
                vec = np.load(path).astype(np.float32)
                if vec.ndim > 1:
                    vec = vec.mean(axis=0)
                if len(vec) == self.metabolite_dim:
                    return vec
                if len(vec) > self.metabolite_dim:
                    return vec[:self.metabolite_dim]
                padded = np.zeros(self.metabolite_dim, dtype=np.float32)
                padded[:len(vec)] = vec
                return padded
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Utility: coverage check
    # ─────────────────────────────────────────────────────────────────────

    def check_coverage(self, gene_names: List[str], module_names: List[str]) -> dict:
        """Report embedding coverage before building."""
        gene_found = sum(1 for g in gene_names if self._load_protein_emb(g) is not None)
        met_found  = sum(1 for m in module_names if self._load_metabolite_emb(m) is not None)
        return {
            "gene_total": len(gene_names),
            "gene_found": gene_found,
            "gene_coverage": gene_found / max(len(gene_names), 1),
            "metabolite_total": len(module_names),
            "metabolite_found": met_found,
            "metabolite_coverage": met_found / max(len(module_names), 1),
        }
