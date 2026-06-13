"""
metabolite.py — Map scFEA module outputs to metabolite sensor receptors.

Chain (CID-based, not name-based):
  scFEA module name
    → scFEA_compound.csv   : Compound_name → KEGG_ID  (e.g. Lactate → C00256)
    → metabolite_list.csv  : KEGG_ID → PubChem_int    (e.g. C00256 → 107689)
    → MR_Original_Interactions_updated.csv : PubChem_int → [receptor_gene_symbols]

Output:
  databases/scfea_module_receptor_map.json
    {module_name: [receptor_gene1, receptor_gene2, ...]}

Usage:
    python -m mosaic.data.metabolite \
        --config mosaic/configs/breast_config.yaml \
        --dataset breast_new
"""

import pandas as pd
import numpy as np
import json
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded supplements for metabolites missing from MRCLinkdb
# (important immune-metabolic signals not in the database)
# ---------------------------------------------------------------------------
HARDCODED_RECEPTORS = {
    # Lactate — immunosuppressive in TME
    'lactate':    ['HCAR1', 'GPR132', 'SLC16A1', 'SLC16A3', 'SLC16A7'],
    # GABA — increasingly recognised immune modulator
    'gaba':       ['GABRA1', 'GABRA2', 'GABRA3', 'GABRB1', 'GABRB2', 'GABBR1', 'GABBR2'],
    # Pyruvate — transport receptors
    'pyruvate':   ['SLC16A1', 'SLC16A3'],
    # Succinate — inflammation via SUCNR1
    'succinate':  ['SUCNR1'],
    # Fumarate — HIF pathway
    'fumarate':   [],
    # Citrate — extracellular signalling
    'citrate':    [],
    # Acetyl-CoA — no extracellular receptor, skip
    'acetyl-coa': [],
    # Fatty acid — FFA receptors
    'fatty acid': ['FFAR1', 'FFAR2', 'FFAR3', 'FFAR4', 'CD36'],
    # 2-oxoglutarate (alpha-ketoglutarate)
    '2og':        ['SUCNR1'],
}


def load_scfea_compound_map(compound_csv: str) -> Dict[str, str]:
    """
    Load scFEA_compound.csv and return {compound_name_lower: kegg_id}.

    Skips entries with:
      - Compound_ID = C00000 (unknown/placeholder)
      - Compound_ID containing '+' (multi-compound outputs)
      - Non-standard KEGG IDs (not starting with C or G)
    """
    df = pd.read_csv(compound_csv)
    compound_map = {}
    for _, row in df.iterrows():
        name = str(row['Compound_name']).strip().lower()
        kid  = str(row['Compound_ID']).strip()
        if kid == 'C00000' or '+' in kid:
            continue  # placeholder or multi-compound
        if not (kid.startswith('C') or kid.startswith('G')):
            continue  # non-standard ID
        compound_map[name] = kid
    logger.info(f"scFEA compound map: {len(compound_map)} entries with valid KEGG IDs")
    return compound_map


def load_kegg_to_pubchem(metabolite_list_csv: str) -> Dict[str, int]:
    """
    Load metabolite_list.csv and return {kegg_id: pubchem_int}.

    The KEGG column uses format like 'C00256'; PubChem column is integer.
    """
    df = pd.read_csv(metabolite_list_csv, index_col=0)
    kegg_to_pubchem = {}
    for _, row in df.iterrows():
        kegg = str(row.get('KEGG', '')).strip()
        pub  = row.get('PubChem', None)
        if kegg and kegg != 'nan' and pub is not None and not pd.isna(pub):
            try:
                kegg_to_pubchem[kegg] = int(pub)
            except (ValueError, TypeError):
                continue
    logger.info(f"KEGG→PubChem map: {len(kegg_to_pubchem)} entries")
    return kegg_to_pubchem


def load_pubchem_to_receptors(mr_csv: str,
                               expressed_genes: Optional[set] = None) -> Dict[int, List[str]]:
    """
    Load MR_Original_Interactions_updated.csv and return {pubchem_int: [receptor_genes]}.

    Args:
        mr_csv:          Path to MR_Original_Interactions_updated.csv
        expressed_genes: If provided, filter receptors to only those expressed in dataset.
    """
    df = pd.read_csv(mr_csv, index_col=0)
    pub_to_recs = {}
    for _, row in df.iterrows():
        pub = row.get('PubChem_id', None)
        rec = str(row.get('Receptor', '')).strip().upper()
        if pub is None or pd.isna(pub) or not rec or rec == 'NAN':
            continue
        try:
            pub_int = int(pub)
        except (ValueError, TypeError):
            continue
        if expressed_genes and rec not in expressed_genes:
            continue  # receptor not expressed in dataset
        if pub_int not in pub_to_recs:
            pub_to_recs[pub_int] = []
        if rec not in pub_to_recs[pub_int]:
            pub_to_recs[pub_int].append(rec)
    logger.info(f"PubChem→Receptor map: {len(pub_to_recs)} metabolites with receptors")
    return pub_to_recs


def build_module_receptor_map(
    scfea_modules:        List[str],
    compound_csv:         str,
    metabolite_list_csv:  str,
    mr_csv:               str,
    expressed_genes:      Optional[set] = None,
) -> Dict[str, List[str]]:
    """
    Build complete {scFEA_module_name: [receptor_gene_symbols]} mapping.

    Uses CID-based matching chain:
      module_name → KEGG_ID → PubChem_CID → receptor_symbols

    Falls back to hardcoded HARDCODED_RECEPTORS for key metabolites
    missing from MRCLinkdb (e.g., lactate, GABA).

    Args:
        scfea_modules:       List of column names from scFEA balance.csv
        compound_csv:        Path to scFEA_compound.csv
        metabolite_list_csv: Path to metabolite_list.csv
        mr_csv:              Path to MR_Original_Interactions_updated.csv
        expressed_genes:     Optional set of gene symbols expressed in dataset
                             (used to filter receptors to expressed ones only)

    Returns:
        {module_name: [receptor_gene_symbol, ...]}
        Modules with no known receptor are included with empty list.
    """
    # Load three-table chain
    compound_map   = load_scfea_compound_map(compound_csv)
    kegg_to_pubchem = load_kegg_to_pubchem(metabolite_list_csv)
    pub_to_recs    = load_pubchem_to_receptors(mr_csv, expressed_genes=expressed_genes)

    module_receptor_map = {}
    stats = {'db_matched': 0, 'hardcoded': 0, 'no_match': 0}

    for mod in scfea_modules:
        mod_lower = mod.lower().strip()
        receptors = []

        # --- Step 1: KEGG chain ---
        kegg_id = compound_map.get(mod_lower)
        if kegg_id:
            pubchem_id = kegg_to_pubchem.get(kegg_id)
            if pubchem_id:
                receptors = pub_to_recs.get(pubchem_id, [])
                if receptors:
                    stats['db_matched'] += 1
                    logger.debug(f"  {mod} → KEGG:{kegg_id} → PubChem:{pubchem_id} "
                                 f"→ {len(receptors)} receptors")

        # --- Step 2: Hardcoded supplement (for key missing metabolites) ---
        hard = HARDCODED_RECEPTORS.get(mod_lower, None)
        if hard is not None:
            # Merge: add hardcoded receptors not already found via DB
            added = [r for r in hard if r not in receptors]
            if expressed_genes:
                added = [r for r in added if r in expressed_genes]
            receptors = receptors + added
            if added and not receptors[:len(receptors)-len(added)]:
                stats['hardcoded'] += 1

        if not receptors:
            stats['no_match'] += 1

        module_receptor_map[mod] = receptors

    total = len(scfea_modules)
    mapped = total - stats['no_match']
    logger.info(
        f"Module mapping complete: {mapped}/{total} modules have ≥1 receptor "
        f"(db={stats['db_matched']}, hardcoded={stats['hardcoded']}, none={stats['no_match']})"
    )
    return module_receptor_map


def main():
    parser = argparse.ArgumentParser(
        description='Map scFEA modules to metabolite sensor receptors via CID chain'
    )
    parser.add_argument('--config',  required=True, help='Path to breast_config.yaml')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. breast_new)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg['paths']['project_root'])
    processed_dir = Path(cfg['paths']['processed_dir']) / args.dataset

    # Input paths
    compound_csv        = project_root / 'Databases' / 'scFEA_compound.csv'
    metabolite_list_csv = project_root / 'Databases' / 'metabolite_list.csv'
    mr_csv              = project_root / 'Databases' / 'MR_Original_Interactions_updated.csv'
    scfea_csv           = project_root / cfg['paths']['source_scfea']
    filtered_genes_json = processed_dir / 'filtered_genes.json'
    output_path         = processed_dir / 'databases' / 'scfea_module_receptor_map.json'

    # Load scFEA module names
    scfea_df = pd.read_csv(scfea_csv, index_col=0)
    scfea_modules = list(scfea_df.columns)
    logger.info(f"scFEA modules: {len(scfea_modules)}")

    # Optionally filter to expressed genes
    expressed_genes = None
    if filtered_genes_json.exists():
        with open(filtered_genes_json) as f:
            expressed_genes = set(json.load(f))
        logger.info(f"Expressed gene set: {len(expressed_genes)} genes")

    # Build mapping
    module_receptor_map = build_module_receptor_map(
        scfea_modules        = scfea_modules,
        compound_csv         = str(compound_csv),
        metabolite_list_csv  = str(metabolite_list_csv),
        mr_csv               = str(mr_csv),
        expressed_genes      = expressed_genes,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(module_receptor_map, f, indent=2)
    logger.info(f"Saved: {output_path}")

    # Print summary
    mapped = {k: v for k, v in module_receptor_map.items() if v}
    logger.info(f"Modules with receptors ({len(mapped)}/{len(module_receptor_map)}):")
    for mod, recs in sorted(mapped.items()):
        logger.info(f"  {mod}: {recs}")


if __name__ == '__main__':
    main()
