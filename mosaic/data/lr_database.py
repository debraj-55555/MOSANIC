"""
lr_database.py — Ligand-Receptor and Metabolite Database Loader

Loads, validates, normalises, and saves LR + metabolite databases.
Fixed from src3:
  - Column names validated by name, not position (robust to column order changes)
  - Gene symbols normalised (uppercase, stripped)
  - Source database column tracked
  - Symmetric duplicate removal (A→B and B→A treated as same pair)
  - Technology-aware filtering: for targeted panels (MERFISH), subsetting supported

Usage:
    python -m mosaic.data.lr_database \
        --lr_path data/databases/lr/CellNEST_database.csv \
        --met_path data/databases/metabolite/metabolite_db.txt \
        --output_dir data/processed/breast_new/ \
        [--gene_panel data/processed/breast_new/filtered_genes.json]
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging

from mosaic.data.channel_classifier import classify_lr_pair, classify_dataframe

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known column aliases for LR databases (handles CellChat, OmniPath, NicheNet)
# ---------------------------------------------------------------------------
LR_COLUMN_ALIASES = {
    'ligand':     ['ligand', 'Ligand', 'LIGAND', 'source', 'ligand_symbol', 'gene_a'],
    'receptor':   ['receptor', 'Receptor', 'RECEPTOR', 'target', 'receptor_symbol', 'gene_b'],
    'annotation': ['annotation', 'Annotation', 'pathway', 'interaction_type', 'category', 'type'],
    'reference':  ['reference', 'Reference', 'source_db', 'database', 'pmid', 'evidence'],
}

MET_COLUMN_ALIASES = {
    'hmdb_id':         ['HMDB ID', 'hmdb_id', 'HMDB', 'hmdb'],
    'metabolite_name': ['Metabolite name', 'metabolite_name', 'name', 'compound'],
    'smiles':          ['Canonical SMILES', 'smiles', 'SMILES', 'canonical_smiles'],
    'receptor_symbol': ['Receptor symbol', 'receptor_symbol', 'receptor', 'gene_symbol'],
    'pubchem_id':      ['PubChem CID/SID', 'pubchem_id', 'PubChem', 'pubchem_cid'],
}


def _resolve_column(df: pd.DataFrame, canonical: str, aliases: List[str]) -> Optional[str]:
    """Return the first matching alias column name found in df, or None."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _normalise_gene_symbol(s: pd.Series) -> pd.Series:
    """Uppercase and strip gene symbols (standard HGNC format)."""
    return s.astype(str).str.strip().str.upper()


class DatabaseLoader:
    """Load, validate, normalise, and save LR + metabolite databases."""

    def __init__(self, source_db_name: str = 'unknown'):
        self.lr_data = None
        self.metabolite_data = None
        self.stats = {}
        self.source_db_name = source_db_name  # Track which database this came from

    # ------------------------------------------------------------------
    # LR Database
    # ------------------------------------------------------------------
    def load_lr_database(self, filepath: str) -> pd.DataFrame:
        """
        Load ligand-receptor database from CSV.

        Accepts any CSV that has ligand + receptor columns (by name, not position).
        Normalises gene symbols to HGNC uppercase format.

        Supported databases:
            CellChatDB, OmniPath, NicheNet, CellNEST, LIANA
        """
        logger.debug(f"Loading LR database: {filepath}")
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"LR database not found: {filepath}")

        df = pd.read_csv(filepath)
        logger.debug(f"  Raw shape: {df.shape}, Columns: {list(df.columns)}")

        # --- Resolve columns by name, not position ---
        col_map = {}
        for canonical, aliases in LR_COLUMN_ALIASES.items():
            found = _resolve_column(df, canonical, aliases)
            if found is None and canonical in ('ligand', 'receptor'):
                raise ValueError(
                    f"Required column '{canonical}' not found in LR database.\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Expected one of: {aliases}"
                )
            col_map[found] = canonical

        df = df.rename(columns=col_map)

        # Keep only resolved columns + fill optional ones
        keep_cols = ['ligand', 'receptor']
        for opt in ['annotation', 'reference']:
            if opt in df.columns:
                keep_cols.append(opt)
            else:
                df[opt] = ''

        df = df[['ligand', 'receptor', 'annotation', 'reference']].copy()

        # --- Clean ---
        df = df.dropna(subset=['ligand', 'receptor'])
        df['ligand']     = _normalise_gene_symbol(df['ligand'])
        df['receptor']   = _normalise_gene_symbol(df['receptor'])
        df['annotation'] = df['annotation'].fillna('').astype(str).str.strip()
        df['reference']  = df['reference'].fillna('').astype(str).str.strip()

        # --- Remove clearly invalid entries ---
        df = df[df['ligand'] != '']
        df = df[df['receptor'] != '']
        df = df[df['ligand'] != 'NAN']
        df = df[df['receptor'] != 'NAN']

        # --- Track source database ---
        df['source_db'] = self.source_db_name

        logger.debug(f"  Loaded {len(df)} LR interactions after cleaning")
        self.lr_data = df
        return df

    def filter_lr_for_technology(
        self,
        gene_panel: Optional[List[str]] = None,
        technology: str = 'visium'
    ) -> pd.DataFrame:
        """
        Filter LR pairs to match the measured gene panel.

        For targeted technologies (MERFISH, CosMx), only LR pairs where
        BOTH ligand AND receptor are in the measured panel are kept.
        For untargeted (Visium, Slide-seq), no filtering applied.

        Args:
            gene_panel: List of measured gene symbols (None = no filtering)
            technology:  'visium' | 'merfish' | 'slide_seq' | 'stereo_seq'
        """
        if self.lr_data is None:
            raise ValueError("Load LR database first.")

        targeted_techs = {'merfish', 'cosmx', 'seqfish', 'seqfish_plus'}
        if technology.lower() in targeted_techs and gene_panel is not None:
            panel_upper = {g.upper() for g in gene_panel}
            before = len(self.lr_data)
            mask = (
                self.lr_data['ligand'].isin(panel_upper) &
                self.lr_data['receptor'].isin(panel_upper)
            )
            self.lr_data = self.lr_data[mask].copy()
            logger.debug(
                f"  Technology filter ({technology}): {before} → {len(self.lr_data)} LR pairs "
                f"(kept pairs where both genes are in {len(panel_upper)}-gene panel)"
            )
        else:
            logger.debug(f"  Technology ({technology}): No gene panel filtering applied")

        return self.lr_data

    # ------------------------------------------------------------------
    # Metabolite Database
    # ------------------------------------------------------------------
    def load_metabolite_database(self, filepath: str) -> pd.DataFrame:
        """
        Load metabolite-gene database from tab-delimited file.

        Resolves column names by alias (robust to column order/name changes).
        """
        logger.debug(f"Loading metabolite database: {filepath}")
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Metabolite database not found: {filepath}")

        df = pd.read_csv(filepath, sep='\t')
        logger.debug(f"  Raw shape: {df.shape}")

        # --- Resolve columns ---
        col_map = {}
        for canonical, aliases in MET_COLUMN_ALIASES.items():
            found = _resolve_column(df, canonical, aliases)
            if found is None and canonical in ('hmdb_id', 'receptor_symbol'):
                raise ValueError(
                    f"Required metabolite column '{canonical}' not found.\n"
                    f"Available: {list(df.columns)}\nExpected one of: {aliases}"
                )
            if found is not None:
                col_map[found] = canonical

        df = df.rename(columns=col_map)

        # Fill missing optional columns
        for opt in ['smiles', 'pubchem_id', 'metabolite_name']:
            if opt not in df.columns:
                df[opt] = ''

        df = df.dropna(subset=['hmdb_id', 'receptor_symbol'])

        # Normalise receptor gene symbols
        df['receptor_symbol'] = _normalise_gene_symbol(df['receptor_symbol'])
        df['hmdb_id'] = df['hmdb_id'].astype(str).str.strip()

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        logger.debug(f"  Loaded {len(df)} metabolite-gene interactions")
        self.metabolite_data = df
        return df

    # ------------------------------------------------------------------
    # Channel Type Assignment
    # ------------------------------------------------------------------
    def assign_channel_types(self, orig_lr_path: Optional[str] = None) -> pd.DataFrame:
        """
        Add a 'channel_type' column to self.lr_data.

        Strategy (in priority order):
          1. Annotation string from self.lr_data['annotation'] column
          2. Join with orig_lr_path CSV (LR_Original_Intercation_updated.csv)
             which has richer Annotation coverage for CellNEST pairs
          3. Gene-family prefix heuristic (channel_classifier.py)

        Args:
            orig_lr_path: Optional path to original annotated LR CSV
                          (e.g. Databases/LR_Original_Intercation_updated.csv).
                          Expected columns: Ligand, Receptor, Annotation.

        Returns:
            self.lr_data with 'channel_type' column added.
        """
        if self.lr_data is None:
            raise ValueError("Load LR database first.")

        df = self.lr_data.copy()

        # Step 1: Start with annotation already in the processed DataFrame
        df['channel_type'] = df['annotation'].apply(
            lambda a: classify_lr_pair('', '', a) if a else None
        )

        # Step 2: Supplement from original annotated LR database via join
        if orig_lr_path is not None:
            orig_path = Path(orig_lr_path)
            if orig_path.exists():
                logger.debug(f"  Loading original LR annotations from: {orig_path}")
                orig = pd.read_csv(orig_path, index_col=0) if orig_path.suffix == '.csv' else \
                       pd.read_csv(orig_path)
                # Normalise column names
                orig.columns = [c.strip() for c in orig.columns]
                lig_col = next((c for c in orig.columns if c.lower() in ('ligand', 'lig')), None)
                rec_col = next((c for c in orig.columns if c.lower() in ('receptor', 'rec')), None)
                ann_col = next((c for c in orig.columns if c.lower() in ('annotation', 'ann', 'category')), None)

                if lig_col and rec_col and ann_col:
                    orig_map = {
                        (str(r[lig_col]).upper().strip(),
                         str(r[rec_col]).upper().strip()): str(r[ann_col]).strip()
                        for _, r in orig.iterrows()
                        if str(r[ann_col]).strip() not in ('', 'nan', 'NaN')
                    }
                    logger.debug(f"  Original annotation map: {len(orig_map)} annotated pairs")

                    def _fill_from_orig(row):
                        if row['channel_type'] is not None and row['channel_type'] != 'secreted':
                            return row['channel_type']   # already set from existing annotation
                        ann = orig_map.get((row['ligand'], row['receptor']), '')
                        if ann:
                            ct = classify_lr_pair('', '', ann)
                            return ct
                        return row['channel_type']

                    df['channel_type'] = df.apply(_fill_from_orig, axis=1)
                else:
                    logger.warning(f"  Original LR CSV missing expected columns. Skipping join.")
            else:
                logger.warning(f"  orig_lr_path not found: {orig_path}. Skipping join.")

        # Step 3: Gene-family heuristic for remaining None / unclassified
        unclassified_mask = df['channel_type'].isna()
        if unclassified_mask.any():
            logger.debug(f"  Applying gene-family heuristic to {unclassified_mask.sum()} unclassified pairs")
            df.loc[unclassified_mask, 'channel_type'] = df[unclassified_mask].apply(
                lambda r: classify_lr_pair(r['ligand'], r['receptor'], ''), axis=1
            )

        # Log distribution
        dist = df['channel_type'].value_counts().to_dict()
        logger.debug(f"  Channel type distribution: {dist}")

        self.lr_data = df
        return self.lr_data

    # ------------------------------------------------------------------
    # Duplicate Removal (symmetric)
    # ------------------------------------------------------------------
    def remove_duplicates(self, symmetric: bool = True):
        """
        Remove duplicate interactions.

        Args:
            symmetric: If True, treat (A, B) and (B, A) as duplicates for LR pairs.
                       Default True — standard for undirected CCC graphs.
        """
        if self.lr_data is not None:
            before = len(self.lr_data)
            if symmetric:
                # Create canonical pair key: always (min, max) alphabetically
                self.lr_data['_key'] = self.lr_data.apply(
                    lambda r: tuple(sorted([r['ligand'], r['receptor']])), axis=1
                )
                self.lr_data = self.lr_data.drop_duplicates(subset=['_key'], keep='first')
                self.lr_data = self.lr_data.drop(columns=['_key'])
            else:
                self.lr_data = self.lr_data.drop_duplicates(
                    subset=['ligand', 'receptor'], keep='first'
                )
            logger.debug(f"LR dedup ({'symmetric' if symmetric else 'directed'}): "
                        f"{before} → {len(self.lr_data)}")

        if self.metabolite_data is not None:
            before = len(self.metabolite_data)
            self.metabolite_data = self.metabolite_data.drop_duplicates(
                subset=['hmdb_id', 'receptor_symbol'], keep='first'
            )
            logger.debug(f"Metabolite dedup: {before} → {len(self.metabolite_data)}")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def compute_statistics(self) -> Dict:
        stats = {}

        if self.lr_data is not None:
            stats['lr'] = {
                'total_interactions': int(len(self.lr_data)),
                'unique_ligands':     int(self.lr_data['ligand'].nunique()),
                'unique_receptors':   int(self.lr_data['receptor'].nunique()),
                'unique_genes':       int(len(
                    set(self.lr_data['ligand']) | set(self.lr_data['receptor'])
                )),
                'source_db': self.source_db_name,
            }

        if self.metabolite_data is not None:
            stats['metabolite'] = {
                'total_interactions':  int(len(self.metabolite_data)),
                'unique_metabolites':  int(self.metabolite_data['hmdb_id'].nunique()),
                'unique_receptors':    int(self.metabolite_data['receptor_symbol'].nunique()),
                'with_smiles':         int((self.metabolite_data['smiles'] != '').sum()),
            }

        if self.lr_data is not None and self.metabolite_data is not None:
            lr_genes  = set(self.lr_data['ligand']) | set(self.lr_data['receptor'])
            met_genes = set(self.metabolite_data['receptor_symbol'])
            stats['overlap'] = {
                'total_unique_genes': int(len(lr_genes | met_genes)),
                'genes_in_both':      int(len(lr_genes & met_genes)),
                'genes_only_lr':      int(len(lr_genes - met_genes)),
                'genes_only_met':     int(len(met_genes - lr_genes)),
            }

        self.stats = stats
        return stats

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_processed_databases(self, output_dir: str,
                                   orig_lr_path: Optional[str] = None):
        """
        Save processed databases + universe files.

        Outputs:
            processed_lr_database.csv       (now includes 'channel_type' column)
            processed_metabolite_database.csv
            database_statistics.json
            gene_universe.json     (all LR + metabolite receptor genes)
            metabolite_universe.json

        Args:
            output_dir:   Directory to write output files.
            orig_lr_path: Optional path to original annotated LR CSV for richer
                          channel_type annotation (LR_Original_Intercation_updated.csv).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving to: {output_dir}")

        if self.lr_data is not None:
            # Assign channel types before saving
            if 'channel_type' not in self.lr_data.columns:
                self.assign_channel_types(orig_lr_path=orig_lr_path)
            self.lr_data.to_csv(output_dir / 'processed_lr_database.csv', index=False)
            logger.info(f"  Saved: processed_lr_database.csv (with channel_type)")

        if self.metabolite_data is not None:
            self.metabolite_data.to_csv(
                output_dir / 'processed_metabolite_database.csv', index=False
            )
            logger.info(f"  Saved: processed_metabolite_database.csv")

        if not self.stats:
            self.compute_statistics()

        with open(output_dir / 'database_statistics.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"  Saved: database_statistics.json")

        if self.lr_data is not None and self.metabolite_data is not None:
            lr_genes  = set(self.lr_data['ligand']) | set(self.lr_data['receptor'])
            met_genes = set(self.metabolite_data['receptor_symbol'])
            all_genes = sorted(lr_genes | met_genes)

            with open(output_dir / 'gene_universe.json', 'w') as f:
                json.dump(all_genes, f, indent=2)
            logger.info(f"  Saved: gene_universe.json ({len(all_genes)} genes)")

        if self.metabolite_data is not None:
            metabolites = sorted(self.metabolite_data['hmdb_id'].unique().tolist())
            with open(output_dir / 'metabolite_universe.json', 'w') as f:
                json.dump(metabolites, f, indent=2)
            logger.info(f"  Saved: metabolite_universe.json ({len(metabolites)} metabolites)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Load and process LR + metabolite databases')
    parser.add_argument('--lr_path',    required=True,  help='Path to LR database CSV')
    parser.add_argument('--met_path',   required=True,  help='Path to metabolite database TSV')
    parser.add_argument('--output_dir', required=True,  help='Output directory')
    parser.add_argument('--source_db',  default='unknown', help='Database source name (e.g. CellChatDB)')
    parser.add_argument('--gene_panel', default=None,
                        help='Path to filtered_genes.json for technology-aware filtering')
    parser.add_argument('--technology', default='visium',
                        choices=['visium', 'visium_hd', 'merfish', 'slide_seq', 'stereo_seq', 'cosmx'],
                        help='Spatial technology (affects gene panel filtering)')
    parser.add_argument('--orig_lr_path', default=None,
                        help='Path to original annotated LR CSV for channel_type enrichment '
                             '(e.g. Databases/LR_Original_Intercation_updated.csv)')
    args = parser.parse_args()

    loader = DatabaseLoader(source_db_name=args.source_db)
    loader.load_lr_database(args.lr_path)
    loader.load_metabolite_database(args.met_path)

    # Technology-aware gene panel filtering
    gene_panel = None
    if args.gene_panel:
        with open(args.gene_panel) as f:
            gene_panel = json.load(f)
        logger.debug(f"Gene panel loaded: {len(gene_panel)} genes")
    loader.filter_lr_for_technology(gene_panel=gene_panel, technology=args.technology)

    loader.remove_duplicates(symmetric=True)
    loader.compute_statistics()
    loader.save_processed_databases(args.output_dir, orig_lr_path=args.orig_lr_path)

    logger.info("Database processing complete")


if __name__ == '__main__':
    main()
