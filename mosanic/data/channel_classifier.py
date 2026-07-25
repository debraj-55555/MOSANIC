"""
channel_classifier.py — Biological signalling-channel label for each LR pair.

Labels each ligand-receptor pair with its signalling channel, using:
  1. The database's own interaction-type annotation, if present. The training catalogue
     used in the paper (CellNEST) already tags pairs as secreted / contact / ECM;
     common CellChat-style strings (Secreted Signaling / Cell-Cell Contact / ECM-Receptor)
     are recognised too.
  2. A gene-family prefix heuristic (for pairs with no annotation).
  3. Fallback: 'secreted' (most common and most conservative).

Channel labels:
  'secreted'  — paracrine secreted signalling. This is the only LR channel kept in the
                final model, where it becomes the cell-cell edge type τ₁.
  'contact'   — juxtacrine / direct cell-cell contact.
  'ecm'       — ECM-receptor interactions.
Note: 'contact' and 'ecm' are labelled here for completeness, but they are NOT separate
edges in the final 7-edge graph (contact is a topological subset of secreted, and ECM is
sparse/absent in several datasets), so they are dropped during graph assembly. The paper's
τ₃ edge is the intracellular self-loop, unrelated to these LR channel labels.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Annotation string → canonical channel type
# ---------------------------------------------------------------------------
ANNOTATION_MAP = {
    # Secreted
    'secreted signaling': 'secreted',
    'secreted':           'secreted',
    'paracrine':          'secreted',
    'cytokine':           'secreted',
    'growth factor':      'secreted',
    'chemokine':          'secreted',
    'hormone':            'secreted',
    # Contact
    'cell-cell contact':  'contact',
    'cell contact':       'contact',
    'contact':            'contact',
    'juxtacrine':         'contact',
    'junction':           'contact',
    # ECM
    'ecm-receptor':       'ecm',
    'ecm receptor':       'ecm',
    'extracellular matrix': 'ecm',
    'matrix':             'ecm',
}

# ---------------------------------------------------------------------------
# Gene family prefix → channel type heuristic
# ---------------------------------------------------------------------------

# Contact signaling: membrane-anchored ligands and their receptors
CONTACT_LIGAND_PREFIXES = (
    'CDH',    # Cadherins (E-cadherin CDH1, N-cadherin CDH2)
    'NOTCH',  # Notch receptors (also act as ligands)
    'JAG',    # Jagged (Notch ligands)
    'DLL',    # Delta-like (Notch ligands)
    'EFNA',   # Ephrin-A (GPI-anchored, cell contact)
    'EFNB',   # Ephrin-B (transmembrane)
    'EPHA',   # Eph receptors
    'EPHB',
    'SEMA',   # Semaphorins (some membrane-bound)
    'PLXN',   # Plexins
    'ICAM',   # Intercellular adhesion molecules
    'VCAM',
    'PECAM',
    'NCAM',
    'ALCAM',
    'MCAM',
    'CADM',   # Cell adhesion molecule
    'NECTIN',
    'PVRL',   # Nectin family
    'ITGA',   # Integrins (membrane-bound receptors)
    'ITGB',
    'APP',    # Amyloid precursor protein
    'BST',
    'GAS',
    'TIGIT',
    'LAIR',
    'LILR',
    'SIGLEC',
    'CD200',
)

CONTACT_RECEPTOR_PREFIXES = (
    'NOTCH',
    'EPHA', 'EPHB',
    'ITGA', 'ITGB',
    'PLXN',
    'NRXN',  # Neurexins
)

# ECM signaling: structural matrix proteins
ECM_LIGAND_PREFIXES = (
    'COL',    # Collagens (COL1A1, COL4A1, etc.)
    'FN',     # Fibronectin
    'LAMA',   # Laminin alpha
    'LAMB',   # Laminin beta
    'LAMC',   # Laminin gamma
    'VTN',    # Vitronectin
    'TNC',    # Tenascin-C
    'TNN',    # Tenascin-N
    'FBN',    # Fibrillin
    'POSTN',  # Periostin
    'SPARC',
    'THBS',   # Thrombospondin
    'COMP',   # Cartilage oligomeric matrix protein
    'VCAN',   # Versican
    'HSPG',   # Heparan sulfate proteoglycan
    'AGRN',   # Agrin
    'NIDOGEN',
    'NID',    # Nidogen
    'HAPLN',  # Hyaluronan and proteoglycan link protein
)

ECM_RECEPTOR_PREFIXES = (
    'ITGA', 'ITGB',  # Integrins are main ECM receptors
    'CD44',
    'SDC',   # Syndecans
    'GPC',   # Glypicans
)

# Secreted signaling: clearly secreted proteins
SECRETED_LIGAND_PREFIXES = (
    'CCL',   # CC chemokines
    'CXCL',  # CXC chemokines
    'CX3CL',
    'XCL',
    'IL',    # Interleukins
    'TNF',
    'TNFSF',
    'TGFB',
    'BMP',
    'WNT',
    'FGF',
    'PDGF',
    'VEGF',
    'EGF',
    'EGFL',
    'IGF',
    'HGF',
    'ANGPT',
    'NRG',   # Neuregulin
    'NTF',   # Neurotrophin
    'BDNF',
    'NGF',
    'CNTF',
    'LIF',
    'OSM',
    'CSF',
    'KITLG',
    'THPO',
    'EPO',
    'GH',
    'INS',
    'INSL',
    'PTH',
    'PRLH',
    'SLIT',
    'NTN',   # Netrin
    'SEMA',  # Some secreted semaphorins
)


def classify_by_annotation(annotation: str) -> Optional[str]:
    """
    Map annotation string to channel type.

    Returns channel type string or None if not recognised.
    """
    if not annotation or str(annotation).strip() in ('', 'nan', 'NaN'):
        return None
    ann_lower = str(annotation).lower().strip()
    # Direct lookup
    if ann_lower in ANNOTATION_MAP:
        return ANNOTATION_MAP[ann_lower]
    # Substring match
    for key, ctype in ANNOTATION_MAP.items():
        if key in ann_lower:
            return ctype
    return None


def classify_by_gene_family(ligand: str, receptor: str) -> str:
    """
    Classify LR pair into channel type using gene family prefix heuristic.

    Priority: contact > ecm > secreted > secreted (fallback)

    Args:
        ligand:   Ligand gene symbol (HGNC uppercase)
        receptor: Receptor gene symbol (HGNC uppercase)

    Returns:
        'contact' | 'secreted' | 'ecm'
    """
    lig = str(ligand).upper().strip()
    rec = str(receptor).upper().strip()

    # --- Check Contact ---
    if _starts_with_any(lig, CONTACT_LIGAND_PREFIXES) or \
       _starts_with_any(rec, CONTACT_RECEPTOR_PREFIXES):
        return 'contact'

    # --- Check ECM ---
    if _starts_with_any(lig, ECM_LIGAND_PREFIXES) or \
       _starts_with_any(rec, ECM_RECEPTOR_PREFIXES):
        return 'ecm'

    # --- Check Secreted (explicit) ---
    if _starts_with_any(lig, SECRETED_LIGAND_PREFIXES):
        return 'secreted'

    # --- Fallback ---
    return 'secreted'


def classify_lr_pair(ligand: str, receptor: str, annotation: str = '') -> str:
    """
    Full classification pipeline for a single LR pair.

    1. Try annotation string first (most reliable)
    2. Fall back to gene family heuristic
    3. Final fallback: 'secreted'

    Args:
        ligand:     Ligand gene symbol
        receptor:   Receptor gene symbol
        annotation: Optional annotation string from database

    Returns:
        'contact' | 'secreted' | 'ecm'
    """
    # Try annotation first
    ctype = classify_by_annotation(annotation)
    if ctype is not None:
        return ctype

    # Fall back to gene family
    return classify_by_gene_family(ligand, receptor)


def _starts_with_any(gene: str, prefixes: tuple) -> bool:
    """Return True if gene starts with any of the given prefixes."""
    return any(gene.startswith(p) for p in prefixes)


def classify_dataframe(df, ligand_col='ligand', receptor_col='receptor',
                        annotation_col='annotation') -> list:
    """
    Classify all rows in a DataFrame.

    Args:
        df: DataFrame with ligand/receptor/annotation columns
        ligand_col, receptor_col, annotation_col: column names

    Returns:
        List of channel_type strings, one per row.
    """
    types = []
    for _, row in df.iterrows():
        ann = row.get(annotation_col, '') if annotation_col in df.columns else ''
        ctype = classify_lr_pair(
            row[ligand_col],
            row[receptor_col],
            ann
        )
        types.append(ctype)
    return types
