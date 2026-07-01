import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import yaml

logger = logging.getLogger(__name__)


class AnnDataProcessor:
    """
    Process 10x Visium spatial transcriptomics data.

    Main functionalities:
        - Load raw AnnData from h5ad
        - Quality control and filtering
        - Normalization and scaling
        - Feature extraction (PCA, HVG)
        - Spatial coordinate processing
        - Gene filtering by expression

    Attributes:
        adata (AnnData): Main AnnData object
        config (Dict): Configuration parameters
        spatial_key (str): Key for spatial coordinates in obsm
        processed (bool): Whether data has been processed
    """

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize processor with configuration.

        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary with configuration (alternative to file)
        """
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            # Use default configuration
            self.config = self._get_default_config()

        self.adata = None
        self.spatial_key = self.config.get('spatial', {}).get('spatial_key', 'spatial')
        self.processed = False

        logger.info("AnnDataProcessor initialized")

    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.debug(f"Loaded config from: {config_path}")
        return config

    @staticmethod
    def _get_default_config() -> Dict:
        """Return default configuration."""
        return {
            'data': {
                'raw_data_dir': 'data/raw/spatial',
                'h5ad_filename': 'adata.h5ad',
                'output_dir': 'data/processed/anndata'
            },
            'spatial': {
                'spatial_key': 'spatial',
                'scale_to_micrometers': True,
                'scale_factor_key': 'tissue_hires_scalef'  # From spatial/scalefactors_json.json
            },
            'preprocessing': {
                'min_genes': 200,
                'min_counts': 500,
                'min_cells': 3,
                'max_mito_pct': 20,
                'filter_mito_genes': True,
                'filter_ribo_genes': False,
                'n_top_genes': 2000,
                'normalize': True,
                'log1p': True,
                'scale': False
            },
            'features': {
                'method': 'pca',  # 'pca' or 'scvi'
                'n_components': 50,
                'use_highly_variable': True
            },
            'gene_filtering': {
                'min_expression_fraction': 0.01,  # Expressed in at least 1% of cells
                'min_mean_expression': 0.1
            }
        }

    def load_data(
        self,
        data_dir: Optional[str] = None,
        h5ad_filename: Optional[str] = None,
        spatial_dir: Optional[str] = None
    ) -> ad.AnnData:
        """
        Load 10x Visium data from h5ad file.

        Args:
            data_dir: Directory containing h5ad and spatial folder
            h5ad_filename: Name of h5ad file
            spatial_dir: Directory with spatial images and scalefactors

        Returns:
            AnnData object with spatial information
        """
        # Get paths from config or arguments
        if data_dir is None:
            data_dir = self.config['data']['raw_data_dir']
        if h5ad_filename is None:
            h5ad_filename = self.config['data']['h5ad_filename']

        data_dir = Path(data_dir)
        h5ad_path = data_dir / h5ad_filename

        # Check if file exists
        if not h5ad_path.exists():
            raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")

        logger.info(f"Loading AnnData from: {h5ad_path}")

        # Load AnnData
        adata = sc.read_h5ad(h5ad_path)

        # Validate spatial data
        if self.spatial_key not in adata.obsm:
            raise ValueError(f"Spatial coordinates not found. Expected key: {self.spatial_key}")

        # Load spatial images and scale factors if spatial_dir provided
        if spatial_dir is not None:
            spatial_dir = Path(spatial_dir)
        else:
            spatial_dir = data_dir / 'spatial'

        if spatial_dir.exists():
            logger.debug(f"Loading spatial metadata from: {spatial_dir}")
            self._load_spatial_metadata(adata, spatial_dir)
        else:
            logger.warning(f"Spatial directory not found: {spatial_dir}")

        logger.info(f"Loaded AnnData: {adata.n_obs} spots × {adata.n_vars} genes")

        self.adata = adata
        return adata

    def _load_spatial_metadata(self, adata: ad.AnnData, spatial_dir: Path):
        """
        Load spatial images and scale factors.

        Args:
            adata: AnnData object
            spatial_dir: Directory with spatial metadata
        """
        # Load scale factors
        scalefactors_path = spatial_dir / 'scalefactors_json.json'
        if scalefactors_path.exists():
            with open(scalefactors_path, 'r') as f:
                scalefactors = json.load(f)

            # Store in adata.uns
            if 'spatial' not in adata.uns:
                adata.uns['spatial'] = {}

            # Create default library_id if not exists
            library_id = list(adata.uns['spatial'].keys())[0] if adata.uns['spatial'] else 'visium'

            if library_id not in adata.uns['spatial']:
                adata.uns['spatial'][library_id] = {}

            adata.uns['spatial'][library_id]['scalefactors'] = scalefactors
            logger.debug(f"Loaded scale factors: {scalefactors}")
        else:
            logger.warning(f"Scale factors not found: {scalefactors_path}")

    def quality_control(self, inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Perform quality control filtering.

        Filters:
            - Spots with too few genes
            - Spots with too few counts
            - Spots with high mitochondrial percentage
            - Genes expressed in too few spots

        Args:
            inplace: Modify self.adata or return filtered copy

        Returns:
            Filtered AnnData if not inplace
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Performing quality control...")

        adata = self.adata if inplace else self.adata.copy()

        # Calculate QC metrics if not already present
        if 'n_genes_by_counts' not in adata.obs:
            sc.pp.calculate_qc_metrics(
                adata,
                percent_top=None,
                log1p=False,
                inplace=True
            )

        # Calculate mitochondrial percentage
        if 'pct_counts_mt' not in adata.obs:
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(
                adata,
                qc_vars=['mt'],
                percent_top=None,
                log1p=False,
                inplace=True
            )

        # Get filter thresholds from config
        min_genes = self.config['preprocessing']['min_genes']
        min_counts = self.config['preprocessing']['min_counts']
        max_mito_pct = self.config['preprocessing']['max_mito_pct']
        min_cells = self.config['preprocessing']['min_cells']

        # Store initial counts
        n_obs_before = adata.n_obs
        n_vars_before = adata.n_vars

        # Filter spots
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_cells(adata, min_counts=min_counts)

        # Filter by mitochondrial percentage
        if 'pct_counts_mt' in adata.obs:
            adata = adata[adata.obs['pct_counts_mt'] < max_mito_pct, :]

        # Filter genes
        sc.pp.filter_genes(adata, min_cells=min_cells)

        # Remove mitochondrial and ribosomal genes if configured
        if self.config['preprocessing']['filter_mito_genes']:
            mito_genes = adata.var_names.str.startswith('MT-')
            adata = adata[:, ~mito_genes]
            logger.debug(f"Removed {mito_genes.sum()} mitochondrial genes")

        if self.config['preprocessing']['filter_ribo_genes']:
            ribo_genes = adata.var_names.str.startswith(('RPS', 'RPL'))
            adata = adata[:, ~ribo_genes]
            logger.debug(f"Removed {ribo_genes.sum()} ribosomal genes")

        n_obs_after = adata.n_obs
        n_vars_after = adata.n_vars

        logger.info(f"QC filtering: {n_obs_before} → {n_obs_after} spots ({n_obs_before - n_obs_after} removed)")
        logger.info(f"QC filtering: {n_vars_before} → {n_vars_after} genes ({n_vars_before - n_vars_after} removed)")

        if inplace:
            self.adata = adata
            return None
        else:
            return adata

    def normalize_and_scale(self, inplace: bool = True) -> Optional[ad.AnnData]:
        """
        Normalize and optionally scale expression data.

        Steps:
            1. Store raw counts in layers['raw_counts']
            2. Normalize to median total counts
            3. Log1p transformation
            4. Identify highly variable genes
            5. Optional: Scale to unit variance

        Args:
            inplace: Modify self.adata or return processed copy

        Returns:
            Processed AnnData if not inplace
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info("Normalizing and scaling data...")

        adata = self.adata if inplace else self.adata.copy()

        # Store raw counts if not already stored
        if 'raw_counts' not in adata.layers:
            adata.layers['raw_counts'] = adata.X.copy()

        # Normalize
        if self.config['preprocessing']['normalize']:
            sc.pp.normalize_total(adata, target_sum=1e4)
            logger.debug("Normalized to 10,000 counts per spot")

        # Log1p transform
        if self.config['preprocessing']['log1p']:
            sc.pp.log1p(adata)
            logger.debug("Applied log1p transformation")

        # Identify highly variable genes
        n_top_genes = self.config['preprocessing']['n_top_genes']
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            subset=False,  # Don't subset yet
            flavor='seurat_v3',
            layer='raw_counts'
        )
        logger.info(f"Identified {n_top_genes} highly variable genes")

        # Scale (optional - often not needed for downstream graph models)
        if self.config['preprocessing']['scale']:
            sc.pp.scale(adata, max_value=10)
            logger.debug("Scaled data to unit variance")

        if inplace:
            self.adata = adata
            return None
        else:
            return adata

    def extract_cell_features(
        self,
        method: Optional[str] = None,
        n_components: Optional[int] = None,
        inplace: bool = True
    ) -> np.ndarray:
        """
        Extract low-dimensional cell features.

        Methods:
            - 'pca': Principal Component Analysis
            - 'scvi': scVI variational autoencoder (requires scvi-tools)

        Args:
            method: Feature extraction method
            n_components: Number of components/latent dimensions
            inplace: Store in self.adata or return features

        Returns:
            Cell features [n_cells, n_components]
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if method is None:
            method = self.config['features']['method']
        if n_components is None:
            n_components = self.config['features']['n_components']

        logger.debug(f"Extracting cell features using: {method}")

        adata = self.adata

        # Use highly variable genes if configured
        if self.config['features']['use_highly_variable']:
            if 'highly_variable' not in adata.var:
                logger.warning("Highly variable genes not computed. Using all genes.")
                adata_subset = adata
            else:
                adata_subset = adata[:, adata.var['highly_variable']]
                logger.debug(f"Using {adata_subset.n_vars} highly variable genes")
        else:
            adata_subset = adata

        if method == 'pca':
            # PCA
            sc.tl.pca(adata_subset, n_comps=n_components)
            features = adata_subset.obsm['X_pca']
            logger.debug(f"PCA: Extracted {features.shape[1]} components")

            if inplace:
                self.adata.obsm['X_pca'] = features

        elif method == 'scvi':
            # scVI (requires scvi-tools)
            try:
                import scvi

                # CRITICAL FIX: Create a copy to avoid view issues
                adata_subset = adata_subset.copy()
                logger.debug("Created AnnData copy for scVI (required by scvi-tools)")

                # Setup AnnData for scVI
                scvi.model.SCVI.setup_anndata(
                    adata_subset,
                    layer='raw_counts',
                    batch_key=self.config['features']['scvi'].get('batch_key', None)
                )

                # Train scVI model
                max_epochs = self.config['features']['scvi'].get('max_epochs', 100)
                use_cuda = self.config['features']['scvi'].get('use_cuda', True)

                vae = scvi.model.SCVI(adata_subset, n_latent=n_components)

                logger.info(f"Training scVI model (max_epochs={max_epochs})...")
                vae.train(
                    accelerator="gpu", devices="auto")

                # Extract latent representation
                features = vae.get_latent_representation()
                logger.debug(f"scVI: Extracted {features.shape[1]} latent dimensions")

                if inplace:
                    self.adata.obsm['X_scvi'] = features

            except ImportError:
                logger.error("scvi-tools not installed. Install with: pip install scvi-tools")
                logger.error("Falling back to PCA.")
                sc.tl.pca(adata_subset, n_comps=n_components)
                features = adata_subset.obsm['X_pca']

                if inplace:
                    self.adata.obsm['X_pca'] = features

            except Exception as e:
                logger.error(f"scVI training failed: {e}")
                logger.error("Falling back to PCA.")
                sc.tl.pca(adata_subset, n_comps=n_components)
                features = adata_subset.obsm['X_pca']

                if inplace:
                    self.adata.obsm['X_pca'] = features

        else:
            raise ValueError(f"Unknown method: {method}. Use 'pca' or 'scvi'.")

        return features

    def compute_clustering(
        self,
        method: str = 'leiden',
        resolution: float = 1.0,
        n_neighbors: int = 15,
        use_rep: Optional[str] = None,
        key_added: Optional[str] = None,
        inplace: bool = True
    ) -> Optional[np.ndarray]:
        """
        Perform clustering on processed data.

        Methods:
            - 'leiden': Leiden algorithm (recommended)
            - 'louvain': Louvain algorithm

        Args:
            method: Clustering method ('leiden' or 'louvain')
            resolution: Resolution parameter (higher = more clusters)
            n_neighbors: Number of neighbors for KNN graph
            use_rep: Representation to use ('X_pca', 'X_scvi', None for .X)
            key_added: Key to store clusters in adata.obs (default: method name)
            inplace: Store in self.adata or return cluster labels

        Returns:
            Cluster labels if not inplace, None otherwise
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.info(f"Computing {method} clustering (resolution={resolution})...")

        adata = self.adata

        # Determine which representation to use
        if use_rep is None:
            if 'X_scvi' in adata.obsm:
                use_rep = 'X_scvi'
                logger.debug("Using scVI latent representation for clustering")
            elif 'X_pca' in adata.obsm:
                use_rep = 'X_pca'
                logger.debug("Using PCA representation for clustering")
            else:
                logger.debug("Using normalized expression matrix for clustering")
        else:
            logger.debug(f"Using {use_rep} for clustering")

        # Compute neighborhood graph if not exists
        if 'neighbors' not in adata.uns:
            logger.info(f"Computing neighborhood graph (k={n_neighbors})...")
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                use_rep=use_rep,
                random_state=self.config['resources'].get('random_seed', 42)
            )
        else:
            logger.debug("Using existing neighborhood graph")

        # Perform clustering
        if key_added is None:
            key_added = method

        if method == 'leiden':
            sc.tl.leiden(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=self.config['resources'].get('random_seed', 42)
            )
        elif method == 'louvain':
            sc.tl.louvain(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=self.config['resources'].get('random_seed', 42)
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}. Use 'leiden' or 'louvain'.")

        n_clusters = adata.obs[key_added].nunique()
        logger.debug(f"{method.capitalize()} clustering: {n_clusters} clusters identified")

        # Log cluster sizes
        cluster_counts = adata.obs[key_added].value_counts().sort_index()
        logger.debug(f"Cluster sizes: {dict(cluster_counts)}")

        if not inplace:
            return adata.obs[key_added].values

        return None



    def extract_spatial_coordinates(
        self,
        scale_to_micrometers: Optional[bool] = None
    ) -> np.ndarray:
        """
        Extract and optionally scale spatial coordinates.

        Args:
            scale_to_micrometers: Convert pixel coordinates to micrometers

        Returns:
            Spatial coordinates [n_cells, 2]
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if scale_to_micrometers is None:
            scale_to_micrometers = self.config['spatial']['scale_to_micrometers']

        logger.debug("Extracting spatial coordinates...")

        # Get spatial coordinates
        spatial_coords = self.adata.obsm[self.spatial_key].copy()

        if scale_to_micrometers:
            # Get scale factor from uns
            scale_factor = self._get_scale_factor()

            if scale_factor is not None:
                # Visium spot diameter: 55 micrometers
                # Spot spacing: 100 micrometers center-to-center
                spot_diameter_um = 55.0

                # Scale coordinates
                spatial_coords = spatial_coords * scale_factor * spot_diameter_um
                logger.debug(f"Scaled coordinates to micrometers (scale factor: {scale_factor:.4f})")
            else:
                logger.warning("Scale factor not found. Using pixel coordinates.")

        # Store scaled coordinates
        self.adata.obsm['spatial_scaled'] = spatial_coords

        return spatial_coords

    def _get_scale_factor(self) -> Optional[float]:
        """
        Extract scale factor from AnnData.uns or from scalefactors_json.json file.

        Priority:
            1. Try to get from adata.uns['spatial'][library_id]['scalefactors']
            2. If not found, read from scalefactors_json.json in spatial directory

        Returns:
            Scale factor or None if not found
        """
        scale_factor_key = self.config['spatial']['scale_factor_key']

        # Method 1: Try to get from AnnData.uns
        if 'spatial' in self.adata.uns:
            spatial_data = self.adata.uns['spatial']

            if len(spatial_data) > 0:
                library_id = list(spatial_data.keys())[0]

                if 'scalefactors' in spatial_data[library_id]:
                    scalefactors = spatial_data[library_id]['scalefactors']

                    if scale_factor_key in scalefactors:
                        logger.debug(f"Scale factor '{scale_factor_key}' = {scalefactors[scale_factor_key]} (from adata.uns)")
                        return scalefactors[scale_factor_key]

        # Method 2: Try to read from scalefactors_json.json file
        logger.debug("Scale factor not found in adata.uns, reading from file...")

        data_dir = Path(self.config['data']['raw_data_dir'])
        spatial_dir = data_dir / 'spatial'
        scalefactors_path = spatial_dir / 'scalefactors_json.json'

        if scalefactors_path.exists():
            try:
                with open(scalefactors_path, 'r') as f:
                    scalefactors = json.load(f)

                if scale_factor_key in scalefactors:
                    logger.debug(f"Scale factor '{scale_factor_key}' = {scalefactors[scale_factor_key]} (from file)")
                    return scalefactors[scale_factor_key]
                else:
                    logger.warning(f"Key '{scale_factor_key}' not found. Available: {list(scalefactors.keys())}")
                    return None

            except Exception as e:
                logger.error(f"Failed to read scalefactors: {e}")
                return None
        else:
            logger.warning(f"File not found: {scalefactors_path}")
            return None

    def filter_genes_with_interactions(self) -> List[str]:
        """
        Create gene universe using dual-set strategy:
            - HVGs (3000) for cell state variation
            - All LR genes for interaction modeling
            - All metabolic genes for metabolic communication

        This ensures:
            - Cell-gene edges use informative HVGs
            - Gene-gene/gene-metabolite edges have full interaction space

        Returns:
            List of genes to include in graph (union of HVGs + interaction genes)
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.debug("Creating gene universe with dual-set strategy...")

        # 1. Get HVGs (should already be computed in normalize_and_scale)
        if 'highly_variable' not in self.adata.var:
            logger.warning("HVGs not computed. Running HVG selection...")
            n_top_genes = self.config['preprocessing']['n_top_genes']
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=n_top_genes,
                subset=False,
                flavor='seurat_v3',
                layer='raw_counts'
            )

        hvg_genes = self.adata.var_names[self.adata.var['highly_variable']].tolist()
        logger.debug(f"  HVGs: {len(hvg_genes)} genes")

        # 2. Load LR genes from database
        lr_db_path = self.get_lr_database_path()
        if lr_db_path.exists():
            lr_db = pd.read_csv(lr_db_path)
            lr_genes = set(lr_db['ligand'].tolist() + lr_db['receptor'].tolist())
            lr_genes_in_adata = [g for g in lr_genes if g in self.adata.var_names]
            logger.debug(f"  LR genes: {len(lr_genes_in_adata)} genes (from {len(lr_genes)} in database)")
        else:
            logger.warning(f"LR database not found: {lr_db_path}")
            lr_genes_in_adata = []

        # 3. Load metabolic genes from database
        met_db_path = self.get_metabolite_database_path()
        if met_db_path.exists():
            met_db = pd.read_csv(met_db_path)
            metabolic_genes = set(met_db['receptor_symbol'].tolist())
            metabolic_genes_in_adata = [g for g in metabolic_genes if g in self.adata.var_names]
            logger.debug(f"  Metabolic genes: {len(metabolic_genes_in_adata)} genes (from {len(metabolic_genes)} in database)")
        else:
            logger.warning(f"Metabolite database not found: {met_db_path}")
            metabolic_genes_in_adata = []

        # 4. Create union of all gene sets
        interaction_genes = set(lr_genes_in_adata) | set(metabolic_genes_in_adata)
        final_gene_set = set(hvg_genes) | interaction_genes

        # Calculate statistics
        hvg_only = set(hvg_genes) - interaction_genes
        interaction_only = interaction_genes - set(hvg_genes)
        overlap = set(hvg_genes) & interaction_genes

        logger.debug(f"\n" + "="*60)
        logger.debug("GENE UNIVERSE STATISTICS (Dual-Set Strategy)")
        logger.debug("="*60)
        logger.debug(f"  HVGs only:            {len(hvg_only):,} genes")
        logger.debug(f"  Interaction only:     {len(interaction_only):,} genes")
        logger.debug(f"  Overlap (both):       {len(overlap):,} genes")
        logger.debug(f"  ─────────────────────────────────────────")
        logger.debug(f"  TOTAL gene universe:  {len(final_gene_set):,} genes")
        logger.debug("="*60 + "\n")

        # 5. Apply expression filters (optional, to remove truly unexpressed genes)
        min_expr_fraction = self.config['gene_filtering'].get('min_expression_fraction', 0.0)

        if min_expr_fraction > 0:
            logger.debug(f"Applying minimal expression filter (>{min_expr_fraction*100:.1f}% cells)...")

            if 'raw_counts' in self.adata.layers:
                X = self.adata.layers['raw_counts']
            else:
                X = self.adata.X

            if hasattr(X, 'toarray'):
                X = X.toarray()

            # Only filter out genes expressed in <0.1% of cells (truly absent)
            gene_indices = [self.adata.var_names.get_loc(g) for g in final_gene_set if g in self.adata.var_names]
            X_subset = X[:, gene_indices]
            expression_fraction = (X_subset > 0).mean(axis=0)

            kept_mask = expression_fraction >= min_expr_fraction
            filtered_final_genes = [list(final_gene_set)[i] for i in range(len(final_gene_set)) if kept_mask[i]]

            n_removed = len(final_gene_set) - len(filtered_final_genes)
            logger.debug(f"  Removed {n_removed} unexpressed genes")
            logger.debug(f"  Final gene count: {len(filtered_final_genes):,} genes")

            return filtered_final_genes

        return list(final_gene_set)



    def filter_genes_by_expression(
        self,
        min_expression_fraction: Optional[float] = None,
        min_mean_expression: Optional[float] = None,
        force_keep_database_genes: Optional[bool] = None,
        use_dual_set_strategy: bool = True

    ) -> List[str]:
        """
        Filter genes by expression criteria for graph construction.

        Args:
            min_expression_fraction: Minimum fraction of cells expressing gene
            min_mean_expression: Minimum mean expression
            force_keep_database_genes: Keep genes from LR/metabolite databases even if low expression

        Returns:
            List of gene names passing filters
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if use_dual_set_strategy:
            logger.debug("Using dual-set strategy (HVG + interaction genes)...")
            return self.filter_genes_with_interactions()

        if min_expression_fraction is None:
            min_expression_fraction = self.config['gene_filtering']['min_expression_fraction']
        if min_mean_expression is None:
            min_mean_expression = self.config['gene_filtering']['min_mean_expression']
        if force_keep_database_genes is None:
            force_keep_database_genes = self.config['gene_filtering']['force_keep_database_genes']

        logger.debug("Filtering genes by expression...")

        # Use raw counts for filtering
        if 'raw_counts' in self.adata.layers:
            X = self.adata.layers['raw_counts']
        else:
            X = self.adata.X

        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Calculate expression metrics
        expression_fraction = (X > 0).mean(axis=0)
        mean_expression = X.mean(axis=0)

        # Apply filters
        expression_mask = (expression_fraction >= min_expression_fraction) & (mean_expression >= min_mean_expression)

        # Exclude genes by pattern
        exclude_patterns = self.config['gene_filtering'].get('exclude_gene_patterns', [])
        pattern_mask = np.ones(self.adata.n_vars, dtype=bool)

        for pattern in exclude_patterns:
            pattern_mask &= ~self.adata.var_names.str.match(pattern)

        # Combine masks
        final_mask = expression_mask & pattern_mask

        filtered_genes = self.adata.var_names[final_mask].tolist()

        # Force keep database genes if configured
        if force_keep_database_genes:
            database_genes = self._load_database_genes()

            # Add database genes that are in adata but not in filtered list
            genes_to_add = [g for g in database_genes if g in self.adata.var_names and g not in filtered_genes]
            filtered_genes.extend(genes_to_add)

            logger.debug(f"Force-added {len(genes_to_add)} database genes")

        logger.debug(f"Gene filtering: {self.adata.n_vars} → {len(filtered_genes)} genes")

        return filtered_genes

    def _load_database_genes(self) -> set:
        """
        Load gene universe from processed interaction databases.

        Returns:
            Set of gene symbols from LR and metabolite databases
        """
        gene_universe_path = self.get_gene_universe_path()

        if gene_universe_path.exists():
            with open(gene_universe_path, 'r') as f:
                genes = json.load(f)
            logger.debug(f"Loaded {len(genes)} genes from database universe")
            return set(genes)
        else:
            logger.warning(f"Gene universe file not found: {gene_universe_path}")
            return set()


    def get_cell_metadata(self) -> pd.DataFrame:
        """
        Extract cell metadata for NodeMapper.

        Returns:
            DataFrame with cell ID, spatial coords, cell type, etc.
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        logger.debug("Extracting cell metadata...")

        metadata = pd.DataFrame(index=self.adata.obs_names)

        # Spatial coordinates
        if 'spatial_scaled' in self.adata.obsm:
            coords = self.adata.obsm['spatial_scaled']
        else:
            coords = self.adata.obsm[self.spatial_key]

        metadata['spatial_x'] = coords[:, 0]
        metadata['spatial_y'] = coords[:, 1]

        # Cell type if available
        if 'cell_type' in self.adata.obs:
            metadata['cell_type'] = self.adata.obs['cell_type']
        elif 'leiden' in self.adata.obs:
            metadata['cell_type'] = self.adata.obs['leiden']
        else:
            metadata['cell_type'] = 'unknown'

        # Quality metrics
        if 'n_genes_by_counts' in self.adata.obs:
            metadata['n_genes'] = self.adata.obs['n_genes_by_counts']
        if 'total_counts' in self.adata.obs:
            metadata['total_counts'] = self.adata.obs['total_counts']

        return metadata

    def process_all(self, perform_clustering: bool = True,
    compute_umap_embedding: bool = False) -> ad.AnnData:
        """
        Run complete processing pipeline.

        Steps:
            1. Quality control
            2. Normalization and scaling
            3. Feature extraction
            4. Spatial coordinate scaling

        Returns:
            Fully processed AnnData
        """
        logger.debug("Starting full processing pipeline...")

        # Quality control
        self.quality_control(inplace=True)

        # Normalize and scale
        self.normalize_and_scale(inplace=True)

        # Extract features
        self.extract_cell_features(inplace=True)

        # Clustering
        if perform_clustering:
            clustering_config = self.config.get('clustering', {})
            resolution = clustering_config.get('resolution', 1.0)
            n_neighbors = clustering_config.get('n_neighbors', 15)

            self.compute_clustering(
                method='leiden',
                resolution=resolution,
                n_neighbors=n_neighbors,
                inplace=True
            )
        # if compute_umap_embedding:
        #     self.compute_umap(inplace=True)

            # Extract spatial coordinates
            self.extract_spatial_coordinates()

            self.processed = True
            logger.info("Processing complete!")

        return self.adata
    def _get_interaction_database_path(self, filename_key: str) -> Path:
        """
        Get full path to interaction database file.

        Args:
            filename_key: Key in config['interactions'] for filename

        Returns:
            Full path to file
        """
        processed_dir = Path(self.config['interactions']['processed_dir'])
        filename = self.config['interactions'][filename_key]
        return processed_dir / filename

    def get_lr_database_path(self) -> Path:
        """Get path to processed LR database."""
        return self._get_interaction_database_path('lr_database_file')

    def get_metabolite_database_path(self) -> Path:
        """Get path to processed metabolite database."""
        return self._get_interaction_database_path('metabolite_database_file')

    def get_gene_universe_path(self) -> Path:
        """Get path to gene universe JSON."""
        return self._get_interaction_database_path('gene_universe_file')

    def get_metabolite_universe_path(self) -> Path:
        """Get path to metabolite universe JSON."""
        return self._get_interaction_database_path('metabolite_universe_file')

    def save_processed_data(
        self,
        output_dir: Optional[str] = None,
        save_filtered_genes: bool = True,
        subset_adata_to_filtered_genes: bool = True
    ):
        """
        Save processed AnnData and related files.

        Saves:
            - processed_adata.h5ad
            - cell_features.npy
            - spatial_coordinates.npy
            - filtered_genes.json
            - cell_metadata.csv

        Args:
            output_dir: Output directory
            save_filtered_genes: Save filtered gene list
        """
        if self.adata is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if output_dir is None:
            output_dir = self.config['data']['output_dir']

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Saving processed data to: {output_dir}")

        # Save AnnData
        adata_path = output_dir / 'processed_adata.h5ad'
        self.adata.write_h5ad(adata_path)
        logger.info(f"Saved: {adata_path}")

        # Save cell features
        if 'X_pca' in self.adata.obsm:
            features_path = output_dir / 'cell_features_pca.npy'
            np.save(features_path, self.adata.obsm['X_pca'])
            logger.info(f"Saved: {features_path}")

        if 'X_scvi' in self.adata.obsm:
            features_path = output_dir / 'cell_features_scvi.npy'
            np.save(features_path, self.adata.obsm['X_scvi'])
            logger.info(f"Saved: {features_path}")

        # Save spatial coordinates
        if 'spatial_scaled' in self.adata.obsm:
            coords_path = output_dir / 'spatial_coordinates.npy'
            np.save(coords_path, self.adata.obsm['spatial_scaled'])
            logger.info(f"Saved: {coords_path}")

        # Save filtered genes
        if save_filtered_genes:
            filtered_genes = self.filter_genes_by_expression()

            # CRITICAL: Subset AnnData to filtered genes
            if subset_adata_to_filtered_genes:
                n_genes_before = self.adata.n_vars

                # Keep only filtered genes
                self.adata = self.adata[:, filtered_genes].copy()

                n_genes_after = self.adata.n_vars
                logger.debug(f"Subsetted AnnData: {n_genes_before:,} → {n_genes_after:,} genes")

            genes_path = output_dir / 'filtered_genes.json'
            with open(genes_path, 'w') as f:
                json.dump(filtered_genes, f, indent=2)
            logger.info(f"Saved: {genes_path} ({len(filtered_genes)} genes)")

        # Save cell metadata
        metadata = self.get_cell_metadata()
        metadata_path = output_dir / 'cell_metadata.csv'
        metadata.to_csv(metadata_path)
        logger.info(f"Saved: {metadata_path}")

        logger.info("All files saved successfully")


# ============================================================================
# Standalone execution
# ============================================================================

def main():
    """Example usage of AnnDataProcessor."""
    import argparse

    parser = argparse.ArgumentParser(description='Process 10x Visium AnnData')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, help='Override data directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')

    args = parser.parse_args()

    # Initialize processor
    processor = AnnDataProcessor(config_path=args.config)

    # Load data
    processor.load_data(data_dir=args.data_dir)

    # Process all
    processor.process_all()

    # Save results
    processor.save_processed_data(output_dir=args.output_dir)

    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()
