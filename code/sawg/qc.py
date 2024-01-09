from __future__ import annotations
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def calc_n_transcripts(cell_by_gene: ad.AnnData|pd.DataFrame):
    """Calculate number of detected transcripts per cell."""
    counts = get_counts_from_cbg(cell_by_gene)

    if isinstance(counts, np.ndarray):
        n_transcripts = counts.sum(axis=1)

    elif isinstance(counts, sparse.spmatrix):
        n_transcripts = counts.A.sum(axis=1)

    return n_transcripts


def calc_n_genes(cell_by_gene: ad.AnnData|pd.DataFrame):
    """Calculate number of genes per cell."""
    counts = get_counts_from_cbg(cell_by_gene)

    if isinstance(counts, np.ndarray):
        n_genes = np.count_nonzero(counts, axis=1)

    elif isinstance(counts, sparse.spmatrix):
        n_genes = counts.getnnz(axis=1)

    return n_genes


def calc_n_blanks(cell_by_gene: ad.AnnData|pd.DataFrame):
    """Calculate number of blanks per cell."""
    counts = get_counts_from_cbg(cell_by_gene)

    if isinstance(cell_by_gene, pd.DataFrame):
        blank_cols = cell_by_gene.columns.str.startswith('Blank')

    elif isinstance(cell_by_gene, ad.AnnData):
        blank_cols = cell_by_gene.var_names.str.startswith('Blank')

    if isinstance(counts, np.ndarray):
        blank_counts = counts[:, blank_cols].sum(axis=1)

    elif isinstance(counts, sparse.spmatrix):
        blank_counts = counts.A[:, blank_cols].sum(axis=1)

    return blank_counts


def get_counts_from_cbg(cell_by_gene: ad.AnnData|pd.DataFrame):
    """Get counts array from a cell by gene table in either an AnnData object
    or pandas DataFrame.
    """
    if isinstance(cell_by_gene, pd.DataFrame):
        counts = cell_by_gene.values

    elif isinstance(cell_by_gene, ad.AnnData):
        counts = cell_by_gene.X

    assert np.issubdtype(counts.dtype, np.integer)

    return counts


def run_solo_doublet_detection(cell_by_gene: ad.AnnData, threshold: float|None = None):
    """Detect doublets using SOLO. Returns results in a pandas DataFrame."""
    import scvi
    scvi.model.SCVI.setup_anndata(cell_by_gene)
    vae = scvi.model.SCVI(cell_by_gene)

    # this may need a try except for that intermittent batch size error
    vae.train()

    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()

    results = solo.predict()
    results['solo_prediction'] = solo.predict(soft=False)
    results['doublet_score_diff'] = results.doublet - results.singlet

    if threshold is not None:
        results['doublet_score'] = 'singlet'
        results.loc[results['doublet_score_diff'] > threshold, 'doublet_score'] = 'doublet'
    else:
        results['doublet_score'] = results['solo_prediction']

    return results


def run_doublet_detection(cell_by_gene: ad.AnnData, output_dir: str, method: str, method_kwargs: dict, filter_col: str|None=None):
    """Wrapper function to run a doublet detection algorithm on the HPC.
    Saves results into a csv file and doublet detection metadata into a json.
    """
    if filter_col is not None:
        cbg_filt = cell_by_gene[cell_by_gene.obs[filter_col]].copy()
    else:
        cbg_filt = cell_by_gene.copy()

    if method == 'solo':
        results = run_solo_doublet_detection(cbg_filt, **method_kwargs)

    else:
        raise NotImplementedError('Only SOLO is currently implemented.')

    result_file = f'{output_dir}/doublet_results.csv'
    results.to_csv(result_file)

    meta_file = f'{output_dir}/doublet_detection_params.json'
    doublet_params = {
            'method': method,
            'method_kwargs': method_kwargs,
            'filter_col': filter_col
    }
    with open(meta_file, 'w') as f:
        json.dump(doublet_params, f)
