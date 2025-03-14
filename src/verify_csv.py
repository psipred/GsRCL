import sys
import argparse
from pathlib import Path

import csv
import numpy as np
import pandas as pd


def verify(args):
    try:
        input = pd.read_csv(args.input, header=0, index_col=0)
    except:
        print(
            f'Error 128: The input file could not be read as csv by pandas'
        )
        sys.exit(128)

    if input.shape[0] > 1000:
        print(
            'Error 128: The genes expression matrix should include no more than 1000 rows (i.e. cells),' \
            f' while the given matrix inlcudes {count} rows.'
        )
        sys.exit(128) 

    query_genes = input.columns.to_numpy().flatten()
    query_barcodes = input.index.to_numpy().flatten()
    genes_experssion_mat = input.to_numpy()

    if genes_experssion_mat.shape[0] > genes_experssion_mat.shape[1]:
        print(
            'Error 128: The genes expression matrix topology should be cells (rows) by genes (columns), where' \                    
            ' cells denote rows and genes denote columns. There must be more columns than genes to analyse the data'
        )
        sys.exit(128)

    if not query_genes.dtype == 'object':
        print(
            'Error 128: The input csv file header should contain gene ids as strings,' \
            f' while the given file header includes {query_genes[:5]}.'
        )
        sys.exit(128)

    if np.unique(query_genes).shape[0] != query_genes.shape[0]:
        print(
            'Error 128: The set of query genes has duplicates.' 
        )
        sys.exit(128)

    if not query_barcodes.dtype == 'object':
        print(
            'Error 128: The input csv file rows should start with cell type barcodes as strings,' \
            f' while the given file rows start with {query_barcodes[:5]}.'
        )
        sys.exit(128)

    ref_genes = pd.read_csv(Path(args.model_path, args.reference, f'{args.reference}-reference-genes.csv'), header=None, index_col=False)
    ref_genes = ref_genes.to_numpy().flatten()
    matched_genes_mask = np.isin(query_genes, ref_genes, assume_unique=True)
    if matched_genes_mask.sum() / query_genes.shape[0] <= 0.5:
        print(
            f'Error 128: {round((matched_genes_mask.sum() / query_genes.shape[0]) * 100)}% of query genes are found in reference genes,' \
            f' this will affect the output reliability. Please try a different reference.'
        )
        sys.exit(128)

    np.save(Path(args.output, 'mat.npy'), genes_experssion_mat)
    np.save(Path(args.output, 'query_barcodes.npy'), query_barcodes)
    np.save(Path(args.output, 'query_genes.npy'), query_genes)
    np.save(Path(args.output, 'matched_genes_mask.npy'), matched_genes_mask)
    sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help='the full path to the input file')
    parser.add_argument('-o', '--output', default='.', help='the full path to the output directory', required=True)
    parser.add_argument('-r', '--reference', help='the name of the reference dataset', required=True)
    parser.add_argument('-mp', '--model_path', help='the path to the reference models and the set of reference genes', required=True)

    args = parser.parse_args()
    verify(args)
