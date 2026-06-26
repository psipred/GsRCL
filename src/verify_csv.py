import sys
import argparse
from pathlib import Path

import csv
import numpy as np
import pandas as pd
import io
import re

def verify(args):

    # here we handle the case were zeroes should be whatever format the user
    # tells us so the pandas auto casting isn't confused
    input_data = io.StringIO('')
    with open(args.input, "r", encoding="utf-8") as fh:
        input_data.write(next(fh))
        datareader = csv.reader(fh, delimiter=",")
        zero_pattern = re.compile("^0\.0+$")
        for entries in datareader:
            id = entries[0]
            for i, value in enumerate(entries):
                if i == 0:
                    continue
                # if the user has told us to log the data then find any float zeroes and make them int strings
                if args.log == 1:
                    if zero_pattern.match(value):
                        entries[i] = "0"
                    else:
                        #check the other values for decimals. If we find one then raise an error
                        if "." in value:
                            print(f'Error 128: Float found in integer only file {value}')
                            sys.exit(128)                    
                else:
                    # if the user has told us the file is logs values make sure all the zeroes are
                    if value == "0":
                        entries[i] = "0.0"
                    else:
                        #check the other values for decimals. If we find one then raise an error
                        if "." not in value:
                            print(f'Error 128: integer found in float only file {value}')
                            sys.exit(128)   
            data_line = ','.join(entries)
            input_data.write(data_line+"\n")
    input_data.seek(0)
    
    try:
        input = pd.read_csv(input_data, header=0, index_col=0)
    except:
        print(
            f'Error 128: The input file could not be read as csv by pandas'
        )
        sys.exit(128)
    
    print(input)

    typeset = set()
    for type in input.dtypes:
        typeset.add(f"{type}")
    print(typeset)
    if len(typeset) != 1:
        print('Error 128: Input data has both integer and float values. Please ensure data are either integer count data or log transformed data')
        sys.exit(128)

    #LOG TEST
    if args.log == 1:
        if "int64" not in list(typeset)[0]:
            print('Error 128: Data already appear to be log transformed. Please unselect this option and submit again')
            sys.exit(128) 
    else: 
        if "float64" not in list(typeset)[0]:
            print('Error 128: Data appears to contain only integers, this is likely raw count data. Please select the log transform option and submit again')
            sys.exit(128) 

    if input.shape[0] > 1000:
        print(
            'Error 128: The gene expression matrix should include no more than 1000 rows (i.e. cells),' \
            f' while the given matrix inlcudes {input.shape[0]} rows.'
        )
        sys.exit(128) 

    query_genes = input.columns.to_numpy().flatten()
    query_barcodes = input.index.to_numpy().flatten()
    genes_experssion_mat = input.to_numpy()

    if genes_experssion_mat.shape[0] > genes_experssion_mat.shape[1]:
        print(
            'Error 128: The gene expression matrix topology should be cell IDs (rows) by genes (columns). ' \
            'There must be more columns than rows to analyse the data' 
        )
        sys.exit(128)

    if not query_genes.dtype == 'object':
        print(
            f'Error 128: The input csv file header should contain gene ids as strings,' \
            ' while the given file header includes {query_genes[:5]}.'
        )
        sys.exit(128)

    if np.unique(query_genes).shape[0] != query_genes.shape[0]:
        print(
            'Error 128: The set of query genes has duplicates.' 
        )
        sys.exit(128)

    if not query_barcodes.dtype == 'object':
        print(
            f'Error 128: The input csv file rows should start with cell type barcodes as strings,' \
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
    parser.add_argument('-l', '--log',  type=int, default=1, help='log transform the input matrix if it contains raw counts', required=True)

    args = parser.parse_args()
    verify(args)
