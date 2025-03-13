import sys
import os
import joblib
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from gsrcl_networks import Encoder


def load_files(args):
    mat = np.load(Path(args.output, 'mat.npy'), allow_pickle=True)
    query_barcodes = np.load(Path(args.output, 'query_barcodes.npy'), allow_pickle=True)
    query_genes = np.load(Path(args.output, 'query_genes.npy'), allow_pickle=True)
    matched_genes_mask = np.load(Path(args.output, 'matched_genes_mask.npy'), allow_pickle=True)

    return mat, query_barcodes, query_genes, matched_genes_mask


def cross_reference_genes(args, query_genes, mat, matched_genes_mask):
    ref_genes = pd.read_csv(Path(args.model_path, args.reference, f'{args.reference}-reference-genes.csv'), header=None, index_col=False)
    ref_genes = ref_genes.to_numpy().flatten()
    order = {g: i for i, g in enumerate(ref_genes)}

    query_genes = query_genes[matched_genes_mask]
    mat = mat[:, matched_genes_mask]

    missing = np.setdiff1d(ref_genes, query_genes)
    query_genes = np.hstack((missing, query_genes))
    mat = np.hstack((np.zeros(shape=(mat.shape[0], missing.shape[0])), mat))

    order2 = np.array([order.get(g) for g in query_genes])
    order2 = np.array([i for i, _ in sorted(zip(range(query_genes.shape[0]), order2), key=lambda g:g[1])])
    mat = mat[:, order2]

    return mat


def preprocess(args, mat, epsilon=1e-6):
    mat = torch.tensor(mat, dtype=torch.float32)

    if args.log == 1:
        mat = torch.log(mat + 1.0 + epsilon)

    return mat


def load_encoder(args, checkpoint):
    device = torch.device('cpu')
    params = torch.load(Path(args.model_path, args.reference, checkpoint), map_location=device)
    encoder = Encoder(**params['frozen_kwargs'])
    encoder.load_state_dict(params['forzen_params'])
    encoder.eval()

    return encoder


def get_probs(args, cell_type, checkpoint, mat):
    if checkpoint.endswith('pt'):
        clf = joblib.load(Path(args.model_path, args.reference, f'{cell_type}--svm.joblib'))
        encoder = load_encoder(args, checkpoint)

        with torch.no_grad():
            h = encoder(mat)

        probs = clf.predict_proba(h.detach().numpy())[:, 1]

    else:
        clf = joblib.load(Path(args.model_path, args.reference, checkpoint))
        probs = clf.predict_proba(mat)[:, 1]

    return probs


def format_results(args, results, query_barcodes):
    df = pd.DataFrame(results)
    sum = df.sum(axis=1)
    df = df.div(sum, axis=0)
    preds = df.to_numpy().argmax(axis=1)
    mask = df.to_numpy().max(axis=1) <= args.p_value

    df['preds'] = [df.columns[p] for p in preds]
    df['Identified as'] = df['preds'].where(~mask, other='Unassigned')
    df.index = query_barcodes

    return df


def get_palette(n=20):
    idxs = list(range(n))
    idxs2 = [idxs[i::4] for i in range(4)]
    idxs2 = sum(idxs2, [])
    palette = np.array([plt.cm.tab20c(float(i) / n) for i in range(n)])[idxs2]

    return palette


def get_contour(ax, emb, results, palette):
    cell_types = results.columns.to_numpy()[:-2]
    preds = results['preds'].to_numpy()

    _, ax = plt.subplots()
    for cell_type, colour in zip(cell_types, palette):
        y = np.where(preds == cell_type, 1, 0)
        if np.unique(y).shape[0] == 1:
            print(cell_type, np.unique(y))
            continue

        svm = SVC(probability=True)
        svm.fit(emb, y)
        DecisionBoundaryDisplay.from_estimator(
            svm, emb, response_method='predict_proba', plot_method='contour', ax=ax, zorder=1, grid_resolution=500,
            colors=[colour]
        )
    return ax


def plot(args, mat, results):
    _, ax = plt.subplots()
    palette = get_palette()
    emb = TSNE().fit_transform(mat)
    ax = get_contour(ax, emb, results, palette)

    cell_types = results.columns.to_numpy()[:-2]
    preds = results['Identified as'].to_numpy()
    if 'Unassigned' in preds:
        cell_types = np.append(cell_types, 'Unassigned')

    y = np.hstack([np.where(cell_types == c)[0] for c in preds])
    for i, y_, e in zip(results.index, y, emb):
        if y_ == cell_types.shape[0] - 1:
            ax.scatter(e[0], e[1], facecolor=palette[y_], edgecolor='black', zorder=3, marker='v', s=70, gid=i)

        else:
            ax.scatter(e[0], e[1], facecolor=palette[y_], edgecolor='black', zorder=3, s=30, gid=i)

    ax.set_xlabel('t-SNE 1', fontsize=16)
    ax.set_ylabel('t-SNE 2', fontsize=16)
    handles = [Rectangle(xy=(0, 0), width=1, height=1, edgecolor='black', facecolor=c) for c in palette]
    ax.legend(handles, cell_types, fontsize=12, handlelength=4, handleheight=3, bbox_to_anchor=(1.0, 1.02))
    plt.savefig(Path(args.output, 'tsne.svg'), dpi=600, bbox_inches='tight')
    plt.savefig(Path(args.output, 'tsne.png'), dpi=600, bbox_inches='tight')
    plt.close()


def main(args):
    try:
        results = {}
        mat, query_barcodes, query_genes, matched_genes_mask = load_files(args)
        mat = cross_reference_genes(args, query_genes, mat, matched_genes_mask)
        mat = preprocess(args, mat)

        for checkpoint in os.listdir(Path(args.model_path, args.reference)):
            cell_type = checkpoint.split('--')[0]
            if cell_type in results or checkpoint.endswith('csv'):
                continue

            probs = get_probs(args, cell_type, checkpoint, mat)
            results[cell_type] = probs

        results = format_results(args, results, query_barcodes)
        results.to_csv(Path(args.output, 'probabilities.csv'))
        plot(args, mat.detach().numpy(), results)

    except:
       print(
           f'Error 135: Server error'
       )
       sys.exit(135)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output', help='the full path to the output directory')
    parser.add_argument('-r', '--reference', help='the name of the reference dataset')
    parser.add_argument('-mp', '--model_path', help='the full path to the reference models and the set of reference genes')
    parser.add_argument('-p', '--prob_cutoff', type=float, default=0.5, help='select a probability cut-off for putative new cell types')
    parser.add_argument('-l', '--log',  type=int, default=1, help='log transform the input matrix if it contains raw counts')

    args = parser.parse_args()
    main(args)
