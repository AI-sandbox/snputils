import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from ..processing import PCA
from snputils.snp.io.read import VCFReader

log = logging.getLogger(__name__)


def parse_pca_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='pca', description='Principal Component Analysis plot and save components.')

    parser.add_argument('--vcf_file', required=True, type=str, help='Load .vcf from this path.')
    parser.add_argument('--fig_path', required=True, type=str, help='Path used to save PCA plot.')
    parser.add_argument('--npy_path', required=True, type=str, help='Path used to save principal components in .npy format.') 
    parser.add_argument('--backend', required=True, type=str, default='sklearn', help='Backend used to perform PCA.') 
    
    
    return parser.parse_args(argv)


def plot_and_save_pca(argv: List[str]):
    args = parse_pca_args(argv)
    reader = VCFReader(args.vcf_file)
    snpobj = reader.read()

    pca = PCA(backend=args.backend, n_components=2)
    components = pca.fit_transform(snpobj)
    if args.backend == "pytorch":
        components = components.cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.scatter(components[:,0], components[:,1], linewidth=0, alpha=0.5)
    plt.xlabel("Principal Component 1", fontsize=20)
    plt.ylabel("Principal Component 2", fontsize=20)
    plt.tight_layout()

    # Save plot
    plt.savefig(args.fig_path)

    # Save components in .npy format    
    np.save(args.npy_path, components)
    return 0
