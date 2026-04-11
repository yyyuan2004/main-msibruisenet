"""Precompute PCA projection matrix for MSI data.

Usage:
    python scripts/precompute_pca.py \
        --data_dir /root/autodl-tmp/datasets/185_9bands \
        --output pca_matrix.npz

The output .npz file contains:
    - components: (n_components, 9) PCA projection matrix
    - mean: (9,) per-channel mean for centering
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pca_transform import compute_pca_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Precompute PCA projection matrix for MSI data"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory (contains images/ and whole/)")
    parser.add_argument("--image_dir", type=str, default="images",
                        help="Subdirectory for spectral images")
    parser.add_argument("--whole_dir", type=str, default="whole",
                        help="Subdirectory for whole-apple masks")
    parser.add_argument("--n_components", type=int, default=3,
                        help="Number of PCA components (default 3)")
    parser.add_argument("--max_pixels", type=int, default=500000,
                        help="Max pixels to sample for PCA fitting")
    parser.add_argument("--output", type=str, default="pca_matrix.npz",
                        help="Output path for PCA matrix file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    components, mean = compute_pca_matrix(
        data_dir=args.data_dir,
        image_dir=args.image_dir,
        whole_dir=args.whole_dir,
        n_components=args.n_components,
        max_pixels=args.max_pixels,
        seed=args.seed,
    )

    np.savez(args.output, components=components, mean=mean)
    print(f"\nPCA matrix saved to: {args.output}")
    print(f"  components shape: {components.shape}")
    print(f"  mean shape: {mean.shape}")


if __name__ == "__main__":
    import numpy as np
    main()
