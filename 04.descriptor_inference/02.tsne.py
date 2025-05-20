import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import openTSNE
from glob import glob
import matplotlib.colors as mcolors
from tqdm import tqdm

def get_intermediate_color(color1: str, color2: str, ratio: float) -> str:
    """
    Interpolate between two hex colors.
    """
    if not 0 <= ratio <= 1:
        raise ValueError("Ratio must be between 0 and 1")
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', [color1, color2])
    rgba = cmap(ratio)
    return mcolors.to_hex(rgba, keep_alpha=False)


def compute_and_save_tsne(
    descriptor_dir: str,
    output_path: str,
    pca_components: int = 50,
    perplexity: float = 80,
    learning_rate: float = 200,
    early_exaggeration: float = 15,
    early_exaggeration_iter: int = 250,
    n_iter: int = 750,
    random_state: int = 42
) -> None:
    """
    Load descriptors, run PCA + openTSNE, and save embedding and labels to disk.
    """
    # Collect files
    files = sorted(glob(os.path.join(descriptor_dir, '*.npy')))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {descriptor_dir}")

    all_desc = []
    labels = []
    for path in files:
        data = np.load(path)
        all_desc.append(data)
        name = os.path.splitext(os.path.basename(path))[0]
        labels.extend([name] * len(data))

    X = np.concatenate(all_desc, axis=0)
    # PCA reduction
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # t-SNE embedding
    tsne = openTSNE.TSNE(
        n_jobs=4,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        early_exaggeration_iter=early_exaggeration_iter,
        n_iter=n_iter,
        random_state=random_state,
        verbose=True
    )
    X_embedded = tsne.fit(X_pca)

    # Save
    np.savez_compressed(
        output_path,
        embedding=X_embedded.astype(np.float32),
        labels=np.array(labels, dtype='<U32')
    )
    print(f"Saved t-SNE results to {output_path}")


def plot_tsne(
    npz_path: str,
    color_dict: dict,
    figure_path: str = 'tsne.png',
    eps_path: str = 'tsne.eps',
    figsize: tuple = (8, 6),
    point_size: float = 5,
    alpha: float = 0.5
) -> None:
    """
    Load t-SNE embedding and labels, and create a scatter plot.
    """
    data = np.load(npz_path)
    X_emb = data['embedding']
    labels = data['labels'].tolist()

    # Map colors
    colors = [color_dict.get(lbl, '#333333') for lbl in labels]

    plt.figure(figsize=figsize)
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'Arial'
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.scatter(
        X_emb[:, 0], X_emb[:, 1],
        c=colors, s=point_size,
        alpha=alpha, edgecolors='w', linewidths=0.1
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label=lbl,
               markerfacecolor=c, markersize=8)
        for lbl, c in color_dict.items()
    ]
    plt.legend(
        handles=legend_elems,
        title='Materials',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=False
    )

    # Clean axes
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    plt.savefig(eps_path, dpi=600, bbox_inches='tight')
    print(f"Saved figures to {figure_path} and {eps_path}")

def plot_tsne_per_system(
    npz_path: str,
    color_dict: dict,
    png_path: str = 'tsne_persystem.png',
    eps_path: str = 'tsne_persystem.eps',
    figsize_per_plot: tuple = (2, 2),
    point_size: float = 5,
    alpha: float = 0.5
) -> None:
    """
    Load t-SNE embedding and labels, and create a grid of subplots, one per unique label.
    For each subplot, points of the current label are shown in color, others in gray.
    """
    plt.rcParams['font.family'] = 'Arial'
    # Load data
    data = np.load(npz_path)
    X_emb = data['embedding']
    labels = data['labels'].tolist()

    # Unique labels
    unique_labels = [lbl for lbl in color_dict if lbl in labels]
    n = len(unique_labels)

    # Determine grid size
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(figsize_per_plot[0]*ncols,
                                      figsize_per_plot[1]*nrows),
                             squeeze=False)

    # Plot each label
    for idx, lbl in tqdm(enumerate(unique_labels)):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Masks
        mask = np.array(labels) == lbl
        other_mask = ~mask

        # Plot others in gray
        ax.scatter(
            X_emb[other_mask, 0], X_emb[other_mask, 1],
            c='#CCCCCC', s=point_size,
            alpha=alpha, edgecolors='w', linewidths=0.1
        )
        # Plot current label in color
        ax.scatter(
            X_emb[mask, 0], X_emb[mask, 1],
            c=color_dict.get(lbl, '#333333'), s=point_size,
            alpha=alpha, edgecolors='w', linewidths=0.1
        )

        # Title and aesthetics
        ax.set_title(lbl, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Turn off unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save
    fig.savefig(png_path, dpi=600, bbox_inches='tight')
    fig.savefig(eps_path, dpi=600, bbox_inches='tight')
    print(f"Saved per-system t-SNE figures to {png_path} and {eps_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='t-SNE pipeline')
    parser.add_argument('--compute', action='store_true', help='Compute and save t-SNE')
    parser.add_argument('--plot', action='store_true', help='Plot from saved t-SNE')
    parser.add_argument('--descriptor-dir', type=str, default='descriptors/')
    parser.add_argument('--output-npz', type=str, default='tsne_results.npz')
    parser.add_argument('--png', type=str, default='tsne.png')
    parser.add_argument('--eps', type=str, default='tsne.eps')
    args = parser.parse_args()

    # Define your color mapping here or import from a config file
    color_dict = {
        'Az2Cu': get_intermediate_color('#b09085', '#f48534', 1.0),
        'Az2Hg': get_intermediate_color('#b09085', '#f48534', 0.8),
        'Az2Pb': get_intermediate_color('#b09085', '#f48534', 0.6),
        'AzAg': get_intermediate_color('#b09085', '#f48534', 0.4),
        'AzCu': get_intermediate_color('#b09085', '#f48534', 0.2),
        'AzTl': get_intermediate_color('#b09085', '#f48534', 0.0),
        'AN': '#060ef6',
        'AP': '#f30050',
        'DAP-1': get_intermediate_color('#7275e6', '#c24179', 0.25),
        'DAP-2': get_intermediate_color('#7275e6', '#c24179', 0.35),
        'DAP-3': get_intermediate_color('#7275e6', '#c24179', 0.45),
        'DAP-4': get_intermediate_color('#7275e6', '#c24179', 0.55),
        'DAP-5': get_intermediate_color('#7275e6', '#c24179', 0.65),
        'DAP-6': get_intermediate_color('#7275e6', '#c24179', 0.75),
        'DAP-7': get_intermediate_color('#7275e6', '#c24179', 0.85),
        'DAP-M4': get_intermediate_color('#7275e6', '#c24179', 0.95),
        'CL-20': get_intermediate_color('#a0b085', '#28E0C1', 0.95),
        'HMX': get_intermediate_color('#a0b085', '#28E0C1', 0.65),
        'RDX': get_intermediate_color('#a0b085', '#28E0C1', 0.45),
        'TNT': get_intermediate_color('#a0b085', '#28E0C1', 0.05)
    }

    if args.compute:
        compute_and_save_tsne(
            descriptor_dir=args.descriptor_dir,
            output_path=args.output_npz
        )
    if args.plot:
        plot_tsne_per_system(
            args.output_npz, 
            color_dict
        )
        plot_tsne(
            npz_path=args.output_npz,
            color_dict=color_dict,
            figure_path=args.png,
            eps_path=args.eps
        )
