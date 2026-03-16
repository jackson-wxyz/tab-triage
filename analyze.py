#!/usr/bin/env python3
"""
Tab Triage Analysis: Clustering, Seriation, and Visualization.

Run this AFTER main.py has produced triage_results.db with embeddings.
Loads the embeddings from SQLite, clusters tabs, seriates them, and produces:

1. Cluster assignments with auto-derived category names
2. Seriation order (an alternative to "quick-win" ordering)
3. A beautiful 5D RGB-encoded UMAP scatter plot
4. Updated CSV/SQLite with cluster and seriation columns

Usage:
    python analyze.py                           # uses default output/ dir
    python analyze.py --db path/to/results.db   # custom db path
    python analyze.py --mock                    # generate fake data for testing
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, optimal_leaf_ordering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

import config

# ─── Data Loading ────────────────────────────────────────────────────────────

@dataclass
class TabRecord:
    """A tab with its triage data and embedding."""
    url: str
    title: str
    summary: str
    category: str  # LLM-assigned category
    actionability: int
    implied_action: str
    importance: int
    effort: int
    staleness: int
    insight_density: int
    quick_win_score: float
    embedding: list = field(default_factory=list, repr=False)

    # Added by analysis
    cluster_id: int = -1
    cluster_name: str = ""
    seriation_order: int = -1

    @property
    def short_title(self):
        t = self.title
        if len(t) > 40:
            t = t[:37] + "..."
        return t


def load_from_sqlite(db_path: str) -> list[TabRecord]:
    """Load tab records with embeddings from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT url, title, summary, category, actionability, implied_action,
               importance, effort, staleness, insight_density, quick_win_score,
               embedding
        FROM tabs
        WHERE embedding IS NOT NULL
        ORDER BY rowid
    """)

    tabs = []
    for row in cursor.fetchall():
        embedding = json.loads(row[11]) if row[11] else []
        if not embedding:
            continue
        tabs.append(TabRecord(
            url=row[0], title=row[1] or row[0], summary=row[2] or "",
            category=row[3] or "", actionability=row[4] or 3,
            implied_action=row[5] or "", importance=row[6] or 3,
            effort=row[7] or 3, staleness=row[8] or 3,
            insight_density=row[9] or 3, quick_win_score=row[10] or 0,
            embedding=embedding,
        ))

    conn.close()
    print(f"Loaded {len(tabs)} tabs with embeddings from {db_path}")
    return tabs


def generate_mock_tabs(n: int = 80) -> list[TabRecord]:
    """Generate fake tab data for testing visualizations."""
    import hashlib
    import random

    categories = [
        "AI safety", "health/nutrition", "personal finance",
        "home improvement", "rationality", "space industry",
        "gaming", "EA/philanthropy", "technology", "politics/policy",
    ]

    rng = random.Random(42)
    tabs = []

    for i in range(n):
        cat = categories[i % len(categories)]
        # Create clustered embeddings: items in same category are close together
        base_vec = np.zeros(768)
        # Category "center" — each category gets a different region
        cat_idx = categories.index(cat)
        for d in range(768):
            seed = int(hashlib.md5(f"{cat}_{d}".encode()).hexdigest()[:8], 16)
            base_vec[d] = (seed % 1000) / 500.0 - 1.0
        # Add individual noise
        noise = np.array([rng.gauss(0, 0.3) for _ in range(768)])
        vec = base_vec + noise
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding = vec.tolist()

        tabs.append(TabRecord(
            url=f"https://example.com/article-{i}",
            title=f"{cat}: Article {i} about interesting things",
            summary=f"A mock article about {cat.lower()} with some insights.",
            category=cat,
            actionability=rng.randint(1, 5),
            implied_action="Read for interest" if rng.random() < 0.5 else "Research and implement",
            importance=rng.randint(1, 5),
            effort=rng.randint(1, 5),
            staleness=rng.randint(1, 5),
            insight_density=rng.randint(1, 5),
            quick_win_score=round(rng.uniform(0.5, 5.0), 2),
            embedding=embedding,
        ))

    return tabs


# ─── Clustering ──────────────────────────────────────────────────────────────

def build_embedding_matrix(tabs: list[TabRecord]) -> np.ndarray:
    """Stack all tab embeddings into a matrix."""
    return np.array([t.embedding for t in tabs])


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance matrix."""
    dist = cosine_distances(embeddings)
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # ensure symmetry
    return dist


def find_optimal_k(embeddings: np.ndarray, max_k: int = 20) -> tuple[int, dict]:
    """
    Use silhouette score to find the best number of clusters,
    respecting the MIN_CLUSTERS floor from config.

    With real tab data, silhouette often picks k=2 or k=3 (one huge
    "normal" cluster + one tiny "glitched pages" cluster), which isn't
    useful.  MIN_CLUSTERS forces more granularity.
    """
    min_k = getattr(config, 'MIN_CLUSTERS', 2)
    n = len(embeddings)
    max_k = min(max_k, n - 1)
    min_k = min(min_k, max_k)  # can't have more clusters than items
    scores = {}

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(embeddings, labels, metric='cosine')

    best_k = max(scores, key=scores.get)
    print(f"  Optimal k={best_k} (silhouette={scores[best_k]:.3f})"
          f"  [searched k={min_k}..{max_k}]")
    return best_k, scores


def cluster_tabs(tabs: list[TabRecord], embeddings: np.ndarray,
                 dist_matrix: np.ndarray) -> dict:
    """
    Cluster tabs using hierarchical + k-means.
    Returns cluster info including labels and linkage.
    """
    n = len(tabs)
    condensed = squareform(dist_matrix)

    # Hierarchical clustering
    Z = linkage(condensed, method='average')

    # Find optimal k
    best_k, sil_scores = find_optimal_k(embeddings, max_k=min(20, n // 3))

    # K-means with optimal k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    return {
        'linkage': Z,
        'labels': labels,
        'n_clusters': best_k,
        'silhouette_scores': sil_scores,
        'centroids': km.cluster_centers_,
    }


def derive_cluster_names(tabs: list[TabRecord], cluster_labels: np.ndarray) -> dict:
    """
    Derive cluster names by looking at the most common LLM-assigned categories
    and keywords in each cluster's titles and summaries.
    """
    n_clusters = len(set(cluster_labels))
    cluster_names = {}

    for c in range(n_clusters):
        members = [tabs[i] for i in range(len(tabs)) if cluster_labels[i] == c]
        if not members:
            cluster_names[c] = f"Cluster {c}"
            continue

        # Most common LLM-assigned category in this cluster
        cat_counts = Counter(m.category for m in members if m.category)
        top_cats = cat_counts.most_common(2)

        if top_cats:
            primary = top_cats[0][0]
            if len(top_cats) > 1 and top_cats[1][1] >= top_cats[0][1] * 0.5:
                # Two strong categories — combine them
                cluster_names[c] = f"{primary} / {top_cats[1][0]}"
            else:
                cluster_names[c] = primary
        else:
            cluster_names[c] = f"Cluster {c}"

        # Add count
        cluster_names[c] = f"{cluster_names[c]} ({len(members)})"

    return cluster_names


# ─── Seriation ───────────────────────────────────────────────────────────────

def seriate_greedy_nn(dist_matrix: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor seriation. O(n²), fast even for 500+ items."""
    n = dist_matrix.shape[0]
    # Start from the most "outlier" point (highest avg distance)
    start = np.argmax(dist_matrix.mean(axis=1))
    order = [start]
    visited = {start}

    for _ in range(n - 1):
        current = order[-1]
        dists = dist_matrix[current].copy()
        dists[list(visited)] = np.inf
        nearest = np.argmin(dists)
        order.append(nearest)
        visited.add(nearest)

    return order


def seriate_optimal_leaf(dist_matrix: np.ndarray) -> list[int]:
    """Optimal leaf ordering on hierarchical clustering. Good quality, O(n³)."""
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    Z_opt = optimal_leaf_ordering(Z, condensed)
    return list(leaves_list(Z_opt))


def seriate_spectral(dist_matrix: np.ndarray) -> list[int]:
    """Spectral seriation via the Fiedler vector. O(n²), good for manifolds."""
    from scipy.sparse.csgraph import laplacian

    sigma = np.median(dist_matrix[dist_matrix > 0])
    if sigma == 0:
        sigma = 1.0
    similarity = np.exp(-dist_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(similarity, 0)
    L = laplacian(similarity, normed=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]
    return list(np.argsort(fiedler))


def seriate_tsp_2opt(dist_matrix: np.ndarray, n_restarts: int = 2,
                     max_iters: int = 200) -> list[int]:
    """
    TSP 2-opt local search. Capped iterations for 500+ item scalability.
    """
    best_order = None
    best_cost = float('inf')

    for restart in range(n_restarts):
        if restart == 0:
            order = seriate_greedy_nn(dist_matrix)
        else:
            order = list(np.random.RandomState(restart).permutation(dist_matrix.shape[0]))

        n = len(order)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(1, n - 1):
                for j in range(i + 1, min(i + 50, n)):  # limit scan window
                    old = dist_matrix[order[i-1], order[i]]
                    if j + 1 < n:
                        old += dist_matrix[order[j], order[j+1]]
                    new = dist_matrix[order[i-1], order[j]]
                    if j + 1 < n:
                        new += dist_matrix[order[i], order[j+1]]
                    if new < old - 1e-10:
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True

        current_cost = sum(dist_matrix[order[k], order[k+1]] for k in range(n-1))
        if current_cost < best_cost:
            best_cost = current_cost
            best_order = order[:]

    return best_order


def path_cost(order: list[int], dist_matrix: np.ndarray) -> float:
    return sum(dist_matrix[order[i], order[i+1]] for i in range(len(order) - 1))


def detect_cluster_boundaries(order: list[int], dist_matrix: np.ndarray) -> list[int]:
    """Find natural breakpoints in the seriation (large jumps in distance)."""
    adj = [dist_matrix[order[i], order[i+1]] for i in range(len(order) - 1)]
    mean_d = np.mean(adj)
    std_d = np.std(adj)
    threshold = mean_d + 0.8 * std_d
    boundaries = [i for i, d in enumerate(adj) if d > threshold]
    # Cap at a reasonable number
    max_boundaries = int(np.sqrt(len(order)))
    if len(boundaries) > max_boundaries:
        sorted_adj = sorted(enumerate(adj), key=lambda x: x[1], reverse=True)
        boundaries = sorted([idx for idx, _ in sorted_adj[:max_boundaries]])
    return boundaries


# ─── Visualization ───────────────────────────────────────────────────────────

def plot_rgb_umap(tabs: list[TabRecord], embeddings: np.ndarray,
                  cluster_labels: np.ndarray, cluster_names: dict,
                  seriation_order: list[int], filename: str):
    """
    5D RGB-encoded UMAP: the crown jewel visualization.

    Dims 1-2 = XY position on the plot.
    Dims 3-5 = RGB color of each dot.

    Adapted for 500+ items: smaller dots, selective labeling,
    seriation path drawn as a faint connecting line.
    """
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping RGB UMAP plot")
        print("  Install with: pip install umap-learn")
        return

    n = len(tabs)
    n_neighbors = min(15, n - 2)

    print("  Running 5D UMAP projection...")
    reducer = umap.UMAP(
        n_components=5, n_neighbors=n_neighbors, min_dist=0.15,
        metric='cosine', random_state=42, spread=1.5,
    )
    coords = reducer.fit_transform(embeddings)

    # Normalize each dimension to [0, 1]
    normed = np.zeros_like(coords)
    for d in range(coords.shape[1]):
        mn, mx = coords[:, d].min(), coords[:, d].max()
        if mx - mn > 1e-10:
            normed[:, d] = (coords[:, d] - mn) / (mx - mn)
        else:
            normed[:, d] = 0.5

    # RGB from dims 3, 4, 5 (indices 2, 3, 4)
    rgb_colors = np.column_stack([normed[:, 2], normed[:, 3], normed[:, 4]])
    rgb_colors = 0.15 + 0.7 * rgb_colors  # keep in visible range

    # ── Build the figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 14))

    # Main scatter plot (left ~70%)
    ax = fig.add_axes([0.05, 0.08, 0.60, 0.85])

    # Dot size scales inversely with count
    dot_size = max(15, min(120, 3000 / n))
    edge_width = max(0.15, min(0.5, 15 / n))

    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=rgb_colors, s=dot_size,
        edgecolors='black', linewidths=edge_width,
        alpha=0.85, zorder=3,
    )

    #zoom into the good part of the plot
    #plt.xlim(-8.5,-1)
    #plt.ylim(3,8)

    # Draw seriation path as a faint connecting line
    if seriation_order:
        path_x = [coords[seriation_order[i], 0] for i in range(len(seriation_order))]
        path_y = [coords[seriation_order[i], 1] for i in range(len(seriation_order))]
        ax.plot(path_x, path_y, '-', color='gray', alpha=0.12, linewidth=0.5, zorder=1)

    # Selective labeling: label the most "important" or highest quick-win tabs
    # plus some random ones for coverage
    n_labels = min(300, n)#was 40
    # Pick top quick-win, top importance, and some random
    scored = sorted(range(n), key=lambda i: tabs[i].quick_win_score, reverse=True)
    label_indices = set(scored[:n_labels // 3])
    scored_imp = sorted(range(n), key=lambda i: tabs[i].importance, reverse=True)
    label_indices.update(scored_imp[:n_labels // 3])
    # Fill rest with evenly spaced from seriation order
    if seriation_order:
        step = max(1, len(seriation_order) // (n_labels - len(label_indices) + 1))
        for i in range(0, len(seriation_order), step):
            label_indices.add(seriation_order[i])
            if len(label_indices) >= n_labels:
                break

    for i in label_indices:
        ax.annotate(
            tabs[i].short_title[:45],#was :30 for shorter title length
            (coords[i, 0], coords[i, 1]),
            fontsize=4.5, ha='center', va='bottom',
            xytext=(0, 4), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      alpha=0.65, edgecolor='none'),
        )

    ax.set_xlabel('UMAP Dim 1', fontsize=11)
    ax.set_ylabel('UMAP Dim 2', fontsize=11)
    ax.set_title(
        f'5D UMAP Tab Map — {n} tabs\n'
        'Position = Dims 1–2  |  Color = Dims 3–5 (RGB)',
        fontsize=14, fontweight='bold',
    )

    # Color encoding legend text
    legend_text = (
        "Color encoding:\n"
        "  R channel = UMAP Dim 3\n"
        "  G channel = UMAP Dim 4\n"
        "  B channel = UMAP Dim 5\n\n"
        "Similar colors → similar\n"
        "on hidden dimensions"
    )
    ax.text(
        0.02, 0.02, legend_text, transform=ax.transAxes,
        fontsize=7, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        fontfamily='monospace',
    )

    # ── Color key panel (right ~25%) ──────────────────────────────────
    ax2 = fig.add_axes([0.70, 0.08, 0.28, 0.85])

    grid_size = 20
    slices = [(0.2, 'Low Dim 5'), (0.5, 'Mid Dim 5'), (0.85, 'High Dim 5')]
    total_height = len(slices) * (grid_size + 3)

    for s_idx, (b_level, b_label) in enumerate(slices):
        y_offset = s_idx * (grid_size + 3)
        for ri in range(grid_size):
            for gi in range(grid_size):
                r = 0.15 + 0.7 * (ri / (grid_size - 1))
                g = 0.15 + 0.7 * (gi / (grid_size - 1))
                b = 0.15 + 0.7 * b_level
                ax2.add_patch(plt.Rectangle(
                    (ri, gi + y_offset), 1, 1,
                    facecolor=(r, g, b), edgecolor='none',
                ))
        ax2.text(grid_size / 2, y_offset - 1.5, b_label,
                 ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlim(-2, grid_size + 2)
    ax2.set_ylim(-3, total_height + 1)
    ax2.set_xlabel('← Low Dim 3     High Dim 3 →', fontsize=9)
    ax2.set_ylabel('← Low Dim 4     High Dim 4 →', fontsize=9)
    ax2.set_title('RGB Color Key', fontsize=13, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.tick_params(labelbottom=False, labelleft=False)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_cluster_umap(tabs: list[TabRecord], embeddings: np.ndarray,
                      cluster_labels: np.ndarray, cluster_names: dict,
                      filename: str):
    """
    2D UMAP colored by cluster assignment.

    More interpretable than the 5D RGB plot for understanding "what are
    my tabs about" — each cluster gets a distinct color with a legend.
    """
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping cluster UMAP plot")
        return

    n = len(tabs)
    n_neighbors = min(15, n - 2)
    n_clusters = len(set(cluster_labels))

    print("  Running 2D UMAP projection...")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.15,
        metric='cosine', random_state=42, spread=1.5,
    )
    coords = reducer.fit_transform(embeddings)

    # Use a qualitative colormap with enough distinct colors
    if n_clusters <= 10:
        cmap = cm.get_cmap('tab10', 10)
    elif n_clusters <= 20:
        cmap = cm.get_cmap('tab20', 20)
    else:
        cmap = cm.get_cmap('gist_ncar', n_clusters)

    dot_size = max(20, min(150, 3500 / n))
    edge_width = max(0.2, min(0.6, 20 / n))

    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot each cluster separately for the legend
    unique_clusters = sorted(set(cluster_labels))
    for c in unique_clusters:
        mask = cluster_labels == c
        name = cluster_names.get(c, f"Cluster {c}")
        color = cmap(c % cmap.N)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], s=dot_size, label=name,
            edgecolors='black', linewidths=edge_width,
            alpha=0.8, zorder=3,
        )

        # Add a cluster centroid label
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        # Strip the count suffix for the centroid label
        short_name = name.rsplit(' (', 1)[0] if ' (' in name else name
        ax.annotate(
            short_name,
            (cx, cy),
            fontsize=9, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.85, edgecolor=color, linewidth=1.5),
            zorder=5,
        )

    # Selective individual labels for high-importance tabs
    n_labels = min(30, n // 3)
    scored = sorted(range(n), key=lambda i: tabs[i].importance, reverse=True)
    for i in scored[:n_labels]:
        ax.annotate(
            tabs[i].short_title[:35],
            (coords[i, 0], coords[i, 1]),
            fontsize=5, ha='center', va='bottom',
            xytext=(0, 5), textcoords='offset points',
            alpha=0.7,
        )

    ax.set_xlabel('UMAP Dim 1', fontsize=11)
    ax.set_ylabel('UMAP Dim 2', fontsize=11)
    ax.set_title(
        f'Tab Clusters — {n} tabs, {n_clusters} clusters\n'
        'Colors = cluster assignment  |  Labels = cluster names',
        fontsize=14, fontweight='bold',
    )

    # Legend outside the plot
    ax.legend(
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.9, title='Clusters',
        title_fontsize=10,
    )

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_cluster_summary(tabs: list[TabRecord], cluster_labels: np.ndarray,
                         cluster_names: dict, filename: str):
    """
    Horizontal bar chart showing cluster sizes and avg actionability,
    plus a table of cluster contents.
    """
    n_clusters = len(set(cluster_labels))
    cmap = cm.get_cmap('tab10', max(n_clusters, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, n_clusters * 0.8)),
                                    gridspec_kw={'width_ratios': [1, 1.5]})

    # Left: bar chart of cluster sizes colored by avg actionability
    cluster_ids = sorted(set(cluster_labels))
    sizes = [sum(1 for l in cluster_labels if l == c) for c in cluster_ids]
    avg_actions = []
    for c in cluster_ids:
        members = [tabs[i] for i in range(len(tabs)) if cluster_labels[i] == c]
        avg_actions.append(np.mean([m.actionability for m in members]))

    names = [cluster_names.get(c, f"Cluster {c}") for c in cluster_ids]
    colors = [cmap(c % 10) for c in cluster_ids]

    y_pos = range(len(cluster_ids))
    bars = ax1.barh(y_pos, sizes, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel('Number of Tabs')
    ax1.set_title('Cluster Sizes', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()

    # Add avg actionability as text on bars
    for i, (bar, avg_a) in enumerate(zip(bars, avg_actions)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'avg action: {avg_a:.1f}/5',
                va='center', fontsize=7, color='gray')

    # Right: top 3 tabs per cluster
    ax2.axis('off')
    text_lines = []
    for c in cluster_ids:
        name = cluster_names.get(c, f"Cluster {c}")
        members = [(i, tabs[i]) for i in range(len(tabs)) if cluster_labels[i] == c]
        top = sorted(members, key=lambda x: x[1].quick_win_score, reverse=True)[:3]
        text_lines.append(f"▸ {name}")
        for _, t in top:
            text_lines.append(f"    • {t.short_title}  [action={t.actionability}, imp={t.importance}]")
        text_lines.append("")

    ax2.text(0.02, 0.98, '\n'.join(text_lines),
             transform=ax2.transAxes, fontsize=7, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f8f8', alpha=0.9))
    ax2.set_title('Top Tabs Per Cluster (by quick-win score)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_seriation_quality(tabs, dist_matrix, seriation_order, boundaries, filename):
    """Plot adjacent distances along the seriation path, highlighting cluster breaks."""
    adj = [dist_matrix[seriation_order[i], seriation_order[i+1]]
           for i in range(len(seriation_order) - 1)]
    n = len(adj)

    fig, ax = plt.subplots(1, 1, figsize=(max(14, n * 0.06), 5))

    colors = ['#4CAF50'] * n
    for b in boundaries:
        if b < n:
            colors[b] = '#F44336'

    ax.bar(range(n), adj, color=colors, alpha=0.8, width=1.0, edgecolor='none')
    mean_d = np.mean(adj)
    ax.axhline(y=mean_d, color='gray', linestyle='--', alpha=0.5,
               label=f'Mean: {mean_d:.4f}')

    ax.set_xlabel(f'Seriation Position (1 to {n+1})')
    ax.set_ylabel('Cosine Distance to Next Tab')
    total = sum(adj)
    ax.set_title(
        f'Seriation Quality — {n+1} tabs, total path cost = {total:.3f}\n'
        f'Red bars = cluster boundaries (large gaps)',
        fontsize=12, fontweight='bold',
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# ─── Output ──────────────────────────────────────────────────────────────────

def update_sqlite(db_path: str, tabs: list[TabRecord]):
    """Add cluster and seriation columns to the existing SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add new columns if they don't exist
    for col, col_type in [('cluster_id', 'INTEGER'), ('cluster_name', 'TEXT'),
                          ('seriation_order', 'INTEGER')]:
        try:
            cursor.execute(f"ALTER TABLE tabs ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists

    for tab in tabs:
        cursor.execute("""
            UPDATE tabs
            SET cluster_id = ?, cluster_name = ?, seriation_order = ?
            WHERE url = ?
        """, (tab.cluster_id, tab.cluster_name, tab.seriation_order, tab.url))

    conn.commit()
    conn.close()


def write_seriated_csv(tabs: list[TabRecord], seriation_order: list[int],
                       filename: str):
    """Write a CSV sorted by seriation order (alternative to quick-win sort)."""
    import csv

    columns = [
        "seriation_pos", "url", "title", "summary", "category",
        "cluster_name", "actionability", "implied_action",
        "importance", "effort", "staleness", "insight_density", "quick_win_score",
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for pos, idx in enumerate(seriation_order):
            t = tabs[idx]
            writer.writerow({
                "seriation_pos": pos + 1,
                "url": t.url,
                "title": t.title,
                "summary": t.summary,
                "category": t.category,
                "cluster_name": t.cluster_name,
                "actionability": t.actionability,
                "implied_action": t.implied_action,
                "importance": t.importance,
                "effort": t.effort,
                "staleness": t.staleness,
                "insight_density": t.insight_density,
                "quick_win_score": t.quick_win_score,
            })

    print(f"  Wrote seriated CSV: {filename}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tab Triage Analysis: clustering, seriation, and visualization",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to triage_results.db (default: output/triage_results.db)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Generate mock data for testing (no DB needed)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for visualizations",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # ── Load data ────────────────────────────────────────────────────
    if args.mock:
        print("\n" + "=" * 60)
        print("  Tab Triage Analysis [MOCK MODE]")
        print("=" * 60)
        tabs = generate_mock_tabs(80)
    else:
        db_path = args.db or os.path.join(output_dir, config.SQLITE_FILENAME)
        if not os.path.exists(db_path):
            print(f"Error: database not found at {db_path}")
            print("Run main.py first, or use --mock for testing.")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("  Tab Triage Analysis")
        print("=" * 60)
        tabs = load_from_sqlite(db_path)

    if len(tabs) < 3:
        print("Need at least 3 tabs with embeddings. Exiting.")
        sys.exit(1)

    n = len(tabs)

    # ── Build embedding matrix & distances ────────────────────────────
    print(f"\n📐 Building distance matrix for {n} tabs...")
    embeddings = build_embedding_matrix(tabs)
    embeddings = normalize(embeddings)  # L2 normalize
    dist_matrix = compute_distance_matrix(embeddings)

    # ── Clustering ────────────────────────────────────────────────────
    print(f"\n🔬 Clustering...")
    cluster_info = cluster_tabs(tabs, embeddings, dist_matrix)
    cluster_labels = cluster_info['labels']
    cluster_names = derive_cluster_names(tabs, cluster_labels)

    print(f"\n  {cluster_info['n_clusters']} clusters found:")
    for c in sorted(set(cluster_labels)):
        print(f"    {cluster_names[c]}")

    # Apply to tabs
    for i, tab in enumerate(tabs):
        tab.cluster_id = int(cluster_labels[i])
        tab.cluster_name = cluster_names[cluster_labels[i]]

    # ── Seriation ─────────────────────────────────────────────────────
    print(f"\n🔀 Running seriation algorithms...")

    seriations = {}

    print("  Greedy NN...")
    seriations['Greedy NN'] = seriate_greedy_nn(dist_matrix)

    print("  Optimal Leaf Order...")
    seriations['Optimal Leaf Order'] = seriate_optimal_leaf(dist_matrix)

    print("  Spectral (Fiedler)...")
    seriations['Spectral'] = seriate_spectral(dist_matrix)

    if n <= 300:
        print("  TSP 2-opt (capped at 300 items)...")
        seriations['TSP 2-opt'] = seriate_tsp_2opt(dist_matrix, n_restarts=2,
                                                     max_iters=100)

    # Evaluate and pick best
    print("\n  Method comparison:")
    best_method = None
    best_cost = float('inf')
    for name, order in seriations.items():
        cost = path_cost(order, dist_matrix)
        mean_adj = cost / (len(order) - 1)
        print(f"    {name:25s}  total={cost:.4f}  mean_adj={mean_adj:.4f}")
        if cost < best_cost:
            best_cost = cost
            best_method = name

    print(f"\n  🏆 Best: {best_method}")
    best_order = seriations[best_method]
    boundaries = detect_cluster_boundaries(best_order, dist_matrix)

    # Apply seriation order to tabs
    for pos, idx in enumerate(best_order):
        tabs[idx].seriation_order = pos + 1

    # ── Visualizations ────────────────────────────────────────────────
    print(f"\n🎨 Generating visualizations...")

    print("\n  [1/4] 2D Cluster UMAP plot...")
    plot_cluster_umap(
        tabs, embeddings, cluster_labels, cluster_names,
        os.path.join(output_dir, 'tab_clusters_2d.png'),
    )

    print("\n  [2/4] 5D RGB UMAP plot...")
    plot_rgb_umap(
        tabs, embeddings, cluster_labels, cluster_names,
        best_order, os.path.join(output_dir, 'tab_map_5d_rgb.png'),
    )

    print("\n  [3/4] Cluster summary...")
    plot_cluster_summary(
        tabs, cluster_labels, cluster_names,
        os.path.join(output_dir, 'cluster_summary.png'),
    )

    print("\n  [4/4] Seriation quality...")
    plot_seriation_quality(
        tabs, dist_matrix, best_order, boundaries,
        os.path.join(output_dir, 'seriation_quality.png'),
    )

    # ── Output files ──────────────────────────────────────────────────
    print(f"\n💾 Writing outputs...")

    # Seriation-ordered CSV
    write_seriated_csv(
        tabs, best_order,
        os.path.join(output_dir, 'triage_seriated.csv'),
    )

    # Update SQLite if not mock
    if not args.mock:
        db_path = args.db or os.path.join(output_dir, config.SQLITE_FILENAME)
        update_sqlite(db_path, tabs)
        print(f"  Updated SQLite: {db_path}")

    # Print seriation order with boundaries
    print("\n" + "=" * 60)
    print(f"  SERIATION ORDER ({best_method})")
    print("=" * 60)
    section = 1
    print(f"\n  ── Section {section} ──")
    for pos, idx in enumerate(best_order[:30]):  # show first 30
        t = tabs[idx]
        print(f"    {pos+1:3d}. {t.short_title:40s}  [{t.cluster_name}]")
        if pos in boundaries and pos < len(best_order) - 1:
            section += 1
            print(f"         ─── gap ───")
            print(f"  ── Section {section} ──")
    if len(best_order) > 30:
        print(f"    ... ({len(best_order) - 30} more tabs)")

    print(f"\n  Outputs in {output_dir}/:")
    print(f"    tab_clusters_2d.png   — 2D UMAP with cluster colors")
    print(f"    tab_map_5d_rgb.png    — 5D UMAP visualization")
    print(f"    cluster_summary.png   — cluster overview")
    print(f"    seriation_quality.png — seriation adjacent distances")
    print(f"    triage_seriated.csv   — tabs in seriation order")
    print()


if __name__ == '__main__':
    main()
