import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import TwoSlopeNorm, Normalize


def visualize_neuron_weight(weights, layer_name, neuron_idx, norm, out_dir="output", width=None, height=None):
    """
    Generate and save a 2D heatmap and bar chart (true-value histogram) for a single neuron's weights.
    - width, height: if provided and matching, reshape accordingly; otherwise fallback to 1×N.
    """
    w = np.array(weights)
    n = w.shape[0]

    # Determine 2D reshape shape
    if width is not None and height is not None and width * height == n:
        w2d = w.reshape((height, width))
    elif width is not None and n % width == 0:
        h = n // width
        w2d = w.reshape((h, width))
    else:
        w2d = w.reshape((1, n))
        print(f"[!] Neuron #{neuron_idx} weight length {n} does not match shape, using 1×{n}")

    # Create figure with two subplots
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    # 2D heatmap
    im = sns.heatmap(w2d, cmap='seismic', norm=norm, cbar=True, ax=ax0)
    ax0.set_title(f'{layer_name} Neuron #{neuron_idx} - 2D Map')
    ax0.set_xlabel('Col')
    ax0.set_ylabel('Row')
    for spine in ax0.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # Only show the neuron's actual unique weights on the colorbar ticks
    unique_vals = np.unique(w)
    cbar = im.collections[0].colorbar
    cbar.set_ticks(unique_vals)
    cbar.set_ticklabels([f"{v:.3f}" for v in unique_vals])

    # True-value histogram: bar chart of unique weights
    vals, counts = unique_vals, np.unique(w, return_counts=True)[1]
    total_count = counts.sum()

    # Compute bar width based on smallest gap between adjacent values
    diffs = np.diff(vals)
    if diffs.size > 0:
        bar_width = diffs.min() * 0.8
    else:
        bar_width = 0.1

    bars = ax1.bar(vals, counts, width=bar_width, edgecolor='black', alpha=0.7)

    # Annotate percentages above each bar
    for v, c in zip(vals, counts):
        pct = c / total_count * 100
        ax1.text(v, c, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # X-axis ticks are the true weight values
    ax1.set_xticks(vals)
    ax1.set_xticklabels([f"{v:.3f}" for v in vals], rotation=45, ha='right')
    ax1.set_title(f'{layer_name} Neuron #{neuron_idx} - Histogram')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    fig.suptitle(f'{layer_name} - Neuron #{neuron_idx}')
    plt.tight_layout()

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{layer_name.replace(".", "_")}_neuron_{neuron_idx}.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'[✓] Saved: {out_path}')


if __name__ == "__main__":
    json_path = "analysis_output_0622/kmeans_4_quantized_network_prune.json"
    with open(json_path, 'r') as f:
        data = json.load(f)

    selected_layers = [
        "network.0.weight",

    ]
    layer_shapes = {
        "network.0.weight": (28, 28),

    }
    default_width = 8

    base_out = "archive/0626_weight_picture_prune_k4"
    for layer_name in selected_layers:
        weights_list = data.get(layer_name)
        if weights_list is None:
            print(f"[!] Layer {layer_name} does not exist, skipping")
            continue

        arr = np.array(weights_list)
        if arr.ndim != 2:
            print(f"[!] Layer {layer_name} is not 2D weights, skipping")
            continue

        vmin, vmax = arr.min(), arr.max()
        if vmin < 0 and vmax > 0:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            print(f"[!] Layer {layer_name} using Normalize (vmin={vmin}, vmax={vmax})")

        width, height = layer_shapes.get(layer_name, (default_width, None))
        out_dir = os.path.join(base_out, layer_name.replace('.', '_'))
        total_neurons = arr.shape[0]
        print(f"Output Layer {layer_name}: {total_neurons} neurons, shape=({width},{height}) -> {out_dir}")

        for idx, w in enumerate(weights_list):
            visualize_neuron_weight(
                weights=w,
                layer_name=layer_name,
                neuron_idx=idx,
                norm=norm,
                out_dir=out_dir,
                width=width,
                height=height
            )
