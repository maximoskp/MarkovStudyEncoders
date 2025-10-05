import matplotlib.pyplot as plt
import os
import numpy as np

def save_attention_maps(all_layers_attns, save_dir="attn_maps", prefix="layer"):
    """
    Save attention maps for all layers and all heads, plus an overlay per layer.

    Args:
        encoder: Transformer encoder with custom layers storing `last_attn_weights`
        save_dir: directory to save images
        prefix: prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    avg_all = None
    for i, attn in enumerate(all_layers_attns):
        if attn is None:
            print(f"⚠️ No attention stored in layer {i}. Did you run a forward pass?")
            continue

        B, H, L, _ = attn.shape

        # ---- Per-head plots ----
        for h in range(H):
            attn_map = attn[0, h].cpu().numpy()  # take first batch

            plt.figure(figsize=(6, 5))
            plt.imshow(attn_map, cmap="gray_r", aspect="auto")
            plt.colorbar()
            plt.title(f"Attention Map - Layer {i}, Head {h}")
            plt.xlabel("Key/Value positions")
            plt.ylabel("Query positions")

            # # boundary lines
            # plt.axvline(x=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)
            # plt.axhline(y=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)

            plt.tight_layout()
            fname = os.path.join(save_dir, f"{prefix}_L{i}_H{h}.png")
            plt.savefig(fname)
            plt.close()

        # ---- Overlay / aggregate plot ----
        # Option 1: average across heads
        attn_avg = attn[0].mean(dim=0).cpu().numpy()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_avg, cmap="gray_r", aspect="auto")
        plt.colorbar()
        plt.title(f"Attention Map - Layer {i} (Avg across {H} heads)")
        plt.xlabel("Key/Value positions")
        plt.ylabel("Query positions")

        # # boundary lines
        # plt.axvline(x=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)
        # plt.axhline(y=melody_len - 0.5, color="red", linestyle="--", linewidth=1.5)

        plt.tight_layout()
        fname = os.path.join(save_dir, f"{prefix}_L{i}_ALL.png")
        plt.savefig(fname)
        plt.close()

        if avg_all is None:
            avg_all = attn_avg
        else:
            avg_all += attn_avg
        
    plt.figure(figsize=(6, 5))
    plt.imshow(avg_all, cmap="gray_r", aspect="auto")
    # plt.colorbar()
    # plt.title(f"Attention Map - Average across all layers/heads")
    plt.xlabel("Key/Value positions")
    plt.ylabel("Query positions")
    plt.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_AVG_ALL.png")
    plt.savefig(fname)
    plt.close()

# end save_attention_maps

def save_attention_maps_with_split(all_layers_attns, melody_len, save_dir="attn_maps", prefix="layer"):
    """
    Save attention maps for all layers and all heads, plus an overlay per layer.

    Args:
        encoder: Transformer encoder with custom layers storing `last_attn_weights`
        save_dir: directory to save images
        prefix: prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    avg_all = None
    for i, attn in enumerate(all_layers_attns):
        if attn is None:
            print(f"⚠️ No attention stored in layer {i}. Did you run a forward pass?")
            continue

        B, H, L, _ = attn.shape
        
        # ---- Per-head plots ----
        for h in range(H):
            attn_map = attn[0, h].cpu().numpy()  # take first batch

            plt.figure(figsize=(6, 5))
            plt.imshow(attn_map, cmap="gray_r", aspect="auto")
            plt.colorbar()
            plt.title(f"Attention Map - Layer {i}, Head {h}")
            plt.xlabel("Key/Value positions")
            plt.ylabel("Query positions")

            # # boundary lines
            plt.axvline(x=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)
            plt.axhline(y=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)

            plt.tight_layout()
            fname = os.path.join(save_dir, f"{prefix}_L{i}_H{h}.png")
            plt.savefig(fname)
            plt.close()

        # ---- Overlay / aggregate plot ----
        # Option 1: average across heads
        attn_avg = attn[0].mean(dim=0).cpu().numpy()

        plt.figure(figsize=(6, 5))
        plt.imshow(attn_avg, cmap="gray_r", aspect="auto")
        plt.colorbar()
        plt.title(f"Attention Map - Layer {i} (Avg across {H} heads)")
        plt.xlabel("Key/Value positions")
        plt.ylabel("Query positions")

        # # boundary lines
        plt.axvline(x=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)
        plt.axhline(y=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)

        plt.tight_layout()
        fname = os.path.join(save_dir, f"{prefix}_L{i}_ALL.png")
        plt.savefig(fname)
        plt.close()

        if avg_all is None:
            avg_all = attn_avg
        else:
            avg_all += attn_avg

    plt.figure(figsize=(6, 5))
    plt.imshow(avg_all, cmap="gray_r", aspect="auto")
    # plt.colorbar()
    # plt.title(f"Attention Map - Average across all layers/heads")
    plt.xlabel("Key/Value positions", fontsize=18)
    plt.ylabel("Query positions", fontsize=18)

    # Set custom tick positions and labels for both axes
    tick_positions = [melody_len // 2, melody_len + melody_len // 2]
    tick_labels = ['melody', 'harmony']
    plt.xticks(tick_positions, tick_labels, fontsize=16)
    plt.yticks(tick_positions, tick_labels, rotation=90, fontsize=16)
    plt.tick_params(axis='both', length=0)  # Remove tick lines

    # # boundary lines
    plt.axvline(x=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)
    plt.axhline(y=melody_len - 0.5, color="black", linestyle="--", linewidth=1.5)

    plt.tight_layout()
    fname = os.path.join(save_dir, f"{prefix}_AVG_ALL.png")
    plt.savefig(fname, dpi=300)
    plt.close()

# end save_attention_maps_with_split