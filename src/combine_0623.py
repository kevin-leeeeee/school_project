
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dispatch_catch 狀態移除後只保留 dispatch 邏輯

Batch wrapper
=============

• 對指定 layer/neuron 掃描 Segments (4–12) × ContainerCount (4–12)。
• 每組 (S,C) 產生：
    - rearrange/en_acc 文檔（含權重索引對照 & quota 表）
    - cycle-by-cycle CSV
    - 3-D surface（含 / 不含投影）+ 多視角截圖
    - 2-D heat-map
    - TCL status 報告
• 若某組合的「非零權重種類」> container count，自動跳過。
• CLI 參數：
    --json / -j   權重 JSON 路徑（預設 analysis_output_0622/kmeans_4_quantized_network_prune.json）
    --out  / -o   輸出根資料夾（預設 json_0623_k4_group_prune）
"""

import os
import json
import math
import csv
import argparse
import traceback
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# >>>> 進度列／後援機制 ---------------------------------------------------------
try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False
    def tqdm(iterable=None, total=None, desc=""):
        print(f"[INFO] {desc} ...")
        for item in iterable:
            yield item
# ------------------------------------------------------------------------------

# ────────────────────────────── Config ──────────────────────────────
JSON_PATH_DEFAULT    = "archive/analysis_output_0622/kmeans_4_quantized_network_prune.json"
OUTPUT_BASE_DEFAULT  = "archive/json_0623_k4_group_prune"

SEG_MIN, SEG_MAX     = 4, 12
CONT_MIN, CONT_MAX   = 4, 12

USE_SEQUENTIAL_INPUT = False
EXCLUDE_ZERO_WEIGHT  = True
USE_AUTO_QUOTA       = True
MANUAL_QUOTA         = {0: 2, 1: 4, 2: 2}

FRACTIONAL_BITS      = 5
wbits = 8
LAYERS_TO_RUN        = [4]

TXT_REARRANGE        = "rearrange_index_enacc.txt"
TXT_ENACC_DEC        = "en_acc_group_view.txt"
TXT_IDX_MEM          = "channel_idx_mem.mif"
TXT_ENACC_MEM        = "en_acc_mem.mif"
OUTPUT_CSV           = "dispatch_each_clk_log.csv"
TCL_STATUS_FILE      = "status_report.tcl"

WEIGHT_IDX_MEM_W     = math.ceil(math.log2(SEG_MAX))
EN_ACC_MEM_W         = math.ceil(math.log2(CONT_MAX + 1))

# ──────────────────── Helper ────────────────────
def compute_quota(ids, cnt):
    """依頻率比例分配 quota，回傳 {wid: channels}"""
    if not ids:
        return {}
    freq = defaultdict(int)
    for wid in ids:
        freq[wid] += 1
    uniq = sorted(freq)
    if cnt <= len(uniq):
        return {wid: 1 for wid in uniq}
    total = sum(freq.values()) or 1
    remain = cnt - len(uniq)
    raw = [freq[wid] / total * remain for wid in uniq]
    base = [math.floor(x) for x in raw]
    extra = remain - sum(base)
    frac = [raw[i] - base[i] for i in range(len(uniq))]
    for _ in range(extra):
        idx = max(range(len(uniq)), key=lambda j: frac[j])
        base[idx] += 1
        frac[idx] = 0
    return {uniq[i]: base[i] + 1 for i in range(len(uniq))}


def twos_bits(val, width):
    return format(val & ((1 << width) - 1), f"0{width}b")

# ──────────────────── Core ────────────────────
def run_single(base_dir, model, layer_idx, neuron_idx, n_seg, cont_cnt):
    key = f"network.{layer_idx}.weight"
    flat_raw = np.array(model[key])[neuron_idx].flatten()
    if any(v is None for v in flat_raw):
        print(f"[WARNING] Found None in weights: layer={layer_idx}, neuron={neuron_idx}")
    flat = np.array([0.0 if v is None else v for v in flat_raw])
    vals = sorted(set(flat.tolist()))

    v2i = {v: i for i, v in enumerate(vals)}
    ids = [v2i[v] for v in flat]
    zero_id = v2i.get(0.0)
    if zero_id is None:
        zero_id = max(v2i.values()) + 1
        v2i[0.0] = zero_id

    # 篩除 zero weight
    nnz_ids = [wid for wid in ids if not (EXCLUDE_ZERO_WEIGHT and wid == zero_id)]
    if len(set(nnz_ids)) > cont_cnt:
        return None, len(set(nnz_ids))

    # 計算 quota
    quota = compute_quota(nnz_ids, cont_cnt) if USE_AUTO_QUOTA else MANUAL_QUOTA.copy()
    if zero_id is not None and EXCLUDE_ZERO_WEIGHT:
        quota[zero_id] = 0

    # channel map
    cmap, ch_base = {}, 0
    for wid, q in sorted(quota.items()):
        cmap[wid] = list(range(ch_base, ch_base + q))
        ch_base += q

    prefix = f"{layer_idx}{neuron_idx}{n_seg}{cont_cnt}_"
    out_dir = os.path.join(base_dir, f"segment_{n_seg}", f"container_{cont_cnt}")
    os.makedirs(out_dir, exist_ok=True)

    # --- header & rearrange ---
    header_path = os.path.join(out_dir, prefix + TXT_REARRANGE)
    with open(header_path, "w", encoding="utf-8") as fh:
        fh.write(f"# NEURON_INDEX = {neuron_idx}\n")
        fh.write(f"# FRACTIONAL_BITS = {FRACTIONAL_BITS}\n")
        fh.write(f"# weight_idx_mem_width = {WEIGHT_IDX_MEM_W}\n")
        fh.write(f"# en_acc_mem_width     = {EN_ACC_MEM_W}\n\n")
        fh.write("=== Index→WeightValue & Fixed-Point Mapping ===\n")
        for v in vals:
            fixed = int(round(v * (1 << FRACTIONAL_BITS)))
            fh.write(f"Index {v2i[v]}: {v:.6f} → {twos_bits(fixed, wbits)}\n")
        fh.write("\n=== Channel Quotas ({'Auto' if USE_AUTO_QUOTA else 'Manual'}) ===\n")
        fh.write("WID | Freq |  %  | Quota | Channels\n")
        fh.write("----|------|-----|-------|---------\n")
        freq = defaultdict(int)
        for wid in nnz_ids:
            freq[wid] += 1
        total_nnz = len(nnz_ids) or 1
        for wid in sorted(quota):
            cnt = freq.get(wid, 0)
            pct_str = f"{cnt / total_nnz * 100:>3.0f}%"
            fh.write(f"{wid:>3} | {cnt:>4} | {pct_str} | {quota[wid]:>5} | {cmap[wid]}\n")
        fh.write("\n")

    # --- each channel weight binary ---
    each_file = os.path.join(out_dir, prefix + "each_channel_weight.txt")
    with open(each_file, "w", encoding="utf-8") as fh2:
        channel_vals = [None] * cont_cnt
        for wid, channels in cmap.items():
            for ch in channels:
                channel_vals[ch] = vals[wid]
        for v in channel_vals:
            val = 0.0 if v is None else v
            fixed = int(round(val * (1 << FRACTIONAL_BITS)))
            fh2.write(f"{twos_bits(fixed, wbits)}\n")

    # --- human-readable en_acc groups & memory dumps ---
    pad = (-len(ids)) % n_seg
    ids_pad = ids + [zero_id] * pad
    if USE_SEQUENTIAL_INPUT:
        total_groups = math.ceil(len(ids_pad) / n_seg)
        get_batch = lambda g: ids_pad[g * n_seg:(g + 1) * n_seg]
    else:
        seg_sz = len(ids_pad) // n_seg
        segs = [ids_pad[i * seg_sz:(i + 1) * seg_sz] for i in range(n_seg)]
        total_groups = seg_sz
        get_batch = lambda g: [segs[s][g] for s in range(n_seg)]

    all_after, all_en = [], []
    with open(os.path.join(out_dir, prefix + TXT_ENACC_DEC), "w", encoding="utf-8") as fh:
        for g in range(total_groups):
            batch = get_batch(g)
            seen, aft, en0 = defaultdict(int), [], []
            for wid in batch:
                seen[wid] += 1
                chans = cmap.get(wid, [])
                after_val = chans[(seen[wid] - 1) % len(chans)] if chans else wid
                if wid == zero_id and EXCLUDE_ZERO_WEIGHT:
                    en_val = 0
                elif seen[wid] <= len(chans):
                    en_val = 1
                else:
                    en_val = seen[wid] - len(chans) + 1
                aft.append(after_val)
                en0.append(en_val)
            all_after.extend(aft)
            all_en.extend(en0)
            fh.write(f"Group {g}: before={batch} after={aft} en0={en0}\n")

    with open(os.path.join(out_dir, prefix + TXT_IDX_MEM), "w", encoding="utf-8") as fh:
        for v in all_after:
            fh.write(f"{v:0{WEIGHT_IDX_MEM_W}b}\n")
    with open(os.path.join(out_dir, prefix + TXT_ENACC_MEM), "w", encoding="utf-8") as fh:
        for v in all_en:
            fh.write(f"{v:0{EN_ACC_MEM_W}b}\n")

    # --- cycle log (dispatch only) ---
    clk = 0
    current_group = 0
    en_acc = np.zeros(n_seg, dtype=int)
    cycle_log = []
    while True:
        if current_group >= total_groups and np.all(en_acc <= 1):
            break
        entry = {"cycle": clk, "current_group": current_group}
        for i in range(n_seg):
            entry[f"en_acc_{i}"] = int(en_acc[i])
        cycle_log.append(entry)

        if np.all(en_acc <= 1) and current_group < total_groups:
            batch = get_batch(current_group)
            current_group += 1
            cnt_tmp = defaultdict(int)
            load = np.zeros(n_seg, dtype=int)
            for i, wid in enumerate(batch):
                cnt_tmp[wid] += 1
                chans = cmap.get(wid, [])
                if wid == zero_id and EXCLUDE_ZERO_WEIGHT:
                    l = 0
                elif cnt_tmp[wid] <= len(chans):
                    l = 1
                else:
                    l = cnt_tmp[wid] - len(chans) + 1
                load[i] = l
            en_acc = np.maximum(en_acc, load)
        else:
            en_acc = np.maximum(en_acc - 1, 0)

        clk += 1

    if cycle_log:
        csv_path = os.path.join(out_dir, prefix + OUTPUT_CSV)
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(cycle_log[0].keys()))
            writer.writeheader()
            writer.writerows(cycle_log)

    return clk, None

# ──────────────────── Plotting ────────────────────
def plot_and_save(base_dir, summary, layer_idx, neuron_idx):
    s_vals = list(range(SEG_MIN, SEG_MAX + 1))
    c_vals = list(range(CONT_MIN, CONT_MAX + 1))
    xf, yf = np.meshgrid(s_vals, c_vals)
    zf = np.array([[summary[s][c] for s in s_vals] for c in c_vals], dtype=float)
    mask = ~np.isnan(zf)
    title = f"Layer {layer_idx} | Neuron {neuron_idx}"

    def draw(ax, proj):
        surf = ax.plot_surface(xf, yf, zf, cmap="viridis", edgecolor="none", alpha=0.8)
        ax.plot_wireframe(xf, yf, zf, linewidth=0.5, alpha=0.3)
        if proj:
            zmin, _ = ax.get_zlim()
            ax.contourf(xf, yf, zf, zdir="z", offset=zmin, cmap="viridis", alpha=0.7)
        if mask.any():
            idxm = np.nanargmin(zf)
            xi, yi, zi = xf.flatten()[idxm], yf.flatten()[idxm], zf.flatten()[idxm]
            ax.scatter([xi], [yi], [zi], color="red", s=50)
            ax.text(xi, yi, zi + 0.02 * (ax.get_zlim()[1] - ax.get_zlim()[0]), f"Min={zi:.0f}", color="red")
        ax.set_xlabel("Segments")
        ax.set_ylabel("Container Count")
        ax.set_zlabel("Total Cycles")
        return surf

    pic_dir = os.path.join(base_dir, "picture")
    os.makedirs(pic_dir, exist_ok=True)

    for proj, tag in [(False, "no_proj"), (True, "proj")]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = draw(ax, proj)
        ax.set_title(title, pad=20)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        fig.tight_layout()
        fig.savefig(os.path.join(pic_dir, f"3d_surface_{tag}.png"), dpi=300)
        plt.close(fig)

        for elev, azim in [(30, 45), (30, 135), (60, 45), (60, 135)]:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            surf = draw(ax, proj)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{title} (E{elev}/A{azim})", pad=20)
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
            fig.tight_layout()
            fig.savefig(os.path.join(pic_dir, f"3d_e{elev}_a{azim}_{tag}.png"), dpi=300)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    im = ax.imshow(zf, origin="lower", extent=[min(s_vals)-0.5, max(s_vals)+0.5,
                min(c_vals)-0.5, max(c_vals)+0.5], aspect="auto", cmap="viridis")
    for i, s in enumerate(s_vals):
        for j, c in enumerate(c_vals):
            ax.text(s, c, f"{summary[s][c]:.0f}", ha="center", va="center", fontsize=6, color="white")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Total Cycles")
    ax.set_xlabel("Segments")
    ax.set_ylabel("Container Count")
    ax.set_title(f"Cycle Heat-Map | {title}", pad=10)
    ax.set_xticks(s_vals)
    ax.set_yticks(c_vals)
    ax.set_xlim(min(s_vals)-0.5, max(s_vals)+0.5)
    ax.set_ylim(min(c_vals)-0.5, max(c_vals)+0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(pic_dir, "heatmap_cycles.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

# ──────────────────── main ────────────────────
def main():
    parser = argparse.ArgumentParser(description="Segment × Container batch sweep")
    parser.add_argument("-j", "--json", default=JSON_PATH_DEFAULT,
                        help="Path to weight JSON (default: analysis_output_0622/kmeans_4_quantized_network_prune.json)")
    parser.add_argument("-o", "--out", default=OUTPUT_BASE_DEFAULT,
                        help="Output root folder (default: json_0623_k4_group_prune)")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    with open(args.json, "r", encoding="utf-8") as fh:
        model = json.load(fh)

    layers_all = sorted(
        int(k.split(".")[1])
        for k in model
        if k.startswith("network.") and k.endswith(".weight")
    )

    if LAYERS_TO_RUN is None:
        layers = layers_all
    elif isinstance(LAYERS_TO_RUN, list):
        layers = [l for l in layers_all if l in LAYERS_TO_RUN]
    else:
        raise ValueError("LAYERS_TO_RUN must be None or list")

    total_tasks = 0
    for L in layers:
        weights_shape = np.array(model[f"network.{L}.weight"]).shape
        total_tasks += weights_shape[0] * (SEG_MAX - SEG_MIN + 1) * (CONT_MAX - CONT_MIN + 1)

    pbar = tqdm(total=total_tasks, desc="Overall Progress") if _tqdm_available else None

    for L in (tqdm(layers, desc="Layers") if _tqdm_available else layers):
        weights = np.array(model[f"network.{L}.weight"])
        n_neurons = weights.shape[0]
        neuron_iter = tqdm(range(n_neurons), desc=f"Layer {L} Neurons", leave=False) if _tqdm_available else range(n_neurons)

        for N in neuron_iter:
            base_dir = os.path.join(args.out, f"layer_{L}", f"neuron_{N}")
            summary, status = {}, {}
            for S in range(SEG_MIN, SEG_MAX + 1):
                summary[S], status[S] = {}, {}
                for C in range(CONT_MIN, CONT_MAX + 1):
                    try:
                        cycles, skip_cause = run_single(base_dir,
                                                         model, L, N, S, C)
                        if cycles is None:
                            summary[S][C] = float("nan")
                            status[S][C] = f"SKIP (unique_NNZ={skip_cause} > container={C})"
                        else:
                            summary[S][C] = cycles
                            status[S][C] = "SUCCESS"
                    except Exception:
                        summary[S][C] = float("nan")
                        status[S][C] = "FAIL"
                        traceback.print_exc()
                    if pbar:
                        pbar.update(1)

            os.makedirs(base_dir, exist_ok=True)
            # status report
            with open(os.path.join(base_dir, TCL_STATUS_FILE), "w", encoding="utf-8") as tf:
                tf.write(f"# Layer {L}, Neuron {N} status\n")
                for S in summary:
                    for C in summary[S]:
                        prefix_dir = f"segment_{S}/container_{C}"
                        tf.write(f'puts "S={S},C={C}: {status[S][C]} (dir: {prefix_dir})"\n')

            # cycle summary
            with open(os.path.join(base_dir, "cycle_summary.json"),
                      "w", encoding="utf-8") as jf:
                json.dump(summary, jf, indent=2)

            plot_and_save(base_dir, summary, L, N)

    if pbar:
        pbar.close()
        print("[DONE] 全部組合模擬完成。")

if __name__ == "__main__":
    main()

