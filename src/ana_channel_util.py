#!/usr/bin/env python3
# analyze_enacc_utilization_by_block.py

import os
import csv
from collections import defaultdict
import pandas as pd

# Constants: 只需修改這兩個變數
BASE_DIR = "archive/json_0623_k4"
FIRST_ONLY = True

def parse_path(parts):
    layer = neuron = segment = container = None
    for p in parts:
        if p.startswith("layer_"):
            layer = int(p.split("_")[1])
        elif p.startswith("neuron_"):
            neuron = int(p.split("_")[1])
        elif p.startswith("segment_"):
            segment = int(p.split("_")[1])
        elif p.startswith("container_"):
            container = int(p.split("_")[1])
    return layer, neuron, segment, container

def read_dispatch_csv(path, first_only=False):
    z = o = ot = tot = 0
    last_dispatch = False
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("state") != "dispatch":
                last_dispatch = False
                continue
            if first_only and last_dispatch:
                continue
            last_dispatch = True
            for k, v in row.items():
                if not k.startswith("en_acc_"):
                    continue
                try:
                    val = int(v)
                except ValueError:
                    continue
                tot += 1
                if val == 0:
                    z += 1
                elif val == 1:
                    o += 1
                else:
                    ot += 1
    return z, o, ot, tot

def write_sheet_with_header(writer, sheet_name, df, header_lines):
    """
    將 df 寫入指定工作表，並在最上方加入 header_lines 說明文字。
    header_lines: list of str，每一項對應一列說明。
    """
    df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(header_lines))
    ws = writer.sheets[sheet_name]
    for i, line in enumerate(header_lines):
        ws.cell(row=i+1, column=1, value=line)

def main(base_dir: str, first_only: bool = False):
    block_cnt = defaultdict(lambda: [0,0,0,0])
    rows = []

    # 掃描資料夾
    for root, _, files in os.walk(base_dir):
        dispatch_f = next((f for f in files if f.endswith("dispatch_each_clk_log.csv")), None)
        if not dispatch_f:
            continue

        parts = os.path.relpath(root, base_dir).split(os.sep)
        layer, neuron, segment, container = parse_path(parts)
        if None in (layer, neuron, segment, container):
            continue

        z, o, ot, tot = read_dispatch_csv(os.path.join(root, dispatch_f), first_only)
        rz = z/tot if tot else 0.0
        r1 = o/tot if tot else 0.0
        ro = ot/tot if tot else 0.0

        # neuron-level row
        rows.append({
            "level": "neuron",
            "layer": layer,
            "neuron": neuron,
            "segment": segment,
            "container": container,
            "count_zero": z,
            "count_one": o,
            "count_other": ot,
            "total_samples": tot,
            "ratio_zero": rz,
            "ratio_one": r1,
            "ratio_other": ro
        })

        # accumulate block
        key = (layer, segment, container)
        cnt = block_cnt[key]
        cnt[0] += z
        cnt[1] += o
        cnt[2] += ot
        cnt[3] += tot

    # block-level "all"
    for (ly, seg, cont), (z, o, ot, tot) in block_cnt.items():
        rz = z/tot if tot else 0.0
        r1 = o/tot if tot else 0.0
        ro = ot/tot if tot else 0.0
        rows.append({
            "level": "all",
            "layer": ly,
            "neuron": "all",       # 顯示為 all 代表統整所有 neuron
            "segment": seg,
            "container": cont,
            "count_zero": z,
            "count_one": o,
            "count_other": ot,
            "total_samples": tot,
            "ratio_zero": rz,
            "ratio_one": r1,
            "ratio_other": ro
        })

    df = pd.DataFrame(rows)

    # 第一頁：all_config (neuron-level + block-level rows)，並排序
    df_all_config = (
        df
        .sort_values(
            ["layer", "segment", "container", "level", "neuron"],
            ascending=[True, True, True, True, True]
        )
        .reset_index(drop=True)
    )

    # 第二頁：all (只保留 block-level)，並排序
    df_all = (
        df_all_config[df_all_config.level == "all"]
        .drop(columns="level")
        .sort_values(["layer", "segment", "container"], ascending=[True, True, True])
        .reset_index(drop=True)
    )

    # 第三頁：summary (從 neuron-level 彙總後排序)
    df_neuron = df_all_config[df_all_config.level == "neuron"]
    df_summary = (
        df_neuron
        .groupby(["layer", "segment", "container"], as_index=False)
        .agg({
            "count_zero": "sum",
            "count_one": "sum",
            "count_other": "sum",
            "total_samples": "sum"
        })
        .sort_values(["layer", "segment", "container"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    df_summary["ratio_zero"]  = df_summary["count_zero"]  / df_summary["total_samples"]
    df_summary["ratio_one"]   = df_summary["count_one"]   / df_summary["total_samples"]
    df_summary["ratio_other"] = df_summary["count_other"] / df_summary["total_samples"]

    out_xlsx = os.path.join(base_dir, "enacc_utilization_by_block.xlsx")

    try:
        # 嘗試覆寫舊檔
        try:
            os.remove(out_xlsx)
        except OSError:
            pass

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            header_all_config = [
                f"Constants: BASE_DIR = {base_dir}",
                f"Constants: FIRST_ONLY = {first_only}",
                "Description: 原始 neuron-level 與 block-level(all) 資料，已依 layer/segment/container/level/neuron 排序"
            ]
            header_all = [
                f"Constants: BASE_DIR = {base_dir}",
                f"Constants: FIRST_ONLY = {first_only}",
                "Description: Block-level 資料 (level=all)，已依 layer/segment/container 排序"
            ]
            header_summary = [
                f"Constants: BASE_DIR = {base_dir}",
                f"Constants: FIRST_ONLY = {first_only}",
                "Description: Summary 資料，neuron-level 彙總至 block，已依 layer/segment/container 排序"
            ]

            write_sheet_with_header(writer, "all_config", df_all_config, header_all_config)
            write_sheet_with_header(writer, "all",        df_all,        header_all)
            write_sheet_with_header(writer, "summary",    df_summary,    header_summary)

        print(f"[OK] Written Excel → {out_xlsx}")

    except PermissionError:
        print("[ERROR] Cannot write Excel — maybe it's open? Falling back to CSV.")
        csv1 = os.path.join(base_dir, "all_config.csv")
        csv2 = os.path.join(base_dir, "all.csv")
        csv3 = os.path.join(base_dir, "summary.csv")
        df_all_config.to_csv(csv1, index=False)
        df_all       .to_csv(csv2, index=False)
        df_summary   .to_csv(csv3, index=False)
        print(f"[OK] Written CSVs → {csv1}, {csv2}, {csv3}")

    except (ImportError, ModuleNotFoundError):
        print("[WARN] openpyxl not found, wrote CSVs instead.")
        csv1 = os.path.join(base_dir, "all_config.csv")
        csv2 = os.path.join(base_dir, "all.csv")
        csv3 = os.path.join(base_dir, "summary.csv")
        df_all_config.to_csv(csv1, index=False)
        df_all       .to_csv(csv2, index=False)
        df_summary   .to_csv(csv3, index=False)
        print(f"[OK] Written CSVs → {csv1}, {csv2}, {csv3}")

if __name__ == "__main__":
    main(BASE_DIR, first_only=FIRST_ONLY)
