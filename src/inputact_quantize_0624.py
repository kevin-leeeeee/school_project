#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantize_sequential.py

Convert MNIST JSON → A-bit signed fixed-point binary output.
Generate tags.txt recording each sample’s label.

支援在每筆 sample 的量化結果內，按 groups 把 784 個值分段並交錯輸出。

用法：
    python quantize_sequential.py \
      -j mnist_dataset.json \
      -o quantized_input_act \
      -s 0 -e None \
      -f 6 -t 8 \
      -g 8
"""

import os
import sys
import math
import argparse
import ijson
from ijson.common import IncompleteJSONError

# ------------- DEFAULTS ------------- #
JSON_PATH    = "analysis_output_0622/MnistDataset.json"   # path to your MNIST JSON file
TOTAL_BITS   = 8                      # total bit-width A
FRAC_BITS    = 6                      # fractional bit-width B
GROUPS       = 8                      # >1: 在單一 sample 裡分組並交錯
START_INDEX  = 0                      # start sample index (inclusive)
END_INDEX    = 19                   # end sample index (inclusive), None=last
OUT_ROOT     = "archive/quantized_input_act_0624"  # outer output folder
# ------------------------------------ #

def int_to_bin_str(value: int, total_bits: int) -> str:
    mask = (1 << total_bits) - 1
    return format(value & mask, f'0{total_bits}b')

def quantize_sample(sample_data, frac_bits, total_bits):
    scale = 1 << frac_bits
    min_q = -(1 << (total_bits - 1))
    max_q =  (1 << (total_bits - 1)) - 1
    flat = []
    for row in sample_data:
        for pix in row:
            q = int(round(pix * scale))
            q = max(min_q, min(max_q, q))
            flat.append(q)
    return flat

def interleave_data(flat_list, groups):
    """
    把 flat_list (長度 784) 分成 groups 段，pad 0 到等長，
    然後以 round-robin 方式交錯輸出。
    """
    total = len(flat_list)
    seg = math.ceil(total / groups)
    # pad 0
    segments = []
    for i in range(groups):
        start = i * seg
        chunk = flat_list[start : start + seg]
        chunk += [0] * (seg - len(chunk))
        segments.append(chunk)

    interleaved = []
    for idx in range(seg):
        for chunk in segments:
            interleaved.append(chunk[idx])
    return interleaved

def process(json_path, start_idx, end_idx, frac_bits, total_bits, out_dir, groups):
    """
    單一流程：streaming 處理每筆 sample，
    若 groups>1，就在每筆 sample 裡做分組交錯。
    """
    os.makedirs(out_dir, exist_ok=True)
    tags_file = os.path.join(out_dir, "tags.txt")

    with open(json_path, "rb") as f, open(tags_file, "w", encoding="utf-8") as tag_f:
        try:
            for idx, sample in enumerate(ijson.items(f, "samples.item")):
                if idx < start_idx:
                    continue
                if end_idx is not None and idx > end_idx:
                    break

                img   = sample["data"][0]           # 28×28 list
                label = sample.get("label", None)

                flat = quantize_sample(img, frac_bits, total_bits)
                if groups > 1:
                    flat = interleave_data(flat, groups)

                # 轉為二進位字串
                lines = [int_to_bin_str(v, total_bits) for v in flat]

                fn   = f"test_data_q{total_bits}f{frac_bits}_{idx:04d}.txt"
                path = os.path.join(out_dir, fn)
                with open(path, "w", encoding="utf-8") as fw:
                    fw.write("\n".join(lines))

                tag_f.write(f"{label}\n")
                print(f"[{idx}] → {path}, label={label}")

        except IncompleteJSONError:
            # 若 JSON 提早 EOF，也只是警告後結束迴圈
            print(f"Warning: JSON truncated at sample {idx}; stopping.", file=sys.stderr)

    print(f"\nAll labels written to: {tags_file}")

def main():
    p = argparse.ArgumentParser(description="MNIST JSON → fixed-point tool")
    p.add_argument("-j","--json", default=JSON_PATH, help="MNIST JSON path")
    p.add_argument("-o","--out",  default=OUT_ROOT, help="output root dir")
    p.add_argument("-s","--start",type=int, default=START_INDEX, help="start idx incl.")
    p.add_argument("-e","--end",  type=int, default=END_INDEX,   help="end idx incl.")
    p.add_argument("-f","--frac", type=int, default=FRAC_BITS,   help="fraction bits")
    p.add_argument("-t","--total",type=int, default=TOTAL_BITS,  help="total bits")
    p.add_argument("-g","--groups",type=int, default=GROUPS,
                   help=">1: group within each sample and interleave")

    args = p.parse_args()

    mode = f"f{args.frac}_serial" if args.groups == 1 else f"f{args.frac}_group{args.groups}"
    out_dir = os.path.join(args.out, mode)

    process(
        args.json, args.start, args.end,
        args.frac, args.total, out_dir, args.groups
    )

if __name__ == "__main__":
    main()
