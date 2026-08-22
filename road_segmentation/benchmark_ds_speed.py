#!/usr/bin/env python3
"""
Speed Benchmark for Depthwise-Separable BiKA (BiKASegNet-DS).
Runs purely on speed (no dataset / checkpoint required).
Supports CPU (with configurable threads) and GPU (CUDA).
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/mnt/storage/M2_internship/TwinLiteNet")

from bika.BiKA_DSConv import BiKASegNet_DS, BiKA_DWConv2d, BiKA_DSConvBlock
from bika.segmentation import BiKASegNet


def parse_args():
    p = argparse.ArgumentParser(description="Speed Benchmark for BiKASegNet-DS vs Dense vs TwinLiteNet")
    p.add_argument("--base_channels", type=int, default=16, choices=[16, 32],
                   help="Base channels (16 or 32)")
    p.add_argument("--input_h", type=int, default=192, help="Input height")
    p.add_argument("--input_w", type=int, default=256, help="Input width")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                   help="Execution device")
    p.add_argument("--threads", type=int, default=12, help="CPU OpenMP / PyTorch threads")
    p.add_argument("--warmup", type=int, default=30, help="Warmup iterations")
    p.add_argument("--iterations", type=int, default=200, help="Timed benchmark iterations")
    p.add_argument("--profile_layers", action="store_true", help="Profile layer-by-layer breakdown")
    return p.parse_args()


def benchmark_model(model, dummy_input, warmup, iterations, device):
    model.eval()
    is_cuda = (device.type == "cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
        if is_cuda:
            torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    l = np.array(latencies)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    mean_ms = float(np.mean(l))
    median_ms = float(np.median(l))
    min_ms = float(np.min(l))
    p95_ms = float(np.percentile(l, 95))
    fps = 1000.0 / mean_ms

    return {
        "params_m": params_m,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "p95_ms": p95_ms,
        "fps": fps,
    }


def profile_ds_layers(model, x, device, runs=50):
    model.eval()
    is_cuda = (device.type == "cuda")

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        if is_cuda:
            torch.cuda.synchronize()

    totals = {}
    with torch.no_grad():
        for _ in range(runs):
            if is_cuda: torch.cuda.synchronize()
            t0 = time.perf_counter()
            t1 = model.enc1(x)
            if is_cuda: torch.cuda.synchronize()
            t1_time = time.perf_counter()

            p1 = model.pool(t1)
            t2 = model.enc2(p1)
            if is_cuda: torch.cuda.synchronize()
            t2_time = time.perf_counter()

            p2 = model.pool(t2)
            t3 = model.enc3(p2)
            if is_cuda: torch.cuda.synchronize()
            t3_time = time.perf_counter()

            p3 = model.pool(t3)
            bn = model.bottleneck(p3)
            if is_cuda: torch.cuda.synchronize()
            bn_time = time.perf_counter()

            u3 = F.interpolate(bn, size=t3.shape[2:], mode="bilinear", align_corners=False)
            c3 = torch.cat([u3, t3], dim=1)
            d3 = model.dec3(c3)
            if is_cuda: torch.cuda.synchronize()
            d3_time = time.perf_counter()

            u2 = F.interpolate(d3, size=t2.shape[2:], mode="bilinear", align_corners=False)
            c2 = torch.cat([u2, t2], dim=1)
            d2 = model.dec2(c2)
            if is_cuda: torch.cuda.synchronize()
            d2_time = time.perf_counter()

            u1 = F.interpolate(d2, size=t1.shape[2:], mode="bilinear", align_corners=False)
            c1 = torch.cat([u1, t1], dim=1)
            d1 = model.dec1(c1)
            if is_cuda: torch.cuda.synchronize()
            d1_time = time.perf_counter()

            out = model.final(d1)
            if is_cuda: torch.cuda.synchronize()
            fin_time = time.perf_counter()

            totals["enc1 (Stem + DW)"] = totals.get("enc1 (Stem + DW)", 0) + (t1_time - t0) * 1000
            totals["enc2 (DW + PW)"] = totals.get("enc2 (DW + PW)", 0) + (t2_time - t1_time) * 1000
            totals["enc3 (DW + PW)"] = totals.get("enc3 (DW + PW)", 0) + (t3_time - t2_time) * 1000
            totals["bottleneck (DW + PW)"] = totals.get("bottleneck (DW + PW)", 0) + (bn_time - t3_time) * 1000
            totals["dec3 (Interp + DW + PW)"] = totals.get("dec3 (Interp + DW + PW)", 0) + (d3_time - bn_time) * 1000
            totals["dec2 (Interp + DW + PW)"] = totals.get("dec2 (Interp + DW + PW)", 0) + (d2_time - d3_time) * 1000
            totals["dec1 (Interp + DW + PW)"] = totals.get("dec1 (Interp + DW + PW)", 0) + (d1_time - d2_time) * 1000
            totals["final (1x1 Head)"] = totals.get("final (1x1 Head)", 0) + (fin_time - d1_time) * 1000

    tot_ms = sum(totals.values()) / runs
    print("\n  ── Layer-by-Layer Latency Breakdown ──")
    for k, v in totals.items():
        avg = v / runs
        pct = (avg / tot_ms) * 100.0
        bar = "█" * int(pct / 4)
        print(f"  {k:30s} : {avg:6.2f} ms ({pct:5.1f}%) | {bar}")
    print(f"  {'Total forward pass':30s} : {tot_ms:6.2f} ms ({1000.0/tot_ms:5.1f} FPS)\n")


def main():
    args = parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    device = torch.device(args.device)
    print("=" * 80)
    print(f"  BiKASegNet-DS (Depthwise-Separable) Speed Benchmark")
    print(f"  Device     : {args.device.upper()} (Threads: {args.threads if args.device == 'cpu' else 'N/A'})")
    print(f"  Resolution : {args.input_h} × {args.input_w}")
    print(f"  Warmup / It: {args.warmup} / {args.iterations}")
    print("=" * 80)

    dummy_input = torch.randn(1, 3, args.input_h, args.input_w, device=device)

    # 1. BiKASegNet-DS (Current Target)
    print(f"\n[1/3] Benchmarking BiKASegNet-DS ({args.base_channels} base channels)...")
    ds_model = BiKASegNet_DS(num_classes=2, base_channels=args.base_channels).to(device)
    ds_res = benchmark_model(ds_model, dummy_input, args.warmup, args.iterations, device)

    # 2. BiKASegNet-Dense (Baseline for comparison)
    print(f"[2/3] Benchmarking BiKASegNet Dense ({args.base_channels} base channels)...")
    dense_model = BiKASegNet(num_classes=2, base_channels=args.base_channels, legacy_mode=False).to(device)
    if args.device == "cpu":
        for m in dense_model.modules():
            if m.__class__.__name__ == "BiKA_Conv2d":
                m.pack_weights(legacy_mode=False)
    dense_res = benchmark_model(dense_model, dummy_input, args.warmup, args.iterations, device)

    # 3. TwinLiteNet (Comparison)
    print(f"[3/3] Benchmarking TwinLiteNet...")
    try:
        from model.TwinLite import TwinLiteNet
        tl_model = TwinLiteNet().to(device)
        tl_res = benchmark_model(tl_model, dummy_input, args.warmup, args.iterations, device)
        has_tl = True
    except Exception as e:
        has_tl = False
        print(f"  TwinLiteNet load skipped ({e})")

    # Summary table
    print("\n" + "=" * 80)
    print(f"  RESULTS SUMMARY @ {args.input_h}×{args.input_w} ({args.device.upper()})")
    print("=" * 80)
    print(f"  {'Model':35s} | {'Params':>7s} | {'Mean (ms)':>10s} | {'Median':>8s} | {'Min':>8s} | {'FPS':>8s}")
    print("  " + "-" * 88)
    print(f"  {'BiKASegNet-DS ' + str(args.base_channels) + 'ch (Ours)':35s} | {ds_res['params_m']:6.3f}M | {ds_res['mean_ms']:10.2f} | {ds_res['median_ms']:8.2f} | {ds_res['min_ms']:8.2f} | {ds_res['fps']:8.2f}")
    print(f"  {'BiKASegNet-Dense ' + str(args.base_channels) + 'ch':35s} | {dense_res['params_m']:6.3f}M | {dense_res['mean_ms']:10.2f} | {dense_res['median_ms']:8.2f} | {dense_res['min_ms']:8.2f} | {dense_res['fps']:8.2f}")
    if has_tl:
        print(f"  {'TwinLiteNet [48]':35s} | {tl_res['params_m']:6.3f}M | {tl_res['mean_ms']:10.2f} | {tl_res['median_ms']:8.2f} | {tl_res['min_ms']:8.2f} | {tl_res['fps']:8.2f}")
    print("=" * 80)

    # Layer breakdown if requested
    if args.profile_layers:
        profile_ds_layers(ds_model, dummy_input, device)


if __name__ == "__main__":
    main()
