"""Unified IoU + Speed benchmark for BiKASegNet (and competitor models).

Evaluates:
  1. Per-class IoU, mIoU, Dice on BDD100K val set
  2. Inference latency (ms) and FPS on CPU or CUDA

Usage (x86 PC with CUDA):
    python benchmark_speed_iou.py \
        --checkpoint outputs/BiKASegNet_640x360_ft/model_best.pth \
        --label_grouping lane_fg --input_h 360 --input_w 640 \
        --device cuda --warmup 50 --iterations 500

Usage (x86 PC, CPU only — simulates edge deployment):
    python benchmark_speed_iou.py \
        --checkpoint outputs/BiKASegNet_640x360_ft/model_best.pth \
        --label_grouping lane_fg --input_h 360 --input_w 640 \
        --device cpu --warmup 20 --iterations 200

Usage (Raspberry Pi 4B — copy this script + model to Pi):
    python benchmark_speed_iou.py \
        --checkpoint model_best.pth \
        --label_grouping lane_fg --input_h 360 --input_w 640 \
        --device cpu --warmup 10 --iterations 100 \
        --batch_size 1 --num_workers 2 \
        --skip_iou  # (skip IoU if BDD100K data isn't on the Pi)
"""
import argparse
import math
import os
import time
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
from data import (
    BDD100KDataset, BDD100K_NUM_CLASSES, BDD100K_CLASSES, LABEL_GROUPINGS,
)
from models import BiKASegNet
from training.metrics import SegmentationMetric


def parse_args():
    p = argparse.ArgumentParser(
        description="BiKASegNet: IoU accuracy + inference speed benchmark"
    )
    # ── Model ──
    p.add_argument("--checkpoint", required=True,
                   help="Path to model_best.pth or checkpoint_best.pth")
    p.add_argument("--base_channels", default=16, type=int)
    p.add_argument("--label_grouping", default="lane_fg",
                   choices=["none"] + list(LABEL_GROUPINGS))

    # ── Data ──
    p.add_argument("--bdd100k_base",
                   default="/storage/student11/bdd100k_seg/bdd100k/seg")
    p.add_argument("--input_h", default=360, type=int)
    p.add_argument("--input_w", default=640, type=int)
    p.add_argument("--batch_size", default=1, type=int,
                   help="Use batch_size=1 for latency measurement (standard)")
    p.add_argument("--num_workers", default=4, type=int)

    # ── Speed benchmark ──
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="'cpu' for x86/RPi benchmarks, 'cuda' for GPU")
    p.add_argument("--warmup", default=50, type=int,
                   help="Warmup iterations (not timed)")
    p.add_argument("--iterations", default=300, type=int,
                   help="Timed iterations for FPS measurement")

    # ── Flags ──
    p.add_argument("--skip_iou", action="store_true", help="Skip IoU evaluation")
    p.add_argument("--skip_speed", action="store_true", help="Skip speed benchmark")
    p.add_argument("--use_torchscript", action="store_true",
                   help="JIT-trace the model before benchmarking (like TwinLiteNet)")

    return p.parse_args()


def load_model(args):
    """Load BiKASegNet with the correct num_classes."""
    if args.label_grouping != "none":
        g = LABEL_GROUPINGS[args.label_grouping]
        num_classes = g["num_classes"]
    else:
        num_classes = BDD100K_NUM_CLASSES

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    
    # Auto-detect full_precision_head and full_precision_stem
    fp_head = True
    fp_stem = True
    if "final.bias" in state and len(state["final.bias"].shape) == 4:
        fp_head = False
    
    # Map old BiKABlock keys (from sem_seg_bin) to new keys
    new_state = {}
    for k, v in state.items():
        new_k = k
        new_k = new_k.replace(".block.0.", ".conv1.")
        new_k = new_k.replace(".block.1.", ".bn1.")
        new_k = new_k.replace(".block.3.", ".conv2.")
        new_k = new_k.replace(".block.4.", ".bn2.")
        new_state[new_k] = v
        
    if "enc1.conv1.weight" in new_state:
        fp_stem = False
        
    legacy_mode = True
    for k in new_state.keys():
        if "shift_pre" in k or "prelu" in k:
            legacy_mode = False
            break

    model = BiKASegNet(
        num_classes=num_classes,
        base_channels=args.base_channels,
        legacy_mode=legacy_mode,
        full_precision_head=fp_head,
        full_precision_stem=fp_stem,
    )

    model.load_state_dict(new_state, strict=False)
    model.eval()

    if legacy_mode:
        print("  Fusing BatchNorm and ReLU into BiKA_Conv2d CUDA kernels...")
        fuse_bika_model(model)
    else:
        print("  Skipping fusion: RPReLU (legacy_mode=False) cannot be fused into current CUDA kernel.")
        
    return model, num_classes


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fuse_bika_model(model: nn.Module):
    """
    Operator Fusion: Fuses BatchNorm and RPReLU into BiKA_Conv2d.
    """
    for name, module in model.named_modules():
        # Handle original Sequential blocks if any
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 2):
                m_conv = module[i]
                m_bn = module[i+1]
                m_relu = module[i+2]
                
                if m_conv.__class__.__name__ == "BiKA_Conv2d" and isinstance(m_bn, nn.BatchNorm2d):
                    std = torch.sqrt(m_bn.running_var + m_bn.eps)
                    scale = m_bn.weight / std
                    shift = m_bn.bias - m_bn.running_mean * scale
                    
                    if hasattr(m_relu, 'shift_pre'):
                        shift = shift + m_relu.shift_pre.squeeze()
                        scale = scale * m_relu.prelu.weight.squeeze()
                        shift = shift * m_relu.prelu.weight.squeeze() + m_relu.shift_post.squeeze()
                    
                    if hasattr(m_conv, 'output_scale') and m_conv.output_scale is not None:
                        scale = scale * m_conv.output_scale.squeeze()
                        m_conv.output_scale = None
                        
                    m_conv.register_buffer("out_scale", scale.detach().clone())
                    m_conv.register_buffer("out_shift", shift.detach().clone())
                    m_conv.do_relu = True
                    m_conv.pack_weights()
                    
                    module[i+1] = nn.Identity()
                    module[i+2] = nn.Identity()

                elif isinstance(m_conv, nn.Conv2d) and isinstance(m_bn, nn.BatchNorm2d):
                    # FP32 Conv2d + BatchNorm2d fusion
                    std = torch.sqrt(m_bn.running_var + m_bn.eps)
                    w = m_conv.weight * (m_bn.weight / std).view(-1, 1, 1, 1)
                    if m_conv.bias is not None:
                        b = (m_conv.bias - m_bn.running_mean) * (m_bn.weight / std) + m_bn.bias
                    else:
                        b = -m_bn.running_mean * (m_bn.weight / std) + m_bn.bias
                    m_conv.weight = nn.Parameter(w)
                    m_conv.bias = nn.Parameter(b)
                    module[i+1] = nn.Identity()

        # Handle the new BiKAConvBlock
        elif module.__class__.__name__ == "BiKAConvBlock":
            # Fuse conv1 -> bn1 -> act1
            if hasattr(module, 'conv1') and hasattr(module, 'bn1') and hasattr(module, 'act1'):
                m_conv = module.conv1
                m_bn = module.bn1
                m_relu = module.act1
                if m_conv.__class__.__name__ == "BiKA_Conv2d" and isinstance(m_bn, nn.BatchNorm2d):
                    std = torch.sqrt(m_bn.running_var + m_bn.eps)
                    scale = m_bn.weight / std
                    shift = m_bn.bias - m_bn.running_mean * scale
                    
                    if hasattr(m_relu, 'shift_pre'):
                        shift = shift + m_relu.shift_pre.squeeze()
                        scale = scale * m_relu.prelu.weight.squeeze()
                        shift = shift * m_relu.prelu.weight.squeeze() + m_relu.shift_post.squeeze()
                        
                    if hasattr(m_conv, 'output_scale') and m_conv.output_scale is not None:
                        scale = scale * m_conv.output_scale.squeeze()
                        m_conv.output_scale = None
                        
                    m_conv.register_buffer("out_scale", scale.detach().clone())
                    m_conv.register_buffer("out_shift", shift.detach().clone())
                    m_conv.do_relu = True
                    m_conv.pack_weights()
                    
                    module.bn1 = nn.Identity()
                    module.act1 = nn.Identity()

            # Fuse conv2 -> bn2 -> act2
            if hasattr(module, 'conv2') and hasattr(module, 'bn2') and hasattr(module, 'act2'):
                m_conv = module.conv2
                m_bn = module.bn2
                m_relu = module.act2
                if m_conv.__class__.__name__ == "BiKA_Conv2d" and isinstance(m_bn, nn.BatchNorm2d):
                    std = torch.sqrt(m_bn.running_var + m_bn.eps)
                    scale = m_bn.weight / std
                    shift = m_bn.bias - m_bn.running_mean * scale
                    
                    if hasattr(m_relu, 'shift_pre'):
                        shift = shift + m_relu.shift_pre.squeeze()
                        scale = scale * m_relu.prelu.weight.squeeze()
                        shift = shift * m_relu.prelu.weight.squeeze() + m_relu.shift_post.squeeze()
                        
                    if hasattr(m_conv, 'output_scale') and m_conv.output_scale is not None:
                        scale = scale * m_conv.output_scale.squeeze()
                        m_conv.output_scale = None
                        
                    m_conv.register_buffer("out_scale", scale.detach().clone())
                    m_conv.register_buffer("out_shift", shift.detach().clone())
                    m_conv.do_relu = True
                    m_conv.pack_weights()
                    
                    module.bn2 = nn.Identity()
                    module.act2 = nn.Identity()

        # Handle standalone BiKA_Conv2d (e.g. final layer if full_precision_head=False)
        elif module.__class__.__name__ == "BiKA_Conv2d":
            if not hasattr(module, 'packed_weight') or module.packed_weight is None:
                module.pack_weights()


def benchmark_speed(model, device, input_h, input_w, warmup, iterations,
                    use_torchscript=False):
    """Run latency and FPS benchmark."""
    model.eval()
    
    # Re-enable cuDNN now that the CUDA kernel out-of-bounds bug is fixed!
    torch.backends.cudnn.enabled = True
    
    dummy_input = torch.randn(1, 3, input_h, input_w).to(device)

    if use_torchscript:
        print("  [TorchScript] Tracing model...")
        model = torch.jit.trace(model, dummy_input)

    use_cuda_graph = (device.type == "cuda")

    # ── Warmup ──
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # ── CUDA Graph capture ──
    graph = None
    static_input = None
    static_output = None
    if use_cuda_graph:
        print("  [CUDA Graphs] Capturing forward pass...")
        static_input = dummy_input.clone()
        # Warmup for graph capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):
                    static_output = model(static_input)
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = model(static_input)
        print("  [CUDA Graphs] Captured successfully!")

    # ── Timed runs ──
    print(f"  Benchmarking ({iterations} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if graph is not None:
                graph.replay()
            else:
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)),
    }


def evaluate_iou(model, device, args, num_classes):
    """Run IoU evaluation on BDD100K val set."""
    if args.label_grouping != "none":
        g = LABEL_GROUPINGS[args.label_grouping]
        class_names = g["classes"]
        label_map = g["label_map"]
    else:
        class_names = BDD100K_CLASSES
        label_map = None

    val_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(args.bdd100k_base, "labels", "val", "*.png")))
    ]

    val_ds = BDD100KDataset(
        img_ids=val_ids,
        img_dir=os.path.join(args.bdd100k_base, "images", "val"),
        mask_dir=os.path.join(args.bdd100k_base, "labels", "val"),
        num_classes=num_classes,
        input_h=args.input_h, input_w=args.input_w,
        is_training=False, mask_suffix="_train_id", label_map=label_map,
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    metric = SegmentationMetric(num_classes)
    with torch.no_grad():
        for inp, target, _ in loader:
            out = model(inp.to(device))
            metric.update(out, target.to(device))

    miou, mdice, per_class = metric.compute()
    return miou, mdice, per_class, class_names


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # ── Load model ──
    print("=" * 60)
    print("BiKASegNet Benchmark")
    print("=" * 60)
    model, num_classes = load_model(args)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)

    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {device}")
    print(f"  Input      : {args.input_h} x {args.input_w}")
    print(f"  Classes    : {num_classes} ({args.label_grouping})")
    print(f"  Parameters : {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Params (M) : {total_params / 1e6:.3f}M")

    # ── Speed Benchmark ──
    speed = {"fps": 0.0, "mean_ms": 0.0}
    if not args.skip_speed:
        print()
        print("-" * 60)
        print("SPEED BENCHMARK")
        print("-" * 60)
        try:
            speed = benchmark_speed(
                model, device, args.input_h, args.input_w,
                args.warmup, args.iterations, args.use_torchscript,
            )
            print()
            print(f"  Mean latency   : {speed['mean_ms']:.2f} ms")
            print(f"  Std latency    : {speed['std_ms']:.2f} ms")
            print(f"  Median latency : {speed['median_ms']:.2f} ms")
            print(f"  Min latency    : {speed['min_ms']:.2f} ms")
            print(f"  Max latency    : {speed['max_ms']:.2f} ms")
            print(f"  P95 latency    : {speed['p95_ms']:.2f} ms")
            print(f"  P99 latency    : {speed['p99_ms']:.2f} ms")
            print(f"  FPS            : {speed['fps']:.2f}")
        except Exception as e:
            print(f"  [Speed Benchmark Failed]: {e}")

    # ── IoU Evaluation ──
    if not args.skip_iou:
        print()
        print("-" * 60)
        print("IoU EVALUATION (BDD100K val)")
        print("-" * 60)
        miou, mdice, per_class, class_names = evaluate_iou(
            model, device, args, num_classes
        )
        print()
        print(f"  {'Class':<20} {'IoU':>8}")
        print("  " + "-" * 30)
        for i, v in enumerate(per_class):
            name = class_names.get(i, str(i))
            if math.isnan(v):
                print(f"  {name:<20} {'N/A':>8}")
            else:
                print(f"  {name:<20} {v:>8.4f}")
        print("  " + "-" * 30)
        print(f"  {'Mean IoU':<20} {miou:>8.4f}")
        print(f"  {'Mean Dice':<20} {mdice:>8.4f}")

    # ── Summary ──
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model      : BiKASegNet (base_channels={args.base_channels})")
    print(f"  Params     : {total_params / 1e6:.3f}M")
    print(f"  Input      : {args.input_h}x{args.input_w}")
    print(f"  Device     : {device}")
    print(f"  FPS        : {speed['fps']:.2f}")
    print(f"  Latency    : {speed['mean_ms']:.2f} ms (mean)")
    if not args.skip_iou:
        print(f"  mIoU       : {miou:.4f}")
        print(f"  Road IoU   : {per_class[0]:.4f}")
        print(f"  mDice      : {mdice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
