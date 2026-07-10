import argparse
import os
import random
import shutil
import sys
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import models as archs
import training.losses as losses
from bika import BiKA_Conv2d, BiKA_Linear
from data import BDD100KDataset, BDD100K_NUM_CLASSES, LABEL_GROUPINGS
from training.metrics import SegmentationMetric


# ── Knowledge Distillation helpers ───────────────────────────────────────────

def load_teacher_model(teacher_weights, kan_seg_root, num_classes, device):
    """Load a trained UKAN teacher model for knowledge distillation.
    
    Args:
        teacher_weights: Path to the teacher's model_best.pth.
        kan_seg_root: Root of the KAN-Road-Segmentation/Seg_UKAN directory.
        num_classes: Number of segmentation classes (must match teacher).
        device: CUDA device to place the teacher on.
    
    Returns:
        Frozen teacher model in eval mode, or None if loading fails.
    """
    # Add KAN-Road-Segmentation to path so we can import UKAN
    if kan_seg_root not in sys.path:
        sys.path.insert(0, kan_seg_root)
    try:
        from archs import UKAN
    except ImportError:
        print("[KD] WARNING: Could not import UKAN from KAN-Road-Segmentation. "
              "Skipping knowledge distillation.")
        return None
    
    # Read teacher config to get kan_type and embed_dims
    config_path = os.path.join(os.path.dirname(teacher_weights), "config.yml")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            teacher_cfg = yaml.safe_load(f)
        kan_type = teacher_cfg.get("kan_type", "ReLU")
        embed_dims = teacher_cfg.get("input_list", [64, 128, 256])
    else:
        kan_type = "ReLU"
        embed_dims = [64, 128, 256]
    
    teacher = UKAN(
        num_classes=num_classes,
        input_channels=3,
        embed_dims=embed_dims,
        kan_type=kan_type,
    )
    
    state = torch.load(teacher_weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    teacher.load_state_dict(state, strict=False)
    teacher = teacher.to(device).eval()
    
    # Freeze all parameters
    for p in teacher.parameters():
        p.requires_grad = False
    
    print(f"[KD] Loaded teacher UKAN ({kan_type}) from '{teacher_weights}'")
    return teacher


def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """KL-divergence distillation loss between student and teacher logits."""
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature * temperature)


def boundary_loss(pred_logits, target, kernel_size=3):
    """Auxiliary loss that penalizes errors on boundary pixels (AMP-safe)."""
    with torch.no_grad():
        target_f = target.float().unsqueeze(1)
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            device=target.device, dtype=target_f.dtype)
        pad = kernel_size // 2
        dilated = F.conv2d(target_f, kernel, padding=pad).clamp(0, 1)
        eroded = 1.0 - F.conv2d(1.0 - target_f, kernel, padding=pad).clamp(0, 1)
        boundary_mask = ((dilated - eroded) > 0).float().squeeze(1)  # (B, H, W)
    
    # Use class-0 logits directly (AMP-safe, no softmax needed)
    # For 2-class: logit difference = logits[:,0] - logits[:,1]
    boundary_sum = boundary_mask.sum() + 1e-5
    logit_diff = pred_logits[:, 0] - pred_logits[:, 1]  # (B, H, W)
    target_binary = (target == 0).float()
    loss = F.binary_cross_entropy_with_logits(
        logit_diff * boundary_mask,
        target_binary * boundary_mask,
        reduction="sum",
    ) / boundary_sum
    return loss


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    import argparse
    raise argparse.ArgumentTypeError("Boolean value expected.")


def list_type(s):
    return [int(a) for a in s.split(",")]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default=None, help="experiment name")
    parser.add_argument("--model_name", default="BiKASegNet", choices=["BiKASegNet"])
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("-b", "--batch_size", default=8, type=int)
    parser.add_argument("--dataseed", default=2981, type=int)

    # Model
    parser.add_argument("--arch", "-a", default="BiKASegNet", choices=archs.__all__)
    parser.add_argument("--input_channels", default=3, type=int)
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--input_w", default=256, type=int)
    parser.add_argument("--input_h", default=192, type=int)
    parser.add_argument("--base_channels", default=32, type=int)

    # Loss
    parser.add_argument("--loss", default="CrossEntropyDiceLoss", choices=losses.__all__ + ["CrossEntropyLoss"])
    parser.add_argument(
        "--class_weights",
        default="",
        type=str,
        help="Per-class loss weights to counteract imbalance. Either a "
             "comma-separated list (len = num_classes), or a path to a "
             ".npy/.pt file (as produced by compute_class_freq.py). Empty "
             "or 'none' disables weighting.",
    )

    # Dataset
    parser.add_argument("--dataset", default="bdd100k")
    parser.add_argument(
        "--label_grouping",
        default="none",
        choices=["none"] + list(LABEL_GROUPINGS),
        help="Remap the 20 seg classes into a smaller grouped set (no new "
             "data). 'drive5' = road/sidewalk/vehicle/person/background. "
             "Sets num_classes automatically.",
    )
    parser.add_argument("--data_dir", default="inputs")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument(
        "--bdd100k_base", default="/mnt/ssd-0/M2_internship/bdd100k_seg/bdd100k/seg"
    )

    # Optimizer
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)
    parser.add_argument(
        "--clip_grad",
        default=5.0,
        type=float,
        help="Max gradient norm for clipping; <= 0 disables clipping.",
    )
    parser.add_argument(
        "--bika_weight_clamp",
        default=0.0,
        type=float,
        help="Clamp BiKA weights to [-v, v] after each optimizer step "
             "(0 disables). Keeps |z|=(x+b)*w inside the STE window so "
             "connections keep receiving gradient (BNN practice); try 1.0 "
             "if training stalls/regresses after some epochs.",
    )

    # Scheduler
    parser.add_argument(
        "--scheduler",
        default="CosineAnnealingLR",
        choices=[
            "CosineAnnealingLR",
            "ReduceLROnPlateau",
            "MultiStepLR",
            "ConstantLR",
            "OneCycleLR",
        ],
    )
    parser.add_argument("--min_lr", default=1e-5, type=float)
    parser.add_argument("--factor", default=0.1, type=float)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--milestones", default="1,2", type=str)
    parser.add_argument("--gamma", default=2 / 3, type=float)
    parser.add_argument("--early_stopping", default=-1, type=int)

    parser.add_argument("--cfg", type=str, metavar="FILE")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--resume", default=False, type=str2bool)
    parser.add_argument(
        "--checkpoint_path",
        default="",
        type=str,
        help="Explicit path to a .pth checkpoint to resume from. "
             "If empty and --resume True, auto-discovers "
             "'checkpoint_last.pth' inside the experiment output dir.",
    )
    parser.add_argument(
        "--init_weights",
        default="",
        type=str,
        help="Path to a checkpoint to load MODEL WEIGHTS ONLY from, then "
             "train from epoch 0 with a fresh optimizer/scheduler at --lr. "
             "Use for a warm restart (SGDR-style) of a converged run — "
             "unlike --resume, which also restores the (decayed) optimizer "
             "LR. Ignored if --resume is True.",
    )

    # DDP/AMP
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--use_amp", default=True, type=str2bool)
    parser.add_argument("--amp_dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--sync_bn", default=True, type=str2bool)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--compile_model", default=False, type=str2bool)

    # Knowledge Distillation (CKD — BNext, 2024)
    parser.add_argument(
        "--teacher_weights", default="", type=str,
        help="Path to teacher model_best.pth (UKAN from KAN-Road-Segmentation). "
             "Set to enable knowledge distillation.",
    )
    parser.add_argument(
        "--kan_seg_root", default="", type=str,
        help="Path to KAN-Road-Segmentation/Seg_UKAN root (for importing UKAN). "
             "Auto-detected if not set.",
    )
    parser.add_argument(
        "--kd_alpha", default=0.3, type=float,
        help="Distillation weight: loss = (1-alpha)*task + alpha*kd. "
             "CKD schedule: round1=0.3, round2=0.5, round3=0.7.",
    )
    parser.add_argument(
        "--kd_temperature", default=4.0, type=float,
        help="KD softmax temperature (higher = softer probabilities).",
    )
    parser.add_argument(
        "--teacher_input_h", default=0, type=int,
        help="Teacher's training input height (e.g. 384 for UKAN lane_fg). "
             "0 = same as student's --input_h.",
    )
    parser.add_argument(
        "--teacher_input_w", default=0, type=int,
        help="Teacher's training input width (e.g. 640 for UKAN lane_fg). "
             "0 = same as student's --input_w.",
    )
    parser.add_argument(
        "--boundary_weight", default=0.5, type=float,
        help="Weight for boundary-aware auxiliary loss (0 = disabled).",
    )

    return vars(parser.parse_args())


def parse_class_weights(spec, num_classes):
    """Parse --class_weights into a float32 tensor of length num_classes,
    or None. Accepts a comma-separated list or a .npy/.pt file path."""
    spec = (spec or "").strip()
    if not spec or spec.lower() == "none":
        return None
    if "," in spec:
        w = [float(x) for x in spec.split(",")]
    elif os.path.isfile(spec):
        if spec.endswith(".npy"):
            w = np.load(spec).tolist()
        else:
            w = torch.load(spec, map_location="cpu").tolist()
    else:
        raise ValueError(f"--class_weights: not a list or existing file: {spec!r}")
    if len(w) != num_classes:
        raise ValueError(
            f"--class_weights has {len(w)} entries but num_classes={num_classes}"
        )
    return torch.tensor(w, dtype=torch.float32)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
        return True, rank, world_size, local_rank
    if torch.cuda.is_available():
        return False, 0, 1, 0
    raise RuntimeError("CUDA not available")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_tensor(tensor, world_size):
    if world_size == 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def get_base_model(model, distributed):
    return model.module if distributed else model


@torch.no_grad()
def clamp_bika_weights(model, limit):
    for m in model.modules():
        if isinstance(m, (BiKA_Conv2d, BiKA_Linear)):
            m.weight.clamp_(-limit, limit)


def load_bika_model(config) -> nn.Module:
    return archs.__dict__[config["arch"]](
        num_classes=config["num_classes"],
        in_channels=config.get("input_channels", 3),
        base_channels=config.get("base_channels", 16),
    )


def train_one_epoch(
    config,
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    epoch,
    rank,
    world_size,
    scheduler=None,
    teacher=None,
):
    avg_meters = {"loss": AverageMeter()}
    seg_metric = SegmentationMetric(config["num_classes"])
    model.train()
    pbar = (
        tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        if is_main_process(rank)
        else None
    )
    optimizer.zero_grad()

    kd_alpha = config.get("kd_alpha", 0.0) if teacher is not None else 0.0
    kd_temp = config.get("kd_temperature", 4.0)
    bnd_weight = config.get("boundary_weight", 0.0)

    for batch_idx, (inp, target, _) in enumerate(train_loader):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with amp.autocast("cuda", enabled=config["use_amp"], dtype=getattr(torch, config.get("amp_dtype", "float16"))):
            output = model(inp)
            task_loss = criterion(output, target)

            loss = task_loss

            # Knowledge Distillation loss (CKD — BNext, 2024)
            if teacher is not None and kd_alpha > 0:
                with torch.no_grad():
                    # UKAN uses patch embeddings that require specific spatial dims.
                    # Resize input to teacher's training resolution, then resize
                    # output back to student's resolution for the KD loss.
                    t_h = config.get("teacher_input_h") or inp.shape[2]
                    t_w = config.get("teacher_input_w") or inp.shape[3]
                    if (t_h, t_w) != (inp.shape[2], inp.shape[3]):
                        teacher_inp = F.interpolate(
                            inp, size=(t_h, t_w),
                            mode="bilinear", align_corners=False,
                        )
                    else:
                        teacher_inp = inp
                    teacher_out = teacher(teacher_inp)
                    # Always match to student output resolution
                    if teacher_out.shape[2:] != output.shape[2:]:
                        teacher_out = F.interpolate(
                            teacher_out, size=output.shape[2:],
                            mode="bilinear", align_corners=False,
                        )
                distill = kd_loss(output, teacher_out, temperature=kd_temp)
                loss = (1.0 - kd_alpha) * task_loss + kd_alpha * distill

            # Boundary-aware auxiliary loss
            if bnd_weight > 0:
                loss = loss + bnd_weight * boundary_loss(output, target)

            loss = loss / config["grad_accum_steps"]

        scaler.scale(loss).backward()
        seg_metric.update(output.detach(), target)

        if (batch_idx + 1) % config["grad_accum_steps"] == 0:
            if config["clip_grad"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if config["bika_weight_clamp"] > 0:
                clamp_bika_weights(model, config["bika_weight_clamp"])
            if config["scheduler"] == "OneCycleLR" and scheduler is not None:
                scheduler.step()

        loss_reduced = reduce_tensor(
            loss.detach() * config["grad_accum_steps"], world_size
        )
        avg_meters["loss"].update(loss_reduced.item(), inp.size(0))

        if pbar is not None:
            running_iou, _, per_class = seg_metric.compute()
            pbar.set_postfix(
                OrderedDict(
                    loss=f"{avg_meters['loss'].avg:.4f}",
                    iou=f"{running_iou:.4f}",
                    iou_0=f"{per_class[0]:.4f}",
                )
            )
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    seg_metric.all_reduce()
    miou, _, per_class = seg_metric.compute()

    return OrderedDict(loss=avg_meters["loss"].avg, iou=miou, iou_0=per_class[0])


def validate(config, val_loader, model, criterion, rank, world_size):
    avg_meters = {"loss": AverageMeter()}
    seg_metric = SegmentationMetric(config["num_classes"])
    model.eval()
    pbar = (
        tqdm(total=len(val_loader), desc="Validation")
        if is_main_process(rank)
        else None
    )

    with torch.no_grad():
        for inp, target, _ in val_loader:
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with amp.autocast("cuda", enabled=config["use_amp"], dtype=getattr(torch, config.get("amp_dtype", "float16"))):
                output = model(inp)
                loss = criterion(output, target)

            seg_metric.update(output, target)
            loss_reduced = reduce_tensor(loss.detach(), world_size)
            avg_meters["loss"].update(loss_reduced.item(), inp.size(0))

            if pbar is not None:
                running_iou, running_dice, per_class = seg_metric.compute()
                pbar.set_postfix(
                    OrderedDict(
                        loss=f"{avg_meters['loss'].avg:.4f}",
                        iou=f"{running_iou:.4f}",
                        iou_0=f"{per_class[0]:.4f}",
                        dice=f"{running_dice:.4f}",
                    )
                )
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    seg_metric.all_reduce()
    miou, mdice, per_class = seg_metric.compute()

    return OrderedDict(
        loss=avg_meters["loss"].avg,
        iou=miou,
        dice=mdice,
        iou_0=per_class[0],
    )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def main():
    config = parse_args()
    distributed, rank, world_size, local_rank = setup_distributed()

    def seed_torch(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    seed_torch(seed=config["dataseed"])

    if config["dataset"] == "bdd100k":
        seed_torch(seed=config["dataseed"])
    if config["name"] is None:
        config["name"] = f"{config['dataset']}_{config['model_name']}"
    exp_name = config["name"]
    exp_dir = os.path.join(config["output_dir"], exp_name)

    if is_main_process(rank):
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, "config.yml"), "w") as f:
            yaml.safe_dump(config, f)

    label_map = None
    if config["dataset"] == "bdd100k":
        config["num_classes"] = BDD100K_NUM_CLASSES
    grouping = config.get("label_grouping", "none")
    if grouping and grouping != "none":
        g = LABEL_GROUPINGS[grouping]
        config["num_classes"] = g["num_classes"]
        label_map = g["label_map"]
        if is_main_process(rank):
            print(f"[Grouping] '{grouping}' -> {g['num_classes']} classes: "
                  f"{list(g['classes'].values())}")

    writer = SummaryWriter(exp_dir) if is_main_process(rank) else None

    class_weights = parse_class_weights(config.get("class_weights"), config["num_classes"])
    if class_weights is not None and is_main_process(rank):
        print(f"[Loss] Using class weights: {class_weights.tolist()}")

    if config["loss"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).cuda()
    elif config["loss"] == "CrossEntropyDiceLoss":
        criterion = losses.CrossEntropyDiceLoss(class_weights=class_weights).cuda()
    else:
        criterion = losses.__dict__[config["loss"]]().cuda()

    model = load_bika_model(config)

    # Warm restart: load weights only, then train fresh from epoch 0. Done
    # before DDP/cuda wrapping so it targets the plain module. Skipped when
    # --resume is set (resume restores full training state instead).
    if config.get("init_weights") and not config["resume"]:
        ckpt = torch.load(config["init_weights"], map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Keep only tensors whose shape matches the current model, so a
        # checkpoint with a different head (e.g. 20-class v5 -> 19-class
        # bench19) loads its shared backbone and leaves the mismatched
        # layers (the final 1x1 head) at their fresh init. strict=False
        # alone is not enough — it still errors on shape mismatches.
        
        # Map old Sequential keys (e.g., block.0) to new attribute keys (e.g., conv1, bn1)
        mapped_state = {}
        for k, v in state.items():
            new_k = k
            if ".block." in k:
                parts = k.split(".")
                try:
                    block_idx = parts.index("block")
                    idx = parts[block_idx + 1]
                    if idx == "0":
                        parts[block_idx:block_idx+2] = ["conv1"]
                    elif idx == "1":
                        parts[block_idx:block_idx+2] = ["bn1"]
                    elif idx == "3":
                        parts[block_idx:block_idx+2] = ["conv2"]
                    elif idx == "4":
                        parts[block_idx:block_idx+2] = ["bn2"]
                except (ValueError, IndexError):
                    pass
                new_k = ".".join(parts)
            mapped_state[new_k] = v
        state = mapped_state

        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in state.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped = [k for k in state if k not in filtered]
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if is_main_process(rank):
            print(f"[InitWeights] Loaded {len(filtered)}/{len(state)} tensors "
                  f"from '{config['init_weights']}'")
            if skipped:
                print(f"[InitWeights] Re-initialized (shape mismatch / new): {skipped}")

    if is_main_process(rank):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params
        print("\n" + "=" * 70)
        print(f"  MODEL: {config['arch']}")
        print("=" * 70)
        print(f"  Total parameters      : {total_params:>12,}")
        print(f"  Trainable parameters  : {trainable_params:>12,}")
        print(f"  Non-trainable params  : {non_trainable:>12,}")
        print("-" * 70)
        print(f"  Estimated model size (FP32) : {total_params * 4 / 1024**2:>8.2f} MB")
        print(f"  Estimated model size (FP16) : {total_params * 2 / 1024**2:>8.2f} MB")
        print(f"  Estimated model size (INT8) : {total_params * 1 / 1024**2:>8.2f} MB")
        print("=" * 70 + "\n")

    if distributed and config["sync_bn"]:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    if config["compile_model"] and hasattr(torch, "compile"):
        model = torch.compile(model)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    base_model = get_base_model(model, distributed)
    # Exclude ALL BiKA-layer parameters, plus biases and BatchNorm affine
    # params, from weight decay. BiKA forward output depends only on
    # sign(w) and the threshold (= -bias), so no loss pressure opposes the
    # decay term: decaying biases drags thresholds back to the degenerate
    # bunched-at-zero init, and decaying weights grinds |w| to zero, which
    # kills the backward pass entirely (grad_bias and grad_input are both
    # proportional to w). Verified empirically: v4 trunk weights collapsed
    # to ~0 by epoch 30 and training plateaued.
    bika_param_ids = set()
    for m in base_model.modules():
        if isinstance(m, (BiKA_Conv2d, BiKA_Linear)):
            bika_param_ids.update(id(p) for p in m.parameters(recurse=False))

    decay_params, no_decay_params = [], []
    for name, p in base_model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in bika_param_ids or p.ndim == 1 or name.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {
            "params": decay_params,
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": no_decay_params,
            "lr": config["lr"],
            "weight_decay": 0.0,
        },
    ]

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(param_groups)
    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.SGD(
            param_groups,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )

    _use_scaler = config["use_amp"] and config.get("amp_dtype", "float16") == "float16"
    scaler = amp.GradScaler("cuda", enabled=_use_scaler)

    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["factor"],
            patience=config["patience"],
            min_lr=config["min_lr"],
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config["milestones"].split(",")],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "OneCycleLR":
        scheduler = None
    else:
        scheduler = None

    if is_main_process(rank):
        shutil.copy2(__file__, os.path.join(exp_dir, "train.py"))

    # ── Load teacher model for Knowledge Distillation ──────────────────────
    teacher = None
    if config.get("teacher_weights"):
        kan_root = config.get("kan_seg_root", "").strip()
        if not kan_root:
            # Auto-detect: check common layouts relative to BiKA repo
            bika_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            parent = os.path.dirname(bika_root)
            for candidate in [
                os.path.join(parent, "KAN-Road-Segmentation", "Seg_UKAN"),
                os.path.join(parent, "kan", "KAN-Road-Segmentation", "Seg_UKAN"),
            ]:
                if os.path.isdir(candidate):
                    kan_root = candidate
                    break
            else:
                kan_root = os.path.join(parent, "KAN-Road-Segmentation", "Seg_UKAN")
        teacher = load_teacher_model(
            config["teacher_weights"],
            kan_root,
            config["num_classes"],
            device=f"cuda:{local_rank}",
        )

    bdd = config["bdd100k_base"]
    train_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "train", "*.png")))
    ]
    val_img_ids = [
        os.path.splitext(os.path.basename(p))[0].replace("_train_id", "")
        for p in sorted(glob(os.path.join(bdd, "labels", "val", "*.png")))
    ]

    train_dataset = BDD100KDataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(bdd, "images", "train"),
        mask_dir=os.path.join(bdd, "labels", "train"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
        is_training=True,
        mask_suffix="_train_id",
        label_map=label_map,
    )
    val_dataset = BDD100KDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(bdd, "images", "val"),
        mask_dir=os.path.join(bdd, "labels", "val"),
        img_ext=".jpg",
        mask_ext=".png",
        num_classes=config["num_classes"],
        input_h=config["input_h"],
        input_w=config["input_w"],
        is_training=False,
        mask_suffix="_train_id",
        label_map=label_map,
    )

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if distributed else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        prefetch_factor=config["prefetch_factor"] if config["num_workers"] > 0 else None,
        persistent_workers=(config["num_workers"] > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        prefetch_factor=config["prefetch_factor"] if config["num_workers"] > 0 else None,
        persistent_workers=(config["num_workers"] > 0),
    )

    if config["scheduler"] == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"] * 10,
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )

    start_epoch = 0
    best_iou = 0.0
    trigger = 0
    log = OrderedDict(
        epoch=[], lr=[], loss=[], iou=[], train_iou_0=[], val_loss=[], val_iou=[], val_dice=[], val_iou_0=[]
    )

    if config["resume"]:
        ckpt_path = (config.get("checkpoint_path") or "").strip() or os.path.join(
            exp_dir, "checkpoint_last.pth"
        )
        if is_main_process(rank):
            print(f"[Resume] Looking for checkpoint: {ckpt_path}")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(
                ckpt_path,
                map_location=f"cuda:{local_rank}",
                weights_only=False,
            )
            _base = get_base_model(model, distributed)
            _base.load_state_dict(ckpt["model_state_dict"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_iou = float(ckpt.get("best_iou", 0.0))
            if is_main_process(rank):
                log_csv = os.path.join(exp_dir, "log.csv")
                if os.path.isfile(log_csv):
                    log = OrderedDict(pd.read_csv(log_csv).to_dict(orient="list"))
                print(
                    f"[Resume] Loaded '{ckpt_path}' "
                    f"(saved epoch {ckpt['epoch']}) -> "
                    f"best_iou={best_iou:.4f}, "
                    f"continuing from epoch {start_epoch}"
                )
            remaining = config["epochs"] - start_epoch
            if remaining > 0:
                if config["scheduler"] == "CosineAnnealingLR":
                    scheduler = lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=remaining,
                        eta_min=config["min_lr"],
                    )
                elif config["scheduler"] == "OneCycleLR":
                    scheduler = lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=config["lr"] * 10,
                        epochs=remaining,
                        steps_per_epoch=len(train_loader),
                        pct_start=0.1,
                    )
        else:
            if is_main_process(rank):
                print(
                    f"[Resume] WARNING: checkpoint not found at '{ckpt_path}'. "
                    "Starting from scratch."
                )

    if distributed:
        _t = torch.tensor([start_epoch], dtype=torch.long, device=f"cuda:{local_rank}")
        dist.broadcast(_t, src=0)
        start_epoch = int(_t.item())

    for epoch in range(start_epoch, config["epochs"]):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_log = train_one_epoch(
            config,
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            rank,
            world_size,
            scheduler=scheduler,
            teacher=teacher,
        )
        val_log = validate(config, val_loader, model, criterion, rank, world_size)

        if config["scheduler"] == "CosineAnnealingLR" and scheduler is not None:
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau" and scheduler is not None:
            scheduler.step(val_log["loss"])
        elif config["scheduler"] == "MultiStepLR" and scheduler is not None:
            scheduler.step()

        if is_main_process(rank):
            lr_now = optimizer.param_groups[0]["lr"]
            log["epoch"].append(epoch)
            log["lr"].append(lr_now)
            log["loss"].append(train_log["loss"])
            log["iou"].append(train_log["iou"])
            log["train_iou_0"].append(train_log["iou_0"])
            log["val_loss"].append(val_log["loss"])
            log["val_iou"].append(val_log["iou"])
            log["val_dice"].append(val_log["dice"])
            log["val_iou_0"].append(val_log["iou_0"])
            pd.DataFrame(log).to_csv(os.path.join(exp_dir, "log.csv"), index=False)

            if writer is not None:
                writer.add_scalar("train/loss", train_log["loss"], epoch)
                writer.add_scalar("train/iou", train_log["iou"], epoch)
                writer.add_scalar("train/iou_0", train_log["iou_0"], epoch)
                writer.add_scalar("val/loss", val_log["loss"], epoch)
                writer.add_scalar("val/iou", val_log["iou"], epoch)
                writer.add_scalar("val/dice", val_log["dice"], epoch)
                writer.add_scalar("val/iou_0", val_log["iou_0"], epoch)
                writer.add_scalar("lr", lr_now, epoch)

            base_model = get_base_model(model, distributed)
            checkpoint_last = {
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict(),
                "best_iou": best_iou,
                "config": config,
            }
            torch.save(checkpoint_last, os.path.join(exp_dir, "checkpoint_last.pth"))

            if val_log["iou"] > best_iou:
                best_iou = val_log["iou"]
                torch.save(
                    base_model.state_dict(), os.path.join(exp_dir, "model_best.pth")
                )
                torch.save(
                    checkpoint_last, os.path.join(exp_dir, "checkpoint_best.pth")
                )
                trigger = 0
            else:
                trigger += 1

            if config["early_stopping"] >= 0 and trigger >= config["early_stopping"]:
                break

        if distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
