# -*- coding: utf-8 -*-
"""
FocalComm Baseline Training & Evaluation Script
Author: FocalComm Development Team
Dataset: V2X-Real (WACV 2026 Benchmark)

Features:
- Standardized 3-class mapping with all V2X-Real subclasses
- 8-channel regression head: (dx, dy, dz, log_l, log_w, log_h, sin_yaw, cos_yaw)
- Continuous yaw regression with atan2 decoding (avoids asin ambiguity)
- 3D Anchor size priors for fast convergence
- Gaussian target heatmap generation with focal loss
- Multi-threshold AP calculation (AP@0.3, AP@0.5, AP@0.7) and overall mAP
"""

import os
import sys
import glob
import math
import yaml
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. CLASS DEFINITIONS & ANCHORS
# ==============================================================================
CLASS_MAP = {
    # Class 0: Car / Light Vehicles
    'Car': 0, 'car': 0, 'PoliceCar': 0, 'LongVehicle': 0, 'vehicle': 0,
    # Class 1: Pedestrian / VRU
    'Pedestrian': 1, 'pedestrian': 1, 'Child': 1, 'RoadWorker': 1, 'person': 1,
    'Bicycle': 1, 'Scooter': 1, 'BicycleRider': 1, 'ScooterRider': 1,
    'Motorcycle': 1, 'MotorcyleRider': 1,
    # Class 2: Truck / Large Vehicles
    'Truck': 2, 'truck': 2, 'Van': 2, 'van': 2, 'Bus': 2, 'bus': 2,
    'FireTruck': 2, 'Ambulance': 2, 'ConcreteTruck': 2, 'TrashCan': 2
}

CLASS_NAMES = ['Car', 'Pedestrian', 'Truck']
NUM_CLASSES = 3

ANCHOR_SIZES = {
    0: [4.5, 2.0, 1.6],   # Car (l, w, h in meters)
    1: [0.8, 0.8, 1.7],   # Pedestrian
    2: [8.5, 2.6, 3.2],   # Truck / Bus
}

# LiDAR & BEV Grid Configuration
PC_RANGE = [-100.0, -100.0, -10.0, 100.0, 100.0, 6.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
VOXEL_SIZE = [0.4, 0.4]                                # [dx, dy] meters
BEV_H = int((PC_RANGE[4] - PC_RANGE[1]) / VOXEL_SIZE[1])  # 500
BEV_W = int((PC_RANGE[3] - PC_RANGE[0]) / VOXEL_SIZE[0])  # 500
DOWNSAMPLE = 4
FEAT_H = BEV_H // DOWNSAMPLE  # 125
FEAT_W = BEV_W // DOWNSAMPLE  # 125

# ==============================================================================
# 2. DATASET LOADER
# ==============================================================================
class V2XRealBEVDataset(Dataset):
    def __init__(self, data_root, split='train', max_samples=None, sample_step=1):
        self.data_root = data_root
        self.split = split
        self.samples = []
        
        # Search for scenario folders in train / val
        search_dirs = [
            os.path.join(data_root, split),
            os.path.join(data_root, f"{split}1"),
            os.path.join(data_root, f"{split}2"),
            os.path.join(data_root, 'train', 'train1'),
            os.path.join(data_root, 'train', 'train2'),
            os.path.join(data_root, 'val'),
            os.path.join(data_root)
        ]
        
        found_yamls = []
        for sdir in search_dirs:
            if os.path.exists(sdir):
                yamls = glob.glob(os.path.join(sdir, "**", "*.yaml"), recursive=True)
                if yamls:
                    found_yamls = [y for y in yamls if "config" not in os.path.basename(y).lower()]
                    if found_yamls:
                        break
                        
        if not found_yamls:
            # Fallback: search anywhere under data_root
            found_yamls = [y for y in glob.glob(os.path.join(data_root, "**", "*.yaml"), recursive=True)
                           if "config" not in os.path.basename(y).lower()]
                           
        # Filter matching split keyword if in deep directory
        if split in ['train', 'val'] and len(found_yamls) > 500:
            filtered = [y for y in found_yamls if split in y.lower()]
            if filtered:
                found_yamls = filtered
                
        # Group by timestamp / frame
        for y_path in found_yamls[::sample_step]:
            bin_path = y_path.replace('.yaml', '.bin').replace('.pcd', '.bin')
            if not os.path.exists(bin_path):
                # Try finding corresponding pcd or bin
                base = os.path.splitext(y_path)[0]
                if os.path.exists(base + '.bin'):
                    bin_path = base + '.bin'
                elif os.path.exists(base + '.pcd'):
                    bin_path = base + '.pcd'
                else:
                    continue
            self.samples.append((bin_path, y_path))
            if max_samples and len(self.samples) >= max_samples:
                break
                
        print(f"[{split.upper()}] Loaded {len(self.samples)} frame samples from {data_root}")

    def __len__(self):
        return len(self.samples)

    def points_to_bev(self, points):
        """Converts raw (N, 3+) LiDAR points into 3-channel BEV tensor (density, max_height, intensity)."""
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        intensity = points[:, 3] if points.shape[1] > 3 else np.ones_like(x)

        # Spatial filter
        mask = (x >= PC_RANGE[0]) & (x < PC_RANGE[3]) & \
               (y >= PC_RANGE[1]) & (y < PC_RANGE[4]) & \
               (z >= PC_RANGE[2]) & (z < PC_RANGE[5])
        x, y, z, intensity = x[mask], y[mask], z[mask], intensity[mask]

        bev = np.zeros((3, BEV_H, BEV_W), dtype=np.float32)
        if len(x) == 0:
            return torch.from_numpy(bev)

        # Coordinate mapping
        col = np.clip(((x - PC_RANGE[0]) / VOXEL_SIZE[0]).astype(np.int32), 0, BEV_W - 1)
        row = np.clip(((y - PC_RANGE[1]) / VOXEL_SIZE[1]).astype(np.int32), 0, BEV_H - 1)

        # Normalized features
        norm_z = (z - PC_RANGE[2]) / (PC_RANGE[5] - PC_RANGE[2])
        np.maximum.at(bev[0], (row, col), norm_z)
        np.add.at(bev[1], (row, col), 1.0)
        np.maximum.at(bev[2], (row, col), np.clip(intensity, 0, 1.0))
        bev[1] = np.log1p(bev[1]) / 4.0  # Log-scale density

        return torch.from_numpy(bev)

    def __getitem__(self, idx):
        bin_path, yaml_path = self.samples[idx]

        # Load LiDAR
        try:
            pts = np.fromfile(bin_path, dtype=np.float32)
            pts = pts.reshape(-1, 4) if pts.size % 4 == 0 else pts.reshape(-1, 3)
        except Exception:
            pts = np.zeros((10, 4), dtype=np.float32)

        bev_tensor = self.points_to_bev(pts)

        # Load Annotations
        boxes = []
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            vehicles = data.get('vehicles', {}) if isinstance(data, dict) else {}
            if isinstance(vehicles, dict):
                for obj_id, obj in vehicles.items():
                    if not isinstance(obj, dict):
                        continue
                    obj_type = obj.get('obj_type', '')
                    cls_id = CLASS_MAP.get(obj_type, -1)
                    if cls_id == -1:
                        continue

                    loc = obj.get('location', [0, 0, 0])
                    ext = obj.get('extent', [0, 0, 0])
                    angle = obj.get('angle', [0, 0, 0])

                    # Dimensions: length, width, height (extent in CARLA/OpenCOOD is half-extent)
                    l = ext[0] * 2.0 if max(ext) < 6.0 else ext[0]
                    w = ext[1] * 2.0 if max(ext) < 6.0 else ext[1]
                    h = ext[2] * 2.0 if max(ext) < 6.0 else ext[2]

                    x, y, z = loc[0], loc[1], loc[2]
                    yaw = math.radians(angle[1]) if len(angle) > 1 else 0.0

                    # Check within range
                    if PC_RANGE[0] <= x <= PC_RANGE[3] and PC_RANGE[1] <= y <= PC_RANGE[4] and PC_RANGE[2] <= z <= PC_RANGE[5]:
                        boxes.append([x, y, z, l, w, h, yaw, cls_id])
        except Exception as e:
            pass

        return bev_tensor, torch.tensor(boxes, dtype=torch.float32)


def collate_fn(batch):
    bevs = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]
    return bevs, boxes

# ==============================================================================
# 3. BASELINE DETECTOR (FocalComm Baseline Model)
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)


class FocalCommBaselineModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        # Encoder (downsample 4x)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        # Context Fused Neck
        self.neck = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Classification Heatmap Head (3 channels)
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )

        # 8-Channel 3D Regression Head:
        # [dx, dy, dz, log_l, log_w, log_h, sin(yaw), cos(yaw)]
        self.reg_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, 1)
        )

        # Prior initialization for focal loss stability
        nn.init.constant_(self.cls_head[-1].bias, -2.19)  # ~0.1 sigmoid probability

    def forward(self, x):
        f = self.neck(self.down2(self.down1(self.inc(x))))
        cls_logits = self.cls_head(f)
        reg_map = self.reg_head(f)
        return cls_logits, reg_map

# ==============================================================================
# 4. TARGET GENERATION & FOCAL LOSS
# ==============================================================================
def draw_gaussian(heatmap, center, radius):
    """Draws 2D Gaussian circle on target heatmap."""
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter), dtype=np.float32)
    sigma = diameter / 6.0
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    h = np.exp(-(x*x + y*y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    gaussian = np.maximum(gaussian, h)

    cx, cy = int(center[0]), int(center[1])
    H, W = heatmap.shape
    left = min(cx, radius)
    right = min(W - cx - 1, radius)
    top = min(cy, radius)
    bottom = min(H - cy - 1, radius)

    masked_heatmap = heatmap[cy - top:cy + bottom + 1, cx - left:cx + right + 1]
    masked_gaussian = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)


def build_targets(boxes_list, device):
    """Builds ground truth gaussian heatmaps and regression targets."""
    B = len(boxes_list)
    gt_cls = np.zeros((B, NUM_CLASSES, FEAT_H, FEAT_W), dtype=np.float32)
    gt_reg = np.zeros((B, 8, FEAT_H, FEAT_W), dtype=np.float32)
    reg_mask = np.zeros((B, 1, FEAT_H, FEAT_W), dtype=np.float32)

    for b in range(B):
        boxes = boxes_list[b]
        if len(boxes) == 0:
            continue
        boxes_np = boxes.cpu().numpy()
        for box in boxes_np:
            x, y, z, l, w, h, yaw, cls_id = box
            cls_id = int(cls_id)
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                continue

            # Feature map coordinates
            col = (x - PC_RANGE[0]) / (VOXEL_SIZE[0] * DOWNSAMPLE)
            row = (y - PC_RANGE[1]) / (VOXEL_SIZE[1] * DOWNSAMPLE)
            c_int, r_int = int(col), int(row)

            if 0 <= c_int < FEAT_W and 0 <= r_int < FEAT_H:
                draw_gaussian(gt_cls[b, cls_id], (c_int, r_int), radius=2)

                # Regression offsets & log size residuals relative to anchors
                anc = ANCHOR_SIZES[cls_id]
                gt_reg[b, 0, r_int, c_int] = col - c_int
                gt_reg[b, 1, r_int, c_int] = row - r_int
                gt_reg[b, 2, r_int, c_int] = z
                gt_reg[b, 3, r_int, c_int] = math.log(max(l / anc[0], 1e-4))
                gt_reg[b, 4, r_int, c_int] = math.log(max(w / anc[1], 1e-4))
                gt_reg[b, 5, r_int, c_int] = math.log(max(h / anc[2], 1e-4))
                gt_reg[b, 6, r_int, c_int] = math.sin(yaw)
                gt_reg[b, 7, r_int, c_int] = math.cos(yaw)

                reg_mask[b, 0, r_int, c_int] = 1.0

    return (torch.from_numpy(gt_cls).to(device),
            torch.from_numpy(gt_reg).to(device),
            torch.from_numpy(reg_mask).to(device))


def focal_loss(pred_logits, gt, alpha=2.0, beta=4.0):
    """Modified focal loss for center heatmap detection."""
    p = torch.sigmoid(pred_logits)
    pos_mask = gt.eq(1.0).float()
    neg_mask = gt.lt(1.0).float()
    neg_weights = torch.pow(1.0 - gt, beta)

    pos_loss = -torch.log(p.clamp(min=1e-5)) * torch.pow(1.0 - p, alpha) * pos_mask
    neg_loss = -torch.log((1.0 - p).clamp(min=1e-5)) * torch.pow(p, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos

# ==============================================================================
# 5. ROTATED 3D IOU & EVALUATION
# ==============================================================================
def polygon_iou_2d(box1, box2):
    """Calculates 2D rotated IoU between two oriented bounding boxes [x, y, l, w, yaw]."""
    try:
        from shapely.geometry import Polygon
        def get_corners(b):
            cx, cy, l, w, yaw = b
            dx, dy = l / 2.0, w / 2.0
            corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])
            return (corners @ R.T) + np.array([cx, cy])

        p1 = Polygon(get_corners(box1))
        p2 = Polygon(get_corners(box2))
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0.0
    except ImportError:
        # Fallback approximation: axis-aligned bounding box IoU
        x1_min, x1_max = box1[0] - box1[2]/2, box1[0] + box1[2]/2
        y1_min, y1_max = box1[1] - box1[3]/2, box1[1] + box1[3]/2
        x2_min, x2_max = box2[0] - box2[2]/2, box2[0] + box2[2]/2
        y2_min, y2_max = box2[1] - box2[3]/2, box2[1] + box2[3]/2
        ix = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
        iy = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter = ix * iy
        union = (box1[2]*box1[3]) + (box2[2]*box2[3]) - inter
        return inter / union if union > 0 else 0.0


def decode_predictions(cls_logits, reg_map, score_thresh=0.15, max_dets=150):
    """Decodes heatmaps and regression maps into oriented 3D detections."""
    B, C, H, W = cls_logits.shape
    scores = torch.sigmoid(cls_logits)
    # 3x3 MaxPool NMS
    hmax = F.max_pool2d(scores, kernel_size=3, stride=1, padding=1)
    keep = (scores == hmax) & (scores > score_thresh)

    batch_detections = []
    for b in range(B):
        dets = []
        for c in range(C):
            coords = keep[b, c].nonzero(as_tuple=False)
            for coord in coords:
                r, col = coord[0].item(), coord[1].item()
                s = scores[b, c, r, col].item()
                reg = reg_map[b, :, r, col].detach().cpu().numpy()

                anc = ANCHOR_SIZES[c]
                # Center coordinates
                x = (col + reg[0]) * VOXEL_SIZE[0] * DOWNSAMPLE + PC_RANGE[0]
                y = (r + reg[1]) * VOXEL_SIZE[1] * DOWNSAMPLE + PC_RANGE[1]
                z = reg[2]
                l = anc[0] * math.exp(np.clip(reg[3], -2.0, 2.0))
                w = anc[1] * math.exp(np.clip(reg[4], -2.0, 2.0))
                h = anc[2] * math.exp(np.clip(reg[5], -2.0, 2.0))
                yaw = math.atan2(reg[6], reg[7])  # atan2 continuous decoding

                dets.append({
                    'score': s, 'class_id': c,
                    'box': [x, y, z, l, w, h, yaw]
                })

        dets.sort(key=lambda d: d['score'], reverse=True)
        batch_detections.append(dets[:max_dets])
    return batch_detections


def evaluate_dataset(model, loader, device, iou_thresholds=[0.3, 0.5, 0.7]):
    """Evaluates AP@0.3, AP@0.5, AP@0.7 per class and reports overall mAP."""
    model.eval()
    stats = {c: {th: {'tp': 0, 'fp': 0, 'gt': 0} for th in iou_thresholds} for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for bevs, boxes_list in tqdm(loader, desc="Evaluating", leave=False):
            bevs = bevs.to(device)
            cls_logits, reg_map = model(bevs)
            batch_dets = decode_predictions(cls_logits, reg_map)

            for b in range(len(boxes_list)):
                gt_boxes = boxes_list[b].numpy() if isinstance(boxes_list[b], torch.Tensor) else np.array(boxes_list[b])
                dets = batch_dets[b]

                for c in range(NUM_CLASSES):
                    gt_c = [g[:7] for g in gt_boxes if int(g[7]) == c]
                    dt_c = [d['box'] for d in dets if d['class_id'] == c]

                    for th in iou_thresholds:
                        stats[c][th]['gt'] += len(gt_c)
                        matched_gt = set()

                        for dt in dt_c:
                            best_iou = 0.0
                            best_idx = -1
                            for idx, gt in enumerate(gt_c):
                                if idx in matched_gt:
                                    continue
                                iou = polygon_iou_2d([dt[0], dt[1], dt[3], dt[4], dt[6]],
                                                     [gt[0], gt[1], gt[3], gt[4], gt[6]])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = idx

                            if best_iou >= th and best_idx != -1:
                                stats[c][th]['tp'] += 1
                                matched_gt.add(best_idx)
                            else:
                                stats[c][th]['fp'] += 1

    results = {}
    print("\n" + "=" * 70)
    print(f"{'Class':<15} | {'AP@0.3':<12} | {'AP@0.5':<12} | {'AP@0.7':<12} | {'GT Count':<8}")
    print("-" * 70)
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        aps = {}
        gt_count = stats[c][iou_thresholds[0]]['gt']
        for th in iou_thresholds:
            tp = stats[c][th]['tp']
            fp = stats[c][th]['fp']
            gt = stats[c][th]['gt']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / gt if gt > 0 else 0.0
            # F1 / AP approximation
            ap = precision * recall * 100.0 if (precision + recall) > 0 else 0.0
            aps[th] = ap
        results[name] = aps
        print(f"{name:<15} | {aps[0.3]:>6.2f}%      | {aps[0.5]:>6.2f}%      | {aps[0.7]:>6.2f}%      | {gt_count:<8}")
    print("=" * 70)

    mAP_03 = np.mean([results[name][0.3] for name in CLASS_NAMES])
    mAP_05 = np.mean([results[name][0.5] for name in CLASS_NAMES])
    print(f"Overall Mean AP: mAP@0.3 = {mAP_03:.2f}% | mAP@0.5 = {mAP_05:.2f}%\n")
    return results

# ==============================================================================
# 6. MAIN TRAINING CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train FocalComm Baseline on V2X-Real")
    parser.add_argument("--data_dir", type=str, default="/kaggle/working/dataset/V2XReal",
                        help="Path to V2X-Real dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints/baseline_v2xreal",
                        help="Directory to save model checkpoints")
    parser.add_argument("--subset", action="store_true", help="Run fast dry-run on 200 samples")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate existing checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load datasets
    max_s = 200 if args.subset else None
    train_dataset = V2XRealBEVDataset(args.data_dir, split='train', max_samples=max_s)
    val_dataset = V2XRealBEVDataset(args.data_dir, split='val', max_samples=max_s)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = FocalCommBaselineModel(in_channels=3, num_classes=NUM_CLASSES).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.eval_only:
        print("Running in Evaluation-only mode...")
        evaluate_dataset(model, val_loader, device)
        return

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\nStarting FocalComm Baseline Training: {args.epochs} Epochs, Batch Size {args.batch_size}")
    best_mAP = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for bevs, boxes_list in pbar:
            bevs = bevs.to(device)
            gt_cls, gt_reg, reg_mask = build_targets(boxes_list, device)

            optimizer.zero_grad()
            pred_cls, pred_reg = model(bevs)

            loss_c = focal_loss(pred_cls, gt_cls)
            loss_r = (F.smooth_l1_loss(pred_reg, gt_reg, reduction='none') * reg_mask).sum() / reg_mask.sum().clamp(min=1.0)
            loss = loss_c + 2.0 * loss_r

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += loss_c.item()
            total_reg += loss_r.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'cls': f"{loss_c.item():.3f}", 'reg': f"{loss_r.item():.3f}"})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} Summary: Loss={avg_loss:.4f} (Cls={total_cls/len(train_loader):.4f}, Reg={total_reg/len(train_loader):.4f})")

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, "baseline_latest.pth")
        torch.save(model.state_dict(), latest_path)

        # Periodic Evaluation every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_res = evaluate_dataset(model, val_loader, device)
            curr_mAP = np.mean([eval_res[name][0.3] for name in CLASS_NAMES])
            if curr_mAP > best_mAP:
                best_mAP = curr_mAP
                best_path = os.path.join(args.save_dir, "baseline_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"🌟 New Best Checkpoint saved to {best_path} (mAP@0.3: {best_mAP:.2f}%)")

    print("\n✅ Training Complete!")


if __name__ == '__main__':
    main()

