#!/usr/bin/env python3
"""Evaluate QAT checkpoint accuracy — debug version (one2many only).

Usage:
    python eval.py --ckpt epoch3.pt
    # Writes eval_debug.log with full diagnostics.
"""

import argparse
import copy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER
from ultralytics.utils.ax_quantizer import AXQuantizer, ax_load_config
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401

warnings.filterwarnings("ignore")

DEBUG_LOG = Path(__file__).resolve().parent / "eval_debug.log"


def log(msg: str):
    """Print to stdout and append to debug log."""
    print(msg)
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


class FakeTrainer:
    def __init__(self, float_model, qat_model, data_dict, device):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_items = torch.zeros(3)
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        from collections import namedtuple
        self.stopper = namedtuple("Stopper", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(
            half=False, amp=False, compile=False, plots=False, end2end=False,
            conf=0.001, iou=0.7, max_det=300, single_cls=False, agnostic_nms=False,
            save_json=False, save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            return dict(zip([f"{prefix}/{x}" for x in self.loss_names],
                           [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


def parse_args():
    parser = argparse.ArgumentParser(description="QAT eval debug — one2many")
    parser.add_argument("--ckpt", default="./runs/detect/exp32-yolo26n-S16matmul-e2eFalse/weights/epoch3.pt")
    parser.add_argument("--model", default="yolo26n.yaml")
    parser.add_argument("--pretrained", default="yolo26n.pt")
    parser.add_argument("--data", default="coco.yaml")
    parser.add_argument("--quant-config", default="config.json", dest="quant_config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)

    # Clear previous log
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()

    log("=" * 70)
    log(f"eval.py DEBUG — one2many only")
    log(f"ckpt:   {args.ckpt}")
    log(f"config: {args.quant_config}")
    log(f"device: {args.device}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load float model
    # ------------------------------------------------------------------
    model = YOLO(args.model, task="detect").load(args.pretrained)
    float_model = model.model.float().to(device)
    # Must be in TRAIN mode for export_for_training to trace the per-scale training path
    float_model.train()
    # Keep end2end=True so the one2one branch weights are included in the exported graph.
    # The validator will select one2many branch via _rebuild_pt2e_predictions.

    hyp = dict(DEFAULT_CFG_DICT, **float_model.args) if isinstance(float_model.args, dict) else {}
    hyp.setdefault("box", 7.5); hyp.setdefault("cls", 0.5); hyp.setdefault("dfl", 1.5)
    float_model.args = SimpleNamespace(**hyp)

    # ------------------------------------------------------------------
    # 2. Build PT2E prepared model — must keep end2end=True for match
    # ------------------------------------------------------------------
    log("[step 2] prepare_pt2e_qat_model (end2end=True inside model) ...")
    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model,
        device=device,
        config_path=args.quant_config,
        imgsz=args.imgsz,
        dynamic_batch_max=128,
    )

    # Now set end2end=False on loss reference model so validator picks one2many.
    # But first: cache E2ELoss (not v8DetectionLoss) — it handles the nested dict format.
    float_model.model[-1].end2end = True
    float_model.criterion = float_model.init_criterion()
    log(f"  criterion cached: {type(float_model.criterion).__name__}")
    float_model.model[-1].end2end = False  # now switch for one2many validation

    # 3. Load QAT state AFTER eval prep (matches training validation flow)
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    if qat_state is None:
        log("ERROR: checkpoint has no 'qat_ema', 'qat_model' or 'model' key")
        sys.exit(1)
    log(f"  loading from {'qat_ema' if ckpt.get('qat_ema') is not None else 'qat_model'}")

    # Apply the same eval preparation the validator does
    from ultralytics.engine.validator import BaseValidator
    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)
    log(f"  after _prepare_pt2e_model_for_eval: prepared.training={prepared.training}")

    # Now load state
    fresh_keys = set(prepared.state_dict().keys())
    ckpt_keys = set(qat_state.keys())
    log(f"  fresh prepared model keys: {len(fresh_keys)}")
    log(f"  checkpoint qat_model keys: {len(ckpt_keys)}")

    missing = fresh_keys - ckpt_keys
    unexpected = ckpt_keys - fresh_keys
    log(f"  missing: {len(missing)}")
    if missing:
        for k in sorted(missing)[:10]:
            log(f"    - {k}")
    log(f"  unexpected: {len(unexpected)}")
    if unexpected:
        for k in sorted(unexpected)[:10]:
            log(f"    + {k}")

    missing_report = prepared.load_state_dict(qat_state, strict=False)
    log(f"  strict=False: missing={len(missing_report.missing_keys)}, unexpected={len(missing_report.unexpected_keys)}")
    prepared.to(device)
    log("[step 3] QAT state loaded")

    # ------------------------------------------------------------------
    # 4. Read training-internal validation result for comparison
    # ------------------------------------------------------------------
    ckpt_epoch = ckpt.get("epoch", "?")
    results_csv = Path(args.ckpt).parent.parent / "results.csv"
    train_mAP = "?"
    if results_csv.exists():
        import csv
        with results_csv.open() as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i == ckpt_epoch:
                    train_mAP = float(row["metrics/mAP50-95(B)"])
                    log(f"[ref] training internal validation epoch{ckpt_epoch}: mAP50-95 = {train_mAP:.4f} (CSV epoch{i+1})")
                    break
    if train_mAP == "?":
        log(f"[ref] training validation result not found for epoch {ckpt_epoch}")

    # ------------------------------------------------------------------
    # 5. Build validation dataloader
    # ------------------------------------------------------------------
    data_dict = check_det_dataset(args.data)
    gs = max(int(float_model.stride.max()), 32)

    val_bs = 64
    val_dataset = build_yolo_dataset(
        argparse.Namespace(task="detect", data=args.data, imgsz=args.imgsz, batch=val_bs,
                           workers=0, fraction=1.0, augment=False, erasing=0.0,
                           flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                           degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                           perspective=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
                           copy_paste=0.0, auto_augment=None, single_cls=False, classes=None,
                           overlap_mask=False, mask_ratio=4, rect=True, cache=False),
        data_dict["val"], val_bs, data_dict, mode="val", rect=True, stride=gs,
    )
    val_loader = build_dataloader(val_dataset, batch=val_bs, workers=0,
                                   shuffle=False, rank=-1, drop_last=False)

    # ------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------
    cfg = copy.deepcopy(DEFAULT_CFG_DICT)
    cfg.update({"task": "detect", "mode": "val", "data": args.data, "imgsz": args.imgsz,
                 "batch": val_bs, "device": args.device, "workers": 0, "split": "val",
                 "end2end": False, "conf": 0.001, "iou": 0.7, "max_det": 300,
                 "half": False, "plots": False, "save_json": False, "save_hybrid": False})

    validator = DetectionValidator(dataloader=val_loader, args=cfg)
    fake_trainer = FakeTrainer(float_model, prepared, data_dict, device)

    # --- DIAGNOSTIC: hook observer values through validator chain ---
    _orig_prep = BaseValidator._prepare_pt2e_model_for_eval
    def _hooked_prep(model):
        scale_before = {}
        for name, buf in model.named_buffers():
            if name.endswith(".scale") and buf.numel() == 1:
                scale_before[name] = buf.item()
        result = _orig_prep(model)
        # Check if any scales changed
        changed = 0
        for name, buf in result.named_buffers():
            if name.endswith(".scale") and buf.numel() == 1:
                if name in scale_before:
                    if abs(buf.item() - scale_before[name]) > 1e-8:
                        changed += 1
        log(f"  [hook] _prepare_pt2e_model_for_eval: {changed}/{len(scale_before)} observer scales changed")
        if changed > 0:
            log(f"  [hook] model.float() copy appears to alter observer values!")
        return result
    BaseValidator._prepare_pt2e_model_for_eval = staticmethod(_hooked_prep)

    log("\n[step 6] Running one2many validation ...")
    import time as _time
    t0 = _time.time()
    results = validator(trainer=fake_trainer)
    elapsed = _time.time() - t0

    mAP50 = results.get("metrics/mAP50(B)", 0)
    mAP50_95 = results.get("metrics/mAP50-95(B)", 0)

    # ------------------------------------------------------------------
    # 7. Final comparison
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"  eval.py one2many epoch{ckpt_epoch}:")
    log(f"    mAP50:    {mAP50:.4f}")
    log(f"    mAP50-95: {mAP50_95:.4f}")
    if train_mAP != "?":
        delta = mAP50_95 - train_mAP
        status = "MATCH" if abs(delta) < 0.005 else f"DELTA = {delta:+.4f}"
        log(f"  training internal val: {train_mAP:.4f}  ({status})")
    log(f"  validation time: {elapsed:.0f}s")
    log("=" * 70)
    log(f"\nFull debug log saved to: {DEBUG_LOG}")


if __name__ == "__main__":
    main()
# 39.62