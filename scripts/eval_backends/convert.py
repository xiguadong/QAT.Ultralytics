#!/usr/bin/env python3
"""Evaluate a QAT checkpoint after convert_pt2e using real Q/DQ operators."""
import argparse
import copy
import json
from types import SimpleNamespace

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
from ultralytics.utils.torch_utils import select_device
import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401
import warnings
warnings.filterwarnings("ignore")


class FakeTrainer:
    def __init__(self, float_model, qat_model, data_dict, device, end2end):
        self.model = float_model; self.qat_model = qat_model; self.device = device
        self.data = data_dict; self.ema = None; self.amp = False
        self.loss_items = torch.zeros(3); self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.world_size = 1; self.epoch = 0; self.epochs = 1
        from collections import namedtuple
        self.stopper = namedtuple("S", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(half=False, amp=False, compile=False, plots=False, end2end=end2end,
                                       conf=0.001, iou=0.7, max_det=300, single_cls=False,
                                       agnostic_nms=False, save_json=False, save_hybrid=False)

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            return dict(zip([f"{prefix}/{x}" for x in self.loss_names], [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


p = argparse.ArgumentParser(description="Evaluate a converted PT2E QAT detection model")
p.add_argument("--ckpt", required=True)
p.add_argument("--quant-config", dest="quant_config", required=True)
p.add_argument("--model", default="yolo26n.yaml")
p.add_argument("--pretrained", default="yolo26n.pt")
p.add_argument("--data", default="coco.yaml")
p.add_argument("--end2end", default="True")
p.add_argument("--rect", default="False")
p.add_argument("--pycoco", default="True")
p.add_argument("--device", default="cuda:0")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()
requested_e2e = a.end2end.lower() == "true"
rect = a.rect.lower() == "true"
pycoco = a.pycoco.lower() == "true"
device = select_device(a.device)

m = YOLO(a.model, task="detect").load(a.pretrained)
fm = m.model.float().to(device); fm.train()
fm.model[-1].end2end = requested_e2e
e2e = bool(fm.model[-1].end2end)
if e2e != requested_e2e:
    print(f"[convert] requested end2end={requested_e2e}, but {type(fm.model[-1]).__name__} has no one2one head; use end2end={e2e}", flush=True)
hyp = dict(DEFAULT_CFG_DICT, **fm.args) if isinstance(fm.args, dict) else {}
hyp.setdefault("box", 7.5); hyp.setdefault("cls", 0.5); hyp.setdefault("dfl", 1.5)
fm.args = SimpleNamespace(**hyp)
_, prepared = prepare_pt2e_qat_model(
    float_model=fm,
    device=device,
    config_path=a.quant_config,
    imgsz=a.imgsz,
    dynamic_batch_max=128,
)
fm.criterion = fm.init_criterion()

ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
assert qat_state is not None, "checkpoint has no qat_ema/qat_model/model state"
rep = prepared.load_state_dict(qat_state, strict=False)
print(f"[load] missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}", flush=True)
assert not rep.missing_keys, f"observer mismatch: missing={len(rep.missing_keys)}"

# ★ convert_pt2e：训练内 fake-quant → 真实 Q/DQ
prepared.eval()
converted = convert_pt2e(prepared).to(device)
bn = sum(1 for n in converted.graph.nodes if "batch_norm" in str(n.target).lower())
print(f"[convert] 真实量化模型，残留 BN={bn}", flush=True)

data_dict = check_det_dataset(a.data)
gs = max(int(fm.stride.max()), 32)
ns = argparse.Namespace(task="detect", data=a.data, imgsz=a.imgsz, batch=a.batch, workers=a.workers, fraction=1.0,
                        augment=False, erasing=0.0, flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, mosaic=0.0,
                        mixup=0.0, cutmix=0.0, copy_paste=0.0, auto_augment=None, single_cls=False,
                        classes=None, overlap_mask=False, mask_ratio=4, rect=rect, cache=False)
vds = build_yolo_dataset(ns, data_dict["val"], a.batch, data_dict, mode="val", rect=rect, stride=gs)
vl = build_dataloader(vds, batch=a.batch, workers=a.workers, shuffle=False, rank=-1, drop_last=False)
cfg = copy.deepcopy(DEFAULT_CFG_DICT)
cfg.update({"task": "detect", "mode": "val", "data": a.data, "imgsz": a.imgsz, "batch": a.batch,
            "device": a.device, "workers": a.workers, "split": "val", "end2end": e2e, "conf": 0.001,
            "iou": 0.7, "max_det": 300, "half": False, "plots": False,
            "save_json": pycoco, "save_hybrid": False})
validator = DetectionValidator(dataloader=vl, args=cfg)
ft = FakeTrainer(fm, converted, data_dict, device, e2e)
res = validator(trainer=ft)
print(f"\n>>> convert 真实量化 mAP50-95={res.get('metrics/mAP50-95(B)',0):.4f}  mAP50={res.get('metrics/mAP50(B)',0):.4f}", flush=True)
if pycoco and getattr(validator, "jdict", None):
    validator.save_dir.mkdir(parents=True, exist_ok=True)
    predictions = validator.save_dir / "predictions.json"
    with predictions.open("w", encoding="utf-8") as f:
        json.dump(validator.jdict, f)
    coco_results = validator.eval_json(dict(res))
    print(f">>> pycocotools mAP50-95={coco_results.get('metrics/mAP50-95(B)', 0):.4f}", flush=True)
print("EVAL_CONVERT_DONE", flush=True)
