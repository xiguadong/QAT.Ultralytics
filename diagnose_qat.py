"""Direct QAT vs Float model validation on COCO val."""

import sys
from copy import deepcopy
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import torch.ao.quantization as aoq

from ultralytics import YOLO
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model

DEVICE = 0


def main():
    print("=" * 60)
    print("[1] Load models...")
    m = YOLO("yolo26n.pt")
    float_eager = m.model.float().to(DEVICE)
    head = float_eager.model[-1]

    # QAT model
    float_model = deepcopy(float_eager).float().to(DEVICE).train()
    for p in float_model.parameters():
        if p.dtype.is_floating_point and not p.requires_grad:
            p.requires_grad_(True)
    _, prepared_model = prepare_pt2e_qat_model(
        float_model=float_model,
        device=DEVICE,
        config_path="config.json",
        imgsz=640,
        dynamic_batch_max=128,
    )

    # QAT in eval mode (simulating validator)
    qat_eval = deepcopy(prepared_model).float()
    try:
        aoq.allow_exported_model_train_eval(qat_eval)
    except:
        pass
    try:
        aoq.move_exported_model_to_eval(qat_eval)
    except:
        pass
    qat_eval.eval()

    # 2. Load a few COCO val images
    print("[2] Load COCO val data (small sample)...")
    import cv2

    from ultralytics.utils import yaml_load

    data_cfg = yaml_load("ultralytics/cfg/datasets/coco.yaml")
    val_path = data_cfg.get("val", "coco/val2017.txt")

    # Just check if val_path exists
    if isinstance(val_path, str):
        if val_path.endswith(".txt"):
            # Read image list
            with open(val_path) as f:
                img_paths = [line.strip() for line in f if line.strip()][:100]
        else:
            img_paths = [val_path]
    else:
        img_paths = []

    if not img_paths:
        print("  No val images found, using random input for comparison")
        img = torch.rand(1, 3, 640, 640, device=DEVICE)
        # Float inference
        float_eager.eval()
        with torch.no_grad():
            f_out = float_eager(img)
        f_inf = f_out[0]

        # QAT inference
        with torch.no_grad():
            q_out = qat_eval(img)
        pred_dict = q_out["one2one"] if "one2one" in q_out else q_out
        q_inf = head._inference(pred_dict)
        if getattr(head, "end2end", False):
            q_inf = head.postprocess(q_inf.permute(0, 2, 1))

        print(f"  Float inf shape: {f_inf.shape}, mean={f_inf.mean():.4f}")
        print(f"  QAT   inf shape: {q_inf.shape}, mean={q_inf.mean():.4f}")
        print(f"  Diff: {torch.abs(f_inf - q_inf).mean():.4f}")
        return

    # 3. Run comparison on real images
    print(f"[3] Compare predictions on {len(img_paths)} images...")
    all_float_infs = []
    all_qat_infs = []

    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        _h, _w = img.shape[:2]
        img = cv2.resize(img, (640, 640))
        img_t = (
            torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():
            f_out = float_eager(img_t)
        f_inf = f_out[0]

        with torch.no_grad():
            q_out = qat_eval(img_t)
        pred_dict = q_out["one2one"] if "one2one" in q_out else q_out
        q_inf = head._inference(pred_dict)
        if getattr(head, "end2end", False):
            q_inf = head.postprocess(q_inf.permute(0, 2, 1))

        all_float_infs.append(f_inf)
        all_qat_infs.append(q_inf)

    # 4. Compare statistics
    print("\n[4] Statistics:")
    float_cat = torch.cat(all_float_infs, dim=0)
    qat_cat = torch.cat(all_qat_infs, dim=0)

    # Box coordinates diff
    box_diff = torch.abs(float_cat[:, :, :4] - qat_cat[:, :, :4])
    score_diff = torch.abs(float_cat[:, :, 4] - qat_cat[:, :, 4])
    cls_diff = (float_cat[:, :, 5] != qat_cat[:, :, 5]).float()

    print(f"  Box   diff (mean): {box_diff.mean():.4f}")
    print(f"  Score diff (mean): {score_diff.mean():.4f}")
    print(f"  Class diff (rate): {cls_diff.mean():.4f}")

    # Check score distribution
    f_scores = float_cat[:, :, 4]
    q_scores = qat_cat[:, :, 4]
    print(f"  Float scores: mean={f_scores.mean():.4f}, std={f_scores.std():.4f}")
    print(f"  QAT   scores: mean={q_scores.mean():.4f}, std={q_scores.std():.4f}")

    # Number of valid predictions (> 0 score)
    f_valid = (f_scores > 0).sum(dim=1).float()
    q_valid = (q_scores > 0).sum(dim=1).float()
    print(f"  Float valid preds/image: {f_valid.mean():.1f}")
    print(f"  QAT   valid preds/image: {q_valid.mean():.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
