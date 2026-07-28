from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch YOLO26 detect PT2E QAT training from the local workspace.")
    parser.add_argument("--model", default="yolo26n.pt", help="Model checkpoint or YAML path.")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/coco.yaml", help="Dataset YAML path.")
    parser.add_argument("--device", default="1", help="Training device, e.g. 'cpu', '0', or '0,1'.")
    parser.add_argument("--project", default="runs/detect", help="Training project directory.")
    parser.add_argument("--name", default="qat-coco26n", help="Training run name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs if provided.")
    parser.add_argument("--batch", type=float, default=None, help="Override batch if provided.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override imgsz if provided.")
    parser.add_argument("--workers", type=int, default=None, help="Override workers if provided.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing run directory.")
    parser.add_argument("--qat-validate", action="store_true", help="Enable online validation on the PT2E QAT graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_kwargs = {
        "data": args.data,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "qat": True,
        "qat_validate": args.qat_validate,
    }
    if args.epochs is not None:
        train_kwargs["epochs"] = args.epochs
    if args.batch is not None:
        train_kwargs["batch"] = args.batch
    if args.imgsz is not None:
        train_kwargs["imgsz"] = args.imgsz
    if args.workers is not None:
        train_kwargs["workers"] = args.workers

    model = YOLO(args.model)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
