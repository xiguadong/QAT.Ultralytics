from ultralytics import YOLO

model = YOLO("yolo26n.yaml")
model.load("yolo26n.pt")

model.train(
    data="coco.yaml",
    batch=64,
    epochs=50,
    imgsz=640,
    device=2,
    project="runs/detect/",
    name="qat",
    exist_ok=True,
    qat=True,
    qat_config="config_matmul_s8.json",
    qat_validate=True,
    qat_ema=False,
    end2end=False,
    save_period=1,
    fraction=1,
    lr0=2e-5,
    lrf=0.1,
)
