from ultralytics import YOLO

model = YOLO("yolo26n-seg.yaml")
model.load("./weights/yolo26n-seg.pt")

model.train(
    data="coco-seg.yaml",
    batch=64,
    epochs=50,
    imgsz=640,
    device=2,
    project="runs/segment/",
    name="exp1",
    exist_ok=True,
    qat=True,
    qat_config="config_matmul_s8.json",
    qat_validate=True,
    end2end=True,
    save_period=1,
    fraction=1,
    lr0=2e-5,
    lrf=0.1,
)
