from ultralytics import YOLO

model_vehicle = YOLO("yolo12s.pt")
model_lp = YOLO("best.pt")

model_vehicle.export(format="onnx")
model_lp.export(format="onnx")