from ultralytics import YOLO

vehicle_model = YOLO("/content/yolov12/yolov12n.pt")
vehicle_model.train(
    data="/content/data_vehicle.yaml",  
    epochs=30,                          
    imgsz=640,
    batch=16,
    name="yolov12n_vehicle_training",
    verbose=True
)

license_model = YOLO("/content/yolov12/yolov12n.pt")
license_model.train(
    data="/content/data_license.yaml",  
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolov12n_license_training",
    verbose=True
)