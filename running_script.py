from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source='fire_5.jpg', imgsz=640, conf=0.5, save=True)