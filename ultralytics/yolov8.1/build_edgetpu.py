
import os
import shutil
import json

model_list = [
  "yolov8n.pt", 
  "yolov8s.pt", 
  "yolov8m.pt", 
  "yolov8n-oiv7.pt", 
  "yolov8s-oiv7.pt", 
]

from ultralytics import YOLO

for model_name in model_list:
  for model_size in [320]:
    name = os.path.splitext(model_name)[0]
    result = f'{name}_{model_size}x{model_size}_edgetpu.tflite'
    print(f"Exporting {model_name} into {result}")
    base = f'{name}_saved_model/{name}_full_integer_quant_edgetpu.tflite'
    if not os.path.exists(result) or os.path.getsize(result) == 0:
      model = YOLO(model_name)
      model.export(format="edgetpu", imgsz=[model_size, model_size], int8=True)
      os.rename(base, result)
