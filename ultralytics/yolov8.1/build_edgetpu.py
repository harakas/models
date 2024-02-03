
import os
import shutil
import json
import glob

model_list = glob.glob('yolov8*.pt')
target_path = 'yolov8/'

from ultralytics import YOLO

for model_name in model_list:
  for model_size in [320]:
    name = os.path.splitext(model_name)[0]
    result = f'{target_path}{name}_{model_size}x{model_size}_edgetpu.tflite'
    print(f"Exporting {model_name} into {result}")
    base = f'{name}_saved_model/{name}_full_integer_quant_edgetpu.tflite'
    if not os.path.exists(result) or os.path.getsize(result) == 0:
      model = YOLO(model_name)
      model.export(format="edgetpu", imgsz=[model_size, model_size], int8=True)
      os.rename(base, result)

