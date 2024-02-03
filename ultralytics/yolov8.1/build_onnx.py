
from urllib.request import urlretrieve
import os
import shutil
import json
import glob

model_list = glob.glob('yolov8*.pt')

print('Found models: ' + ', '.join(model_list))

target_path = 'yolov8/'

from ultralytics import YOLO

for model_name in model_list:
  for model_size in [320, 640]:
    base_onnx = os.path.splitext(model_name)[0] + '.onnx'
    assert not os.path.exists(base_onnx), f"{base_onnx} in the way, remove it or resolve problem"
    onnx = target_path + os.path.splitext(model_name)[0] + f'_{model_size}x{model_size}.onnx'
    if not os.path.exists(onnx) or os.path.getsize(onnx) == 0:
      print(f"Exporting {model_name} into {onnx}", flush=True)
      model = YOLO(model_name)
      model.export(format="onnx", opset=12, imgsz=[model_size, model_size])
      os.rename(base_onnx, onnx)

