
import os
import shutil
import json
import glob

model_list = glob.glob('yolov8*.pt')

target_path = 'yolov8/'

from ultralytics import YOLO

if False:
  # Openvino can read the regular onnx files just fine so no need to export these "again"
  for model_name in model_list:
    for model_size in [320, 640]:
      name = os.path.splitext(model_name)[0]
      result = f'{target_path}{name}_{model_size}x{model_size}_f32_openvino'
      print(f"Exporting {model_name} into {result}", flush=True)
      base = f'{name}_openvino_model'
      if not os.path.exists(result) or os.path.getsize(result) == 0:
        model = YOLO(model_name)
        model.export(format="openvino", imgsz=[model_size, model_size])
        os.rename(base, result)
        os.unlink(f'{name}.onnx')

for model_name in model_list:
  for model_size in [320, 640]:
    name = os.path.splitext(model_name)[0]
    result = f'{target_path}{name}_{model_size}x{model_size}_i8_openvino'
    print(f"Exporting {model_name} into {result}", flush=True)
    base = f'{name}_int8_openvino_model'
    if not os.path.exists(result) or os.path.getsize(result) == 0:
      model = YOLO(model_name)
      model.export(format="openvino", imgsz=[model_size, model_size], int8=True)
      os.rename(base, result)
      os.unlink(f'{name}.onnx')
