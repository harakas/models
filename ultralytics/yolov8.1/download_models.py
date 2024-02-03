
from urllib.request import urlretrieve
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

base_url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/"

for model_name in model_list:
  if not os.path.exists(model_name) or os.path.getsize(model_name) == 0:
    print(f"Downloading {model_name}", flush=True)
    urlretrieve(base_url + model_name, model_name)

