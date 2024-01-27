
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
    print(f"Downloading {model_name}")
    urlretrieve(base_url + model_name, model_name)

frigate_labels = [
  'person',
  'car',
  'motorcycle',
  'other_vehicle',
  'cat',
  'dog',
  'animal',
  'bird',
  'other',
]

frigate_labels_set = set(frigate_labels)

coco_remap = {
  'bus': 'other_vehicle',
  'train': 'other_vehicle',
  'truck': 'car',
  'horse': 'animal',
  'sheep': 'animal',
  'cow': 'animal',
  'elephant': 'animal',
  'bear': 'animal',
  'zebra': 'animal',
  'giraffe': 'animal',
}

with open('coco-labels-frigate.txt', 'w') as f:
  for label in open('coco-labels.txt'):
    label = label.strip()
    if label in frigate_labels_set:
      f.write(label)
    else:
      f.write(coco_remap.get(label, 'other'))
    f.write("\n")


if False:
  oiv7_remap = {
    'Alpaca': 'animal',
    'Animal': 'animal',
    'Ant': 'animal',
    'Antelope': 'animal',
    'Armadillo': 'animal',
    'Auto': 'auto'
  }


oiv7_struct = json.load(open('oiv7/bbox_labels_600_hierarchy.json', 'r'))
oiv7_key2label = {}
oiv7_label2key = {}
for cls in open('oiv7/class-descriptions-boxable.csv', 'r'):
  key, label = cls.strip().split(',')
  assert key not in oiv7_key2label
  oiv7_key2label[key] = label
  assert label not in oiv7_label2key
  oiv7_label2key[label] = key

oiv7_label2long = {}

def traverse(parent_path, obj):
  for sub in obj['Subcategory']:
    key = sub['LabelName']
    label = oiv7_key2label[key]
    if parent_path == '':
      path = oiv7_key2label[key]
    else:
      path = parent_path + '/' + oiv7_key2label[key]
    oiv7_label2long[label] = path
    if 'Subcategory' in sub:
      traverse(path, sub)

traverse('', oiv7_struct)

oiv7_groupings = [
  ('Person', 'person'),
  ('Animal/Mammal/Carnivore/Cat', 'cat'),
  ('Animal/Mammal/Carnivore/Dog', 'dog'),
  ('Animal/Bird', 'bird'),
  ('Animal', 'animal'),
  ('Animal', 'animal'),
  ('Vehicle/Land vehicle/Car', 'car'),
  ('Vehicle/Land vehicle/Truck', 'car'),
  ('Vehicle/Land vehicle/Ambulance', 'car'),
  ('Vehicle/Land vehicle/Taxi', 'car'),
  ('Vehicle/Land vehicle/Motorcycle', 'motorcycle'),
  ('Vehicle/Land vehicle', 'other_vehicle'),
]

with open('oiv7-labels-frigate.txt', 'w') as f:
  for label in open('oiv7-labels.txt'):
    label = label.strip()
    assert label in oiv7_label2long, label
    full_label = oiv7_label2long[label]
    found = False
    for prefix, target in oiv7_groupings:
      if full_label.startswith(prefix):
        f.write(target)
        found = True
        break
    if not found:
      f.write('other')
    f.write('\n')

from ultralytics import YOLO

for model_name in model_list:
  for model_size in [320, 640]:
    base_onnx = os.path.splitext(model_name)[0] + '.onnx'
    assert not os.path.exists(base_onnx), f"{base_onnx} in the way, remove it or resolve problem"
    onnx = os.path.splitext(model_name)[0] + f'_{model_size}x{model_size}.onnx'
    if not os.path.exists(onnx) or os.path.getsize(onnx) == 0:
      print(f"Exporting {model_name} into {onnx}")
      model = YOLO(model_name)
      model.export(format="onnx", opset=12, imgsz=[model_size, model_size])
      os.rename(base_onnx, onnx)
  if 'oiv7' in model_name:
    labels = 'oiv7-labels.txt'
    frigate_labels = 'oiv7-labels-frigate.txt'
  else:
    labels = 'coco-labels.txt'
    frigate_labels = 'coco-labels-frigate.txt'
  target_labels = os.path.splitext(model_name)[0] + "_labels.txt"
  if not os.path.exists(target_labels) or os.path.getsize(target_labels) == 0:
    shutil.copyfile(labels, target_labels)
  target_labels = os.path.splitext(model_name)[0] + "_labels-frigate.txt"
  if not os.path.exists(target_labels) or os.path.getsize(target_labels) == 0:
    shutil.copyfile(frigate_labels, target_labels)

