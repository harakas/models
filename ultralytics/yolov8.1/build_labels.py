
from urllib.request import urlretrieve
import os
import shutil
import json

target_path = 'yolov8/'

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

with open(target_path + 'labels-frigate.txt', 'w') as f:
  for label in open('coco-labels.txt'):
    label = label.strip()
    if label in frigate_labels_set:
      f.write(label)
    else:
      f.write(coco_remap.get(label, 'other'))
    f.write("\n")

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

with open(target_path + 'labels-oiv7-frigate.txt', 'w') as f:
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

shutil.copyfile('oiv7-labels.txt', target_path + 'labels-oiv7.txt')
shutil.copyfile('coco-labels.txt', target_path + 'labels.txt')

