
from urllib.request import urlretrieve
import os
import shutil
import json

target_path = 'yolov8/'

frigate_labels = [
  'person',
  'vehicle',
  'animal',
  'bird',
  'other',
]

frigate_labels_set = set(frigate_labels)

coco_remap = {
  'car': 'vehicle',
  'bus': 'vehicle',
  'train': 'vehicle',
  'truck': 'vehicle',
  'motorcycle': 'vehicle',
  'cat': 'animal',
  'dog': 'animal',
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
      print(f"{label} -> {label}")
      f.write(label)
    else:
      print(f"{label} -> {coco_remap.get(label, 'other')}")
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
  ('Animal/Bird', 'bird'),
  ('Animal', 'animal'),
  ('Vehicle/Land vehicle', 'vehicle'),
]

with open(target_path + 'labels-oiv7-frigate.txt', 'w') as f:
  for label in open('oiv7-labels.txt'):
    label = label.strip()
    assert label in oiv7_label2long, label
    full_label = oiv7_label2long[label]
    found = False
    for prefix, target in oiv7_groupings:
      if full_label == prefix or full_label.startswith(prefix + '/'):
        print(full_label + ' -> ' + target)
        f.write(target)
        found = True
        break
    if not found:
      f.write('other')
    f.write('\n')

shutil.copyfile('oiv7-labels.txt', target_path + 'labels-oiv7.txt')
shutil.copyfile('coco-labels.txt', target_path + 'labels.txt')

