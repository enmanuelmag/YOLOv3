import os
import sys
import torch
import base64
import config
import asyncio
import warnings
import numpy as np
from model import YOLOv3
import torch.optim as optim
from PIL import Image, ImageFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, Response
from utils import (
  load_checkpoint,
  cells_to_bboxes,
  non_max_suppression,
  get_image
)
warnings.filterwarnings("ignore")

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
  model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

BEST_1_MODEL = './models/best.full.fix.pth.tar'
BEST_2_MODEL = './models/best.2.full.fix.pth.tar'
BEST_LOSS = './models/best.3.full.loss.class.pth.tar'

LAST_BEST_MODEL = './last.checkpoint.full.fix.pht.tar'
LAST_LOSS_MODEL = './last.checkpoint.loss.class.pht.tar'

#1. BEST MODEL 2
#2. BEST MODEL 1
load_checkpoint(BEST_2_MODEL, model, optimizer, config.LEARNING_RATE)

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)


def predict(image_tensor, image):
  scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
  ).to(config.DEVICE)

  with torch.no_grad():
    out = model(image_tensor)
    bboxes = [[] for _ in range(image_tensor.shape[0])]
    for i in range(3):
      batch_size, A, S, _, _ = out[i].shape
      anchor = scaled_anchors[i].to(config.DEVICE)
      boxes_scale_i = cells_to_bboxes(
        out[i], anchor, S=S, is_preds=True
      )
      for idx, (box) in enumerate(boxes_scale_i):
        bboxes[idx] += box
  
  class_labels = config.USD_DIVISA_CLASSES
  nms_boxes = non_max_suppression(
    bboxes[0], iou_threshold=0.85, threshold=0.55
  )

  prediction = {}
  for box in nms_boxes:
    class_pred = int(box[0])
    prediction[class_labels[class_pred]] = prediction.get(class_labels[class_pred], 0) + 1
  
  print('Prediction', prediction)
  image_name = get_image(np.array(Image.open(image.file).convert("RGB")), nms_boxes)

  return prediction, image_name

def transform(image):
  augmentations = config.prod_transforms(image=image)
  return augmentations['image']


"""ENDPOINTS"""

@app.get("/ping")
def ping():
  return { "ping": "pong!" }

@app.post("/predict")
def predict_image(image: UploadFile = File(...)):
  print('DEVICE | Filename', config.DEVICE, image.filename)

  image_array = np.array(Image.open(image.file).convert("RGB"))
  image_array = transform(image_array)
  image_array = np.transpose(image_array, (2, 0, 1))
  image_tensor = torch.from_numpy(np.array([ image_array ])).to(config.DEVICE)
  prediction, image_filename = predict(image_tensor, image)
  
  with open(image_filename, 'rb') as f:
    image_bytes = base64.b64encode( f.read())
  os.remove(image_filename)

  return {
    "prediction": prediction,
    "image": image_bytes
  }
