import config
import torch
import numpy as np
import torch.optim as optim
from PIL import Image, ImageFile
from model import YOLOv3
from utils import (
  load_checkpoint,
  cells_to_bboxes,
  non_max_suppression
)
from dataset import plot_image
from fastapi import FastAPI, File, UploadFile, Form
import warnings
warnings.filterwarnings("ignore")

model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
  model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

load_checkpoint('./first.model.pth.tar', model, optimizer, config.LEARNING_RATE)

app = FastAPI()

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
    bboxes[0], iou_threshold=0.000000005, threshold=0.85
  )

  prediction = {}
  for box in nms_boxes:
    class_pred = int(box[0])
    prediction[class_labels[class_pred]] = prediction.get(class_labels[class_pred], 0) + 1
  
  print('Prediction', prediction)
  plot_image(np.array(Image.open(image.file).convert("RGB")), nms_boxes)
  return prediction

def transform(image):
  augmentations = config.prod_transforms(image=image)
  return augmentations['image']


"""ENDPOINTS"""

@app.get("/ping")
def ping():
  return { "ping": "pong!" }

@app.post("/predict")
def predict_image(image: UploadFile = File(...)):
  print('DEVICE', config.DEVICE)
  print('Filename', image.filename)

  image_array = np.array(Image.open(image.file).convert("RGB"))
  image_array = transform(image_array)
  image_array = np.transpose(image_array, (2, 0, 1))
  image_tensor = torch.from_numpy(np.array([ image_array ])).to(config.DEVICE)
  return {
    "prediction": predict(image_tensor, image)
  }
