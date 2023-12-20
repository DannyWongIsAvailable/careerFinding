from ultralytics import YOLO
import math
import os
import json

# 训练模型
model = YOLO('../user_data/model_data/yolov8s-pose.pt')  # load a pretrained model (recommended for training)

results = model.train(data='../user_data/tmp_data/auto.yaml', pretrained=True, epochs=100, imgsz=512, batch=24,
                      close_mosaic=100, device=0, project='/user_data/model_data', name='model')
