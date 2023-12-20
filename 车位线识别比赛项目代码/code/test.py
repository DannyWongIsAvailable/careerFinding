import math
import os
from ultralytics import YOLO
import json

# Load a model
model = YOLO('../user_data/model_data/close100bc24.pt')  # pretrained YOLOv8n model

# 指定图片路径和保存 JSON 文件的路径
# image_dir = '../img_test'
image_dir = 'F:\YOLOv8\img_test'
# image_dir = '/work/res_data/visual-parking-space-line-recognition-test-set'
json_output_dir = '../prediction_result'
# json_output_dir = '../output/'

# 确保保存 JSON 文件的路径存在
if not os.path.exists(json_output_dir):
    os.makedirs(json_output_dir)

for img in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img)
    # 使用模型预测图像
    results = model(img_path)
    # results = model(img_path)
    # print(results)
    json_data = []

    for i in range(len(results[0])):
        park = {}
        points = []
        keypoints = results[0][i].keypoints  # Masks object
        # 获取第一和第二个点的坐标
        x1, y1 = keypoints.xy[0, 0]
        x2, y2 = keypoints.xy[0, 1]

        # 获取第三和第四个点的坐标
        x3, y3 = keypoints.xy[0, 2]
        # x4, y4 = keypoints.xy[0, 3]
        point1 = [x1.item(), y1.item()]
        point2 = [x2.item(), y2.item()]
        points.extend([point1, point2])

        park["points"] = points

        # 计算角度（弧度）
        angle_13_radians = math.atan2((y1 - y3), (x1 - x3))
        # angle_24_radians = math.atan2((y2 - y4), (x2 - x4))
        angle_24_radians = angle_13_radians

        park["angle1"] = angle_13_radians
        park["angle2"] = angle_24_radians

        boxes = results[0].boxes  # 用于边界框输出的 Boxes 对象
        confidence = boxes[i].conf.item()  # 获取置信度
        park["scores"] = round(confidence, 2)  # 保留2位小数

        json_data.append(park)
        json_data.reverse()  # 车位倒序排列
    slot = {"slot": json_data}

    # 根据图片文件名生成对应的 JSON 文件名
    json_file_name = img.replace(".jpg", ".json")

    # 拼接保存 JSON 文件的完整路径
    json_file_path = os.path.join(json_output_dir, json_file_name)

    # 将字典保存为 JSON 文件
    with open(json_file_path, "w") as json_file:
        json.dump(slot, json_file, indent=2, ensure_ascii=False)

    print(f"字典已成功保存为 {json_file_path} 文件。")
