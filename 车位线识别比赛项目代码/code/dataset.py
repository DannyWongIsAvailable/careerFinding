import shutil

import cv2
import numpy as np
from tqdm import tqdm
import json
import math
import os
import random


# 读取并添加数据增强后对应json文件
class ImgTransform:
    def __init__(self, img_fold, mode):
        self.img_fold = img_fold
        self.mode = mode

    @staticmethod
    def clear_folder(folder_path):
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print("文件夹不存在！")
            return

        # 遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(folder_path, topdown=False):
            # 删除文件
            for file_name in files:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)

            # 删除子文件夹
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)

        print("文件夹已成功清空！")

    @staticmethod
    def merge_folders(source_folders, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for folder in source_folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_folder, file)
                    shutil.copy2(source_path, destination_path)

    @staticmethod
    def delete_folders(folders):
        for folder in folders:
            shutil.rmtree(folder)

    # 获取标签
    @staticmethod
    def label_load(data_path):
        json_file_path = data_path.replace('imgs', 'json').replace('.jpg', '.json')
        with open(json_file_path, 'r') as file:
            json_data = file.read()
            label = json.loads(json_data)
        return label

    # 处理标签
    @staticmethod
    def label_transform(x, y, x_offset, y_offset, center_x, center_y, angle):
        angle_rad = -math.radians(angle)
        x_trans = x - center_x + x_offset
        y_trans = y - center_y + y_offset
        x_final = x_trans * math.cos(angle_rad) - y_trans * math.sin(angle_rad) + center_x
        y_final = x_trans * math.sin(angle_rad) + y_trans * math.cos(angle_rad) + center_y
        return x_final, y_final

    # 筛选倾斜车位图片
    def obl_img(self):
        img_choice = []
        for path in os.listdir(self.img_fold):
            img_path = os.path.join(self.img_fold, path)
            label = self.label_load(img_path)
            for item in label['slot']:
                category = item['category']
                if category == 2.0 or category == 3.0:
                    img_choice.append(img_path)
        # print(len(img_choice), "张倾斜车位图片")
        return img_choice

    # 筛选平行垂直车位图片
    def hv_img(self):
        img_choice = []
        for path in os.listdir(self.img_fold):
            img_path = os.path.join(self.img_fold, path)
            label = self.label_load(img_path)
            for item in label['slot']:
                category = item['category']
                if category != 2.0 and category != 3.0:
                    img_choice.append(img_path)
        # print(len(img_choice), "张垂直车位图片")
        return img_choice

    def process_images1(self):
        img = self.obl_img()
        return img

    def process_images2(self):
        img = self.hv_img()
        # 从列表中随机选择n张图像文件
        img = random.sample(img, 2000)
        return img

    def process_images3(self):
        img = self.hv_img()
        return img


# 翻转操作
class Flip(ImgTransform):
    def flip_plus(self, img):
        original_image = cv2.imread(img)
        # 定义变化变量
        x_offset, y_offset = 0, 0
        angle = 0

        matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), angle, 1)
        matrix[:, 2] += [x_offset, y_offset]  # 平移
        changed = cv2.warpAffine(original_image, matrix, (original_image.shape[1], original_image.shape[0]))

        # 获取中心坐标
        center_x = changed.shape[1] // 2
        center_y = changed.shape[0] // 2

        label = self.label_load(img)
        for item in label['slot']:
            points = item['points']
            angle1 = item['angle1']
            angle2 = item['angle2']

            new_points = []
            new_keypoints = []

            # 车位点
            for point in points:
                x, y = point
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_points.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (255, 0, 0), -1)

            # 关键点
            for point in points:
                x, y = [point[0] - 24 * math.cos(angle1), point[1] - 24 * math.sin(angle2)]
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_keypoints.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (0, 255, 0), -1)

            # 计算角度（弧度制）
            dx_13 = new_points[0][0] - new_keypoints[0][0]
            dy_13 = new_points[0][1] - new_keypoints[0][1]
            angle_13 = math.atan2(dy_13, dx_13)

            # 更新标注数据中的点坐标和角度
            dx_24 = new_points[1][0] - new_keypoints[1][0]
            dy_24 = new_points[1][1] - new_keypoints[1][1]
            angle_24 = math.atan2(dy_24, dx_24)

            item['points'] = new_points
            item['angle1'] = angle_13
            item['angle2'] = angle_24

        # 随机决定是否进行垂直或水平翻转
        flip_direction = random.choice(['horizontal', 'vertical', 'none'])

        if flip_direction == 'horizontal':
            changed = np.fliplr(changed)
            for item in label['slot']:
                for point in item['points']:
                    point[0] = changed.shape[1] - point[0]  # 更新x坐标
                item['angle1'] = math.pi - item['angle1']  # 更新角度1
                item['angle2'] = math.pi - item['angle2']  # 更新角度2

        elif flip_direction == 'vertical':
            changed = np.flipud(changed)
            for item in label['slot']:
                for point in item['points']:
                    point[1] = changed.shape[0] - point[1]  # 更新y坐标
                item['angle1'] = -item['angle1']  # 更新角度1
                item['angle2'] = -item['angle2']  # 更新角度2

        # 保存图片和标签

        save_img_path = img.replace('res_data', 'data_obl_flip').replace('.jpg', '_img_flip.jpg')
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(save_img_path))
        counter = 1
        new_save_path = save_img_path

        while os.path.exists(new_save_path):
            new_save_path = f"{os.path.dirname(save_img_path)}/{base_name}_{counter}{extension}"
            counter += 1

        cv2.imwrite(new_save_path, changed)
        #
        # 更新后的标注数据保存为新的 JSON 文件
        new_json_file_path = save_img_path.replace('imgs', 'json').replace('.jpg', '.json')
        os.makedirs(os.path.dirname(new_json_file_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(new_json_file_path))
        counter = 1
        new_json_file_path_with_counter = new_json_file_path

        while os.path.exists(new_json_file_path_with_counter):
            new_json_file_path_with_counter = f"{os.path.dirname(new_json_file_path)}/{base_name}_{counter}{extension}"
            counter += 1

        with open(new_json_file_path_with_counter, 'w') as new_file:
            json.dump(label, new_file, indent=2)

    def main(self):
        if self.mode == 'obl':
            img_file_list = self.process_images1()
            for img in tqdm(img_file_list, desc="倾斜车位图片翻转", unit="张"):
                self.flip_plus(img)
        if self.mode == 'hv':
            img_file_list = self.process_images2()
            for img in tqdm(img_file_list, desc="垂直车位图片翻转", unit="张"):
                self.flip_plus(img)
        if self.mode == 'hv2':
            img_file_list = self.process_images3()
            for img in tqdm(img_file_list, desc="垂直车位图片翻转", unit="张"):
                self.flip_plus(img)


# 平移操作
class Movement(ImgTransform):
    def movement_plus(self, img_path):

        original_image = cv2.imread(img_path)

        x_offset = random.randint(-int(original_image.shape[1] * 0.05), int(original_image.shape[1] * 0.05))
        y_offset = random.randint(-int(original_image.shape[0] * 0.05), int(original_image.shape[0] * 0.05))
        # x_offset, y_offset = 0, 0
        # angle = random.uniform(-10, 10)
        angle = 0

        matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), angle, 1)
        matrix[:, 2] += [x_offset, y_offset]
        changed = cv2.warpAffine(original_image, matrix, (original_image.shape[1], original_image.shape[0]))

        center_x = changed.shape[1] // 2
        center_y = changed.shape[0] // 2

        label = self.label_load(img_path)
        for item in label['slot']:
            points = item['points']
            angle1 = item['angle1']
            angle2 = item['angle2']

            new_points = []
            new_keypoints = []

            # 车位点
            for point in points:
                x, y = point
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_points.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (255, 0, 0), -1)

            # 关键点
            for point in points:
                x, y = [point[0] - 24 * math.cos(angle1), point[1] - 24 * math.sin(angle2)]
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_keypoints.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (0, 255, 0), -1)

            # 计算角度（弧度制）
            dx_13 = new_points[0][0] - new_keypoints[0][0]
            dy_13 = new_points[0][1] - new_keypoints[0][1]
            angle_13 = math.atan2(dy_13, dx_13)

            # 更新标注数据中的点坐标和角度
            dx_24 = new_points[1][0] - new_keypoints[1][0]
            dy_24 = new_points[1][1] - new_keypoints[1][1]
            angle_24 = math.atan2(dy_24, dx_24)

            item['points'] = new_points
            item['angle1'] = angle_13
            item['angle2'] = angle_24

        # 保存图片和标签
        save_img_path = img_path.replace('data_obl_flip', 'data_obl_m').replace('.jpg', '_m.jpg')
        # save_img_path = img_path.replace('.jpg', '_m.jpg')
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(save_img_path))
        counter = 1
        new_save_path = save_img_path

        while os.path.exists(new_save_path):
            new_save_path = f"{os.path.dirname(save_img_path)}/{base_name}_{counter}{extension}"
            counter += 1

        cv2.imwrite(new_save_path, changed)
        #
        # 更新后的标注数据保存为新的 JSON 文件
        new_json_file_path = save_img_path.replace('imgs', 'json').replace('.jpg', '.json')
        os.makedirs(os.path.dirname(new_json_file_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(new_json_file_path))
        counter = 1
        new_json_file_path_with_counter = new_json_file_path

        while os.path.exists(new_json_file_path_with_counter):
            new_json_file_path_with_counter = f"{os.path.dirname(new_json_file_path)}/{base_name}_{counter}{extension}"
            counter += 1

        with open(new_json_file_path_with_counter, 'w') as new_file:
            json.dump(label, new_file, indent=2)

    def main(self):
        if self.mode == 'obl':
            img_file_list = self.process_images1()
            for img in tqdm(img_file_list, desc="倾斜车位图片平移", unit="张"):
                self.movement_plus(img)
        if self.mode == 'hv':
            img_file_list = self.process_images2()
            for img in tqdm(img_file_list, desc="垂直车位图片平移", unit="张"):
                self.movement_plus(img)
        if self.mode == 'hv2':
            img_file_list = self.process_images3()
            for img in tqdm(img_file_list, desc="垂直车位图片平移", unit="张"):
                self.movement_plus(img)


# 旋转操作
class Rotation(ImgTransform):
    def rotation_plus(self, img_path):
        # img_path = img_path.replace('obl_flip', 'obl_m')
        original_image = cv2.imread(img_path)

        x_offset = random.randint(-int(original_image.shape[1] * 0.05), int(original_image.shape[1] * 0.05))
        y_offset = random.randint(-int(original_image.shape[0] * 0.05), int(original_image.shape[0] * 0.05))
        x_offset, y_offset = 0, 0
        angle = random.uniform(-10, 10)
        # angle = 0

        matrix = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), angle, 1)
        matrix[:, 2] += [x_offset, y_offset]
        changed = cv2.warpAffine(original_image, matrix, (original_image.shape[1], original_image.shape[0]))

        center_x = changed.shape[1] // 2
        center_y = changed.shape[0] // 2

        label = self.label_load(img_path)
        for item in label['slot']:
            points = item['points']
            angle1 = item['angle1']
            angle2 = item['angle2']

            new_points = []
            new_keypoints = []

            # 车位点
            for point in points:
                x, y = point
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_points.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (255, 0, 0), -1)

            # 关键点
            for point in points:
                x, y = [point[0] - 24 * math.cos(angle1), point[1] - 24 * math.sin(angle2)]
                x_transform, y_transform = self.label_transform(x, y, x_offset, y_offset, center_x, center_y, angle)
                new_keypoints.append([x_transform, y_transform])
                # cv2.circle(changed, (int(x_transform), int(y_transform)), 3, (0, 255, 0), -1)

            # 计算角度（弧度制）
            dx_13 = new_points[0][0] - new_keypoints[0][0]
            dy_13 = new_points[0][1] - new_keypoints[0][1]
            angle_13 = math.atan2(dy_13, dx_13)

            # 更新标注数据中的点坐标和角度
            dx_24 = new_points[1][0] - new_keypoints[1][0]
            dy_24 = new_points[1][1] - new_keypoints[1][1]
            angle_24 = math.atan2(dy_24, dx_24)

            item['points'] = new_points
            item['angle1'] = angle_13
            item['angle2'] = angle_24

        # 保存图片和标签
        save_img_path = img_path.replace('data_obl_m', 'data_plus').replace('.jpg', '_r.jpg')
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(save_img_path))
        counter = 1
        new_save_path = save_img_path

        while os.path.exists(new_save_path):
            new_save_path = f"{os.path.dirname(save_img_path)}/{base_name}_{counter}{extension}"
            counter += 1

        cv2.imwrite(new_save_path, changed)

        # 更新后的标注数据保存为新的 JSON 文件
        new_json_file_path = save_img_path.replace('imgs', 'json').replace('.jpg', '.json')
        os.makedirs(os.path.dirname(new_json_file_path), exist_ok=True)  # 创建保存路径

        # 处理可能的同名文件情况
        base_name, extension = os.path.splitext(os.path.basename(new_json_file_path))
        counter = 1
        new_json_file_path_with_counter = new_json_file_path

        while os.path.exists(new_json_file_path_with_counter):
            new_json_file_path_with_counter = f"{os.path.dirname(new_json_file_path)}/{base_name}_{counter}{extension}"
            counter += 1

        with open(new_json_file_path_with_counter, 'w') as new_file:
            json.dump(label, new_file, indent=2)

    def main(self):
        if self.mode == 'obl':
            img_file_ist = self.process_images1()
            for img in tqdm(img_file_ist, desc="倾斜车位图片旋转", unit="张"):
                self.rotation_plus(img)
        if self.mode == 'hv':
            img_file_ist = self.process_images2()
            for img in tqdm(img_file_ist, desc="垂直车位图片旋转", unit="张"):
                self.rotation_plus(img)
        if self.mode == 'hv2':
            img_file_list = self.process_images3()
            for img in tqdm(img_file_list, desc="垂直车位图片旋转", unit="张"):
                self.rotation_plus(img)


if __name__ == '__main__':

    # 调用函数清空指定的文件夹
    folder_path = "../user_data/tmp_data"  # 替换为要清空的文件夹路径
    ImgTransform.clear_folder(folder_path)

    # 数据集文件夹路径
    data_fold = '../xfdata/train'
    # 目标数据集文件夹路径
    target_dataset_folder = '../user_data/tmp_data/res_data'

    print("创建副本")
    # 若目标文件夹已存在，先删除再复制
    if os.path.exists(target_dataset_folder):
        shutil.rmtree(target_dataset_folder)

    # 使用shutil库中的copytree函数复制整个文件夹
    shutil.copytree(data_fold, target_dataset_folder)

    print("文件拷贝完成")
    # 对平行车位操作
    img_fold = target_dataset_folder + "/imgs"
    Flip(img_fold, 'obl').main()  # 倾斜车位翻转
    # 对垂直车位操作
    Flip(img_fold, 'hv').main()  # 垂直车位翻转
    img_fold = img_fold.replace('res_data', 'data_obl_flip')
    Movement(img_fold, 'obl').main()  # 倾斜车位平移
    Movement(img_fold, 'hv2').main()  # 垂直车位平移
    img_fold = img_fold.replace('data_obl_flip', 'data_obl_m')
    Movement(img_fold, 'obl').main()  # 倾斜车位平移
    Movement(img_fold, 'obl').main()  # 倾斜车位平移
    Rotation(img_fold, 'obl').main()  # 倾斜车位旋转
    Rotation(img_fold, 'hv2').main()  # 垂直车位旋转
    # img_fold = img_fold.replace('data_obl_m', 'data_plus')
    # Rotation(img_fold, 'obl').main()  # 倾斜车位旋转

    print("删除临时数据")
    # 指定要删除的文件夹列表
    folders_to_delete = ['../user_data/tmp_data/data_obl_flip', '../user_data/tmp_data/data_obl_m',
                         '../user_data/tmp_data/res_data']

    # 调用函数进行删除
    ImgTransform.delete_folders(folders_to_delete)

