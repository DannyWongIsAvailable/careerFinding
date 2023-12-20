import json
import os


def label_load(data_path):
    json_file_path = data_path.replace('imgs', 'json').replace('.jpg', '.json')
    with open(json_file_path, 'r') as file:
        json_data = file.read()
        label = json.loads(json_data)
    return label


def obl_img(img_fold):
    img_choice = []
    for path in os.listdir(img_fold):
        img_path = os.path.join(img_fold, path)
        label = label_load(img_path)
        for item in label['slot']:
            category = item['category']
            if category == 2.0 or category == 3.0:
                img_choice.append(img_path)
    return img_choice


# 筛选平行垂直车位图片
def hv_img(img_fold):
    img_choice = []
    for path in os.listdir(img_fold):
        img_path = os.path.join(img_fold, path)
        label = label_load(img_path)
        for item in label['slot']:
            category = item['category']
            if category != 2.0 and category != 3.0:
                img_choice.append(img_path)
    return img_choice


if __name__ == '__main__':
    path = 'tmp_data/data_plus/imgs'
    obl_list = obl_img(path)
    hv_list = hv_img(path)
    print(len(obl_list),"张倾斜车位图片")
    print(len(hv_list),"张垂直车位图片")