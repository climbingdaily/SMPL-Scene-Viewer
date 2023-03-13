import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os

class KeypointsCorrector:
    def __init__(self, image_folder, annotations_file):
        self.image_folder = image_folder
        self.annotations_file = annotations_file
        self.current_image_idx = 0
        self.paused = False
        self.load_image_and_annotations()
        self.create_window_and_show_image()

    def load_image_and_annotations(self):
        # 读取图像和关键点标签
        image_file = sorted(os.listdir(self.image_folder))[self.current_image_idx]
        self.image_path = os.path.join(self.image_folder, image_file)
        self.img = cv2.imread(self.image_path)

        # 读取JSON文件
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)

        # 解析JSON数据
        self.keypoints = []
        for annotation in data['annotations']:
            self.keypoints.extend(annotation['keypoints'])
        self.keypoints = np.array(self.keypoints)
        self.keypoints_coords = self.keypoints[:, :2]
        self.keypoints_visibility = self.keypoints[:, 2]

    def create_window_and_show_image(self):
        # 将关键点标签绘制到图像上
        for i, (x, y) in enumerate(self.keypoints_coords):
            if self.keypoints_visibility[i]:
                cv2.circle(self.img, (int(x), int(y)), 3, (0, 0, 255), -1)

        # 创建一个窗口来显示图像
        cv2.namedWindow('image')
        cv2.imshow('image', self.img)

        # 注册鼠标回调函数
        cv2.setMouseCallback('image', self.mouse_callback)

        # 等待用户输入
        while True:
            if not self.paused:
                # 读取下一张图像和关键点标签
                self.current_image_idx = (self.current_image_idx + 1) % len(os.listdir(self.image_folder))
                self.load_image_and_annotations()

                # 绘制关键点标签
                img_copy = self.img.copy()
                for i, (x, y) in enumerate(self.keypoints_coords):
                    if self.keypoints_visibility[i]:
                        cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)

                # 显示图像
                cv2.imshow('image', img_copy)

            key = cv2.waitKey(50)
            if key == ord(' '):
                self.paused = not self.paused
            elif key == 27:
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            # 查找离鼠标最近的关键点的索引
            distances = np.sqrt(np.sum((self.keypoints_coords - [x, y])**2, axis=1))
            nearest_idx = np.argmin(distances)
            # 将该关键点的位置更改为鼠标单击位置
            self.keypoints_coords[nearest_idx] = [x, y]
            # 将更改后的关键点重新绘制到图像上
            img_copy = self.img.copy()
            for i, (x, y) in enumerate(self.keypoints_coords):
                if self.keypoints_visibility[nearest_idx]:
                    cv2.circle(img_copy, (int(x), int(y)), 3, (0, 255, 0), -1)
                else:
                    cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)
            cv2.imshow('image', img_copy)

if __name__ == '__main__':
    image_folder = '/path/to/image/folder'
    annotations_file = '/path/to/annotations/file.json'
    keypoints_corrector = KeypointsCorrector(image_folder, annotations_file)
