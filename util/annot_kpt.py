########################################################################q
# Filename: .\util\correct_keypoints.py
# Created Date: Tuesday, May 2nd 2023, 5:04:56 pm
# Author: climbingdaily
# Copyright (c) 2023 Yudi Dai
########################################################################

import os
import sys
import argparse
from typing import Tuple, Optional

import cv2
import numpy as np

sys.path.append('.')
sys.path.append('..')
from util.sloper4d_loader import SLOPER4D_Dataset

BONES = [    
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8],
    [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

COCO_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170)
]

JOINTS = {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear', 
5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 
9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 
13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}
RED    = (0, 0, 255)
GREEN  = (50, 255, 50)
BLUE   = (255, 0, 0)
CYAN   = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def plot_coco_annotation(img: np.ndarray,
                         keypoints: Optional[np.ndarray] = None,
                         bboxes: Optional[Tuple[int, int, int, int]] = None,
                         keypoint_radius: int = 3,
                         line_width: int = 2,
                         alpha: float = 0.7,
                         text: str='',
                         _KEYPOINT_THRESHOLD: Optional[float] = [0.4]*17,
                         save_path: Optional[str] = None,
                         plot_bone=True) -> np.ndarray:
    
    overlay = np.copy(img)
    
    if bboxes is not None and len(bboxes) > 0:
        for bbox in bboxes:
            if len(bbox) > 0:
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            color=(220, 173, 69), thickness=3)
            
    if keypoints is not None and len(keypoints) > 0:
        for per_kpt in keypoints:
            if len(per_kpt) == 0:
                continue
            per_kpt    = per_kpt.reshape(-1, 3)
            points     = per_kpt[:, :2].astype(int)
            visibility = per_kpt[:, 2]
            
            if plot_bone:
                for i, conn in enumerate(BONES):
                    if visibility[conn[0]] > _KEYPOINT_THRESHOLD[i] and visibility[conn[1]] > _KEYPOINT_THRESHOLD[i]:
                        cv2.line(img, tuple(points[conn[0]]), tuple(points[conn[1]]), 
                                color=(np.array(COCO_COLORS[conn[0]]) + np.array(COCO_COLORS[conn[1]]))/2, 
                                thickness=line_width)
                    else:
                        cv2.line(img, tuple(points[conn[0]]), tuple(points[conn[1]]), 
                                color=(100, 100, 100), 
                                thickness=line_width-1)
                    
                cv2.addWeighted(overlay, 1-alpha, img, alpha, 0, img)

            for i, p in enumerate(points):
                if visibility[i] > _KEYPOINT_THRESHOLD[i]:
                    cv2.circle(img, (p[0], p[1]), 
                               radius=keypoint_radius, 
                               color=COCO_COLORS[i], 
                               thickness=-1)
                else:
                    cv2.circle(img, (p[0], p[1]), 
                               radius=keypoint_radius-1, 
                               color=(100,100,100), 
                               thickness=-1)

    if text is not None and len(text) > 0:
        cv2.putText(img, os.path.basename(text), (30, 60), DEFAULT_FONT, 1, BLACK, 2)

    if save_path is not None:
        cv2.imwrite(save_path, img)
        
    return img

def plot_bbox(image, bbox, label=None, clicked=False):
    x, y, w, h = bbox
    box_color = RED if clicked else (220, 173, 69)
    text_color = RED if clicked else WHITE
    add_mask_to_bbox(image, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
    if label:
        if '_' in label:
            side = label.split('_')[0][0].upper()
            label = label.split('_')[1]
            cv2.putText(image, side, (x+5, round(y+0.3*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        cv2.putText(image, label, (x+5, round(y+0.9*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

def add_mask_to_bbox(image, bbox, alpha=0.3):
    mask = np.copy(image)
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = [0, 200, 200]
    cv2.addWeighted(image, 1 - alpha, mask, alpha, 0, image)


class correct_keypoints():
    def __init__(self, pkl_file, img_folder):
        
        self.frame_index = 0
        self.kpt_select = -1
        self.img_folder = img_folder
        self.type = ''

        self.sequence  = SLOPER4D_Dataset(pkl_file)
        self.keypoints = None
        self.kpt_new   = None
        self.img       = None
        self.img_new   = None
        self.img_name  = ''

        # make bboxes
        box_w  = 100
        box_h  = 50
        gap    = 3
        bboxes = []
        for i in range(5):
            for j in range(4):
                bboxes += [[50+(box_w+gap) * i, 120 + box_h*j + gap*j, box_w, box_h]]
        self.bboxes = bboxes[:17]

    def set_data(self, ):
        """
        This function sets data for an image and displays it with annotations and instructions.
        """
        sample    = self.sequence[self.frame_index]
        self.img_name  = sample['file_basename']
        self.kpt_select= -1 
        self.keypoints = np.array(sample['skel_2d']).reshape(-1, 3)
        self.kpt_new   = self.keypoints.copy()
        self.img       = cv2.imread(os.path.join(self.img_folder, self.img_name))
        h, _, _ = self.img.shape
        cv2.putText(self.img, f"{self.frame_index:06d} - {self.img_name}", (30, 60), DEFAULT_FONT, 1, BLACK, 2)
        cv2.putText(self.img, "'<-'/'->'   Previous/next frame", (30, int(h/2)), DEFAULT_FONT, 0.5, BLACK, 2)
        cv2.putText(self.img, "'q'       Quit", (30,  int(h/2 + 30)), DEFAULT_FONT, 0.5, BLACK, 2)
        cv2.putText(self.img, "'ESC'     Abandon changes", (30, int(h/2+60)), DEFAULT_FONT, 0.5, BLACK, 2)
        cv2.putText(self.img, "'ENTER'   Accept changes", (30, int(h/2+90)), DEFAULT_FONT, 0.5, BLACK, 2)
        cv2.putText(self.img, "' i '     Input frame number in the terminal", (30, int(h/2+120)), DEFAULT_FONT, 0.5, BLACK, 2)
        cv2.putText(self.img, "' d '     Delete current frame keypoints", (30, int(h/2+150)), DEFAULT_FONT, 0.5, BLACK, 2)

        self.img_new   = np.copy(self.img)
        plot_coco_annotation(self.img_new, [self.keypoints.copy(), ])
        for i, bbox in enumerate(self.bboxes):
            plot_bbox(self.img_new, bbox, JOINTS[i])

        cv2.imshow('image', self.img_new)

    def display_image(self, ):
        cv2.imshow('image', self.img_new)
        key = cv2.waitKeyEx(0)
        if key == 27:           # press ESC  refresh the keypoints
            self.set_data()

        elif key == 2424832:    # press <-  previous frame
            self.frame_index -= 1
            self.set_data()

        elif key == 2555904:    # press ->  next frame
            self.frame_index += 1
            self.set_data()

        elif key == ord('q'):    # press q  quit
            self.frame_index = -1
            # self.set_data()

        elif key == ord('d'):    # press d  detele current keypoints
            self.keypoints = []
            self.kpt_new = []
            self.sequence.updata_pkl(self.img_name, keypoints=[], bbox=[])
            self.set_data()

        elif key == 13:          # if press Enter, accept the changes, go to next frame
            if self.kpt_select >= 0:
                self.keypoints[self.kpt_select] = self.kpt_new[self.kpt_select]
                self.kpt_new = self.keypoints.copy()
                print(f"======> {self.kpt_select}: {JOINTS[self.kpt_select]} is annotated." + 
                      f"{self.kpt_new[self.kpt_select]}")
            self.sequence.updata_pkl(self.img_name, keypoints=self.keypoints)
            # self.frame_index += 1
            self.set_data()

        elif key == ord('s'):  # if press s, save the pkl
            self.sequence.save_pkl()

        elif key == ord('i'):  # if press i 
            num = input("Please input a frame number: ")
            self.kpt_select = -1 
            try:
                self.frame_index = int(num.strip())
                self.set_data()
            except:
                print("Input must be a int number")
        
        if self.frame_index < 0 or self.frame_index >= len(self.sequence):
            cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, params):
        scale = 5
        size  = 50

        if event == cv2.EVENT_LBUTTONDOWN:
            self.img_new = np.copy(self.img)
            if self.kpt_select < 0:
                for i, bbox in enumerate(self.bboxes):
                    x1, y1, w, h = bbox
                    if x > x1 and x < x1 + w and y > y1 and y < y1 + h:
                        # Highlight the selected bbox
                        self.kpt_select = i 
                    if abs(self.kpt_new[i][0] - x)<3 and abs(self.kpt_new[i][1] - y)<3:
                        self.kpt_select = i 

                    plot_bbox(self.img_new, bbox, JOINTS[i], clicked=i==self.kpt_select)
                if self.kpt_select >= 0:
                    self.kpt_new[:, 2] = 0.01
                    self.kpt_new[self.kpt_select, 2] = 0.6
            else:
                self.kpt_new[:, 2] = 0.01
                self.kpt_new[self.kpt_select] = [x, y, 0.6]
                for i, bbox in enumerate(self.bboxes):
                    plot_bbox(self.img_new, bbox, JOINTS[i], clicked=i==self.kpt_select)

            plot_coco_annotation(self.img_new, [self.kpt_new, ])
            self.display_image()

        if event == cv2.EVENT_MOUSEMOVE:

            ymin  = max(0, y-size)
            ymax  = min(self.img.shape[0], y+size)
            xmin  = max(0, x-size)
            xmax  = min(self.img.shape[1], x+size)
            xsize = (xmax - xmin) * scale
            ysize = (ymax - ymin) * scale

            area = self.img_new[ymin:ymax, xmin:xmax].copy()
            area = cv2.resize(area, dsize=(xsize, ysize),
                            interpolation=cv2.INTER_LINEAR)
            cv2.circle(area, ((x-xmin) * scale, (y-ymin) * scale), 
                    3, (255, 0, 0), thickness=-1)
            cv2.putText(area, f'{x:.1f} {y:.1f}', 
                        ((x-xmin) * scale, (y-ymin) * scale), 
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
            cv2.imshow('Zoom', area)
            cv2.setWindowProperty("zoom", cv2.WND_PROP_TOPMOST, 1)

    def save_pkl(self, ):
        self.sequence.save_pkl()

    def run(self, ):
        self.set_data()
        plot_coco_annotation(self.img_new, [self.keypoints.copy(), ])
        for i, bbox in enumerate(self.bboxes):
            plot_bbox(self.img_new, bbox, JOINTS[i])

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse_click)
        while self.frame_index >= 0 and self.frame_index < len(self.sequence):
            self.display_image()
        cv2.destroyAllWindows() 

        self.save_pkl()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument('--pkl_file', type=str, required=True, default='', 
                        help='Path to the pkl file')
    parser.add_argument('--img_folder', type=str, required=True, default='', 
                        help='Path to the image folder')
    args = parser.parse_args()
    
    pipeline = correct_keypoints(args.pkl_file, args.img_folder)
    try:
        pipeline.run()
    except Exception as e:
        pipeline.save_pkl()
        print(f"Error {e.args[0]}")