import os
import argparse
from tqdm import tqdm

import cv2

class Vid2imgs:
    def __init__(self, video_path):
        assert os.path.isfile(video_path), "Error: video no exist!"

        self.vid = cv2.VideoCapture()
        self.vid.open(video_path)

        assert self.vid.isOpened(), "Error: video not opened!"

        self.vid_fps   = float(self.vid.get(cv2.CAP_PROP_FPS))
        self.save_path = f"{os.path.splitext(video_path)[0]}_imgs"
        
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Images will be saved to \n ==> {self.save_path}")

    def _vid_frames(self):
        while self.vid.isOpened():
            success, frame = self.vid.read()
            if success: 
                yield frame
            else: 
                break

    def run(self, pkl_path):
        img_idx = 0
        progress_bar = tqdm(self._vid_frames())

        if pkl_path is not None and os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            file_basename = data['RGB_frames']['file_basename'] # list of n strings
            valid_timestamp = [fn[:-4] for fn in file_basename]
        else:
            valid_timestamp = []

        for frame in progress_bar:

            timestamp = f"{img_idx / self.vid_fps:.06f}"
            img_fname = timestamp + ".jpg"
            img_path  = os.path.join(self.save_path, img_fname)

            if len(valid_timestamp) == 0 or timestamp in valid_timestamp:
                progress_bar.set_description(f"Saving {img_fname}")
                cv2.imwrite(img_path, frame)
            img_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("vid_path", type=str, 
                        help="The path to the video")
    parser.add_argument("--pkl", type=str, default='',
                        help="The annotations fiel path")
    return parser.parse_args() 
            
if __name__ == '__main__':
    args        = parse_args()
    vid_path    = args.vid_path
    pkl_path    = args.pkl
    
    seq_name    = os.path.basename(vid_path)[:-4]

    if not os.path.isfile(vid_path): 
        print("Not found: ", vid_path)
    else:
        to_images = Vid2imgs(vid_path)
        to_images.run(pkl_path)
        