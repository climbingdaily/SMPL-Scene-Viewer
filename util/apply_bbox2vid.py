import argparse
import cv2
import numpy as np
from scipy.interpolate import interp1d


def load_bboxes(bboxes_path):
    # 读取pkl, 返回字典，key是原始帧号，value是bbox
    bboxes = {}
    start = 0
    end = 0
    # with open(bboxes_path, 'r') as f:
    #     for line in f:
    #         if line.startswith('#'):
    #             continue
    #         frame_idx, x, y, w, h = line.strip().split(',')
    #         bboxes[int(frame_idx)] = [int(x), int(y), int(w), int(h)]
            
    return bboxes, start, end


def interpolate_bboxes(bboxes):
    frame_indices = sorted(list(bboxes.keys()))
    x_values = [bboxes[i][0] for i in frame_indices]
    y_values = [bboxes[i][1] for i in frame_indices]
    w_values = [bboxes[i][2] for i in frame_indices]
    h_values = [bboxes[i][3] for i in frame_indices]

    f_x = interp1d(frame_indices, x_values, kind='linear', fill_value='extrapolate')
    f_y = interp1d(frame_indices, y_values, kind='linear', fill_value='extrapolate')
    f_w = interp1d(frame_indices, w_values, kind='linear', fill_value='extrapolate')
    f_h = interp1d(frame_indices, h_values, kind='linear', fill_value='extrapolate')

    return f_x, f_y, f_w, f_h


def process_video(original_video_path, mosaic_video_path, bboxes_path):
    bboxes, start_frame, end_frame = load_bboxes(bboxes_path)
    f_x, f_y, f_w, f_h = interpolate_bboxes(bboxes)

    cap_ori = cv2.VideoCapture(original_video_path)
    cap_mos = cv2.VideoCapture(mosaic_video_path)

    # Check if camera opened successfully
    if not cap_ori.isOpened() or not cap_mos.isOpened():
        print("Error opening video stream or file")
        return

    # Get video information
    fps = cap_ori.get(cv2.CAP_PROP_FPS)
    width = int(cap_ori.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_ori.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap_ori.get(cv2.CAP_PROP_FOURCC))

    # Create VideoWriter object
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    while cap_ori.isOpened() and cap_mos.isOpened():
        # Read frames from original and mosaic videos
        ret_ori, frame_ori = cap_ori.read()
        ret_mos, frame_mos = cap_mos.read()

        if ret_ori and ret_mos:
            # Get current frame number and bbox
            frame_idx = int(cap_ori.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if frame_idx < start_frame or frame_idx > end_frame:
                continue
            
            x, y, w, h = bboxes.get(frame_idx, [0, 0, 0, 0])

            if w == 0 or h == 0:
                # If bbox is not available, use the interpolated bbox
                x = int(f_x(frame_idx))
                y = int(f_y(frame_idx))
                w = int(f_w(frame_idx))
                h = int(f_h(frame_idx))

            # Copy original frame pixels to mosaic frame
            frame_mos[y:y+h, x:x+w, :] = frame_ori[y:y+h, x:x+w, :]

            # Write processed frame to output video
            out.write(frame_mos)
        else:
            break

    # Release video objects
    cap_ori.release()
    cap_mos.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True, help='Path to the original video file')
    parser.add_argument('--mosaic', required=True, help='Path to the mosaic video file')
    parser.add_argument('--bbox', required=True, help='Path to the bbox data file')
    args = parser.parse_args()

    process_video(args.original, args.mosaic, args.bbox)
