# vis_lidar_human_scene

A **user-friendly** and **CPU-only** supported toolkit based on Open3D.

- Visulize the 3D mesh model, PCD list, SMPL pkl
- Edit every geometry's material
- Visulize the  in a folder
- Manually track the points in a frame
- Render the view and automatically save the video. (need ffmpeg) 

### 


![](imgs/gui.jpg)

- The sample SMPL file is in `imgs` folder
  ```bash
  # At least contains the 'pose' and 'mocap_trans'  
  smpl_sample.pkl/
  ├──'first_person'
  |  ├── 'pose' # (N, 72) or (N, 24, 3, 3)
  |  ├── 'mocap_trans' # (N, 3)
  |  ├── 'lidar_traj' (optional_1) # (N, a), a ≥ 4 and coordinate xyz in [:, 1:4]
  |  ├── 'opt_pose' (optional_2) # (N, 72) or (N, 24, 3, 3)
  |  └── 'opt_trans' (optional_2) # (N, 3)
  ├──'second_person'
  |  ├── 'pose' (optional_3) # (N, 72) or (N, 24, 3, 3)
  |  ├── 'mocap_trans' (optional_3) # (N, 3)
  |  ├── 'opt_pose' (optional_4) # (N, 72) or (N, 24, 3, 3)
  |  ├── 'opt_trans' (optional_4) # (N, 3)
  |  ├── 'point_frame' (optional_5) # (n, ) n ≤ N, 'point_frame' ∈ 'frame_num'
  |  └── 'point_clouds' (optional_5) # (n, 3) n ≤ N
  └──'frame_num'(optional_5) # (N, )
  ```
  
<!-- ![](imgs/sample.png) -->
### Run
```
python main_gui.py
```
<!-- - *Set `remote` to `False` in `config.py` if your data is on local machine* -->
  
### Requirements
1. Download the required body model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from http://smpl.is.tue.mpg.de and placed it in `smpl` directory of this repository.
2. `open3d`(>0.15.0) 
3. `torch torchvision torchaudio` (CPU version is enough for SMPL model converting) 
4. `ffmpeg` for video processing

## Todos

- [x] Save the video with GUI
- [x] Add shade, HDR map ...
- [ ] Save the video with headless mode
- [ ] Load video
- [ ] Generate / save camera trajectetory