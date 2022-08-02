# vis_lidar_human_scene
Visualization tool for SMPL and lidar data

   
## 1. Visulize one human with its point clouds. 
- Suport binary `pkl` and `hdf5` formats.
- The file should contains `pose` (N, a) and `point_clouds` (N, b) or contains `gt_pose` (N, a), `point_clouds` (N, b) and `pred_rotmats` (N, a)
  - *a = 72 or 24, 3, 3*
  - *b = any number*
  
```bash
python vis_pred_smpl.py -F "/path/to/your/file"
```

## 2. Visulize humans and the scene. 
- Suported human data file structure
  ```bash
  pkl/
  ├──'first_person'
  |  ├── 'pose' # (N, a)
  |  ├── 'mocap_trans' # (N, 3)
  |  ├── 'lidar_traj' (optional_1) # (N, 9), xyz in [:, 1 : 4]
  |  ├── 'opt_pose' (optional_2) # (N, a)
  |  └── 'opt_trans' (optional_2) # (N, 3)
  ├──'second_person'
  |  ├── 'pose' # (N, a)
  |  ├── 'mocap_trans' # (N, 3)
  |  ├── 'opt_pose' (optional_3) # (N, a)
  |  ├── 'opt_trans' (optional_3) # (N, 3)
  |  ├── 'point_frame' (optional_4) # (n, )
  |  └── 'point_clouds' (optional_4) # (N, b)
  └──'frame_num'(optional_4) # (N, )
  ```
  - *a = 72 or 24, 3, 3*
  - *b = any number*
  
```
python vis_smpl_scene.py --smpl_file_path /path/to/pkl_file --scene_path /path/to/scene_pcd  --viewpoint_type third
```
- *Set `remote` to `False` in `config.py` if your data is on local machine*
  
## Requirements
Download the required body model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from http://smpl.is.tue.mpg.de and placed it in `smpl` directory of this repository.

## Todos

- [x] Save the video with GUI
- [ ] Save the video with headless mode
- [ ] Add shade, HDR map ...