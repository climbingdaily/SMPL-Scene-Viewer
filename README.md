# vis_lidar_human_scene
Visualization tool for SMPL and lidar data

## For every human, there are three types data:
1. *point clouds* from LiDAR
2. *Ground truth SMPL pose* from MoCap
3. *Predicted SMPL pose* from LiDARCap
   
## 1. Visulize one human. Predicted SMPL pose is optional.
- Suport binary `pkl` and `hdf5` formats.
- The file should contains `pose` and `point_clouds`
- Or contains `gt_pose`, `point_clouds` and `pred_rotmats`
  
```bash
python vis_pred_smpl.py -F "/path/to/your/file"
```

## 2. Visulize multi-human's data. Predicted SMPL pose is optional.

## Todos

- [ ] 有GUI自动保存视频
- [ ] 无GUI自动保存视频
- [ ] 更改可视化环境光