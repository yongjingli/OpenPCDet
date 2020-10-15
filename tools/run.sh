# test
#python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --batch_size 1 --ckpt pv_rcnn_8369.pth

# train
#python train.py --cfg cfgs/kitti_models/pv_rcnn.yaml --batch_size 1 --pretrained_model pv_rcnn_8369.pth

# demo
# python demo.py --cfg cfgs/kitti_models/pv_rcnn.yaml --data_path /home/liyongjing/sukai/train/OpenPCDet-master/data/kitti/training/velodyne --ckpt pv_rcnn_8369.pth

# save pred result
python demo_save_pred.py --cfg cfgs/kitti_models/pv_rcnn.yaml --data_path /home/liyongjing/sukai/train/OpenPCDet-master/data/kitti/training/velodyne --ckpt pv_rcnn_8369.pth
