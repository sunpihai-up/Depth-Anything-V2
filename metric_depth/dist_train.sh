#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

epoch=100
bs=4
gpus=2
lr=0.000005
encoder=vitl
dataset=dense # vkitti
img_size=518
min_depth=0
max_depth=1000 # 80 for virtual kitti
pretrained_from=/home/sph/code/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth
save_path=/data_nvme/Depth-Estimation/event/depth_anything_v2_${dataset}_normalized_log_${now} # exp/vkitti

mkdir -p $save_path

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
