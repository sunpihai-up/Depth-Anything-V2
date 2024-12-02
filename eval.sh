# python evaluation.py \
#     --predictions_dataset /data_nvme/Depth-Estimation/dense-test/dav2-metric-epoch18/npy \
#     --target_dataset /data_nvme/Depth-Estimation/DENSE/test/test-sequence-00-town10/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --metric

# python evaluation.py \
#     --predictions_dataset /data_nvme/Depth-Estimation/dense-test/dav2-zeroshot/test/npy \
#     --target_dataset /data_nvme/Depth-Estimation/DENSE/test/test-sequence-00-town10/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --inv

python evaluation.py \
    --predictions_dataset /data_nvme/Depth-Estimation/mvsec-test/dav2-dense-nl35/outdoor-night1/npy \
    --target_dataset /data_nvme/Depth-Estimation/MVSEC/outdoor-night/outdoor-night1/left_rect_depth \
    --clip_distance 80.0 \
    --dataset mvsec \
    --json_path /data_nvme/Depth-Estimation/MVSEC/outdoor-night/outdoor-night1/depth2image.json \
    --nan_mask