python evaluation.py \
    --predictions_dataset /data_nvme/Depth-Estimation/dense-test/dav2-epoch35/npy \
    --target_dataset /data_nvme/Depth-Estimation/DENSE/test/test-sequence-00-town10/depth/data \
    --clip_distance 1000.0 \
    --dataset dense 

# python evaluation.py \
#     --predictions_dataset /data_nvme/Depth-Estimation/dense-test/dav2-zeroshot/test/npy \
#     --target_dataset /data_nvme/Depth-Estimation/DENSE/test/test-sequence-00-town10/depth/data \
#     --clip_distance 1000.0 \
#     --dataset dense \
#     --inv