python run.py \
    --encoder vitl \
    --load-from /data_nvme/Depth-Estimation/event/depth_anything_v2_dense_normalized_log_20241126_212202/d1-0.9731687307357788-34.pth \
    --max-depth 1  \
    --img-path /data_nvme/Depth-Estimation/DENSE/test.txt \
    --outdir /data_nvme/Depth-Estimation/dense-test/dav2-epoch35 \
    --save-numpy
# --pred-only