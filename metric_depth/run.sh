python run.py \
    --encoder vitl \
    --load-from /data_nvme/Depth-Estimation/event/depth_anything_v2_dense_normalized_log_20241126_212202/d1-0.9731687307357788-34.pth \
    --max-depth 1  \
    --img-path /data_nvme/Depth-Estimation/MVSEC/outdoor-night/outdoor-night1/left_depth_based_split.txt \
    --outdir /data_nvme/Depth-Estimation/mvsec-test/dav2-dense-nl35/outdoor-night1 \
    --save-numpy
    # --pred-only