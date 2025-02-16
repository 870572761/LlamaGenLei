log_dir=log 
current_date=$(date +%Y-%m-%d)
CUDA_VISIBLE_DEVICES=0,1,2,3 
mkdir -p $log_dir/${current_date}
nohup bash scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
    --data-path /data2/lei/imagenet1080_condition \
    --code-path /data2/lei/llamagen_code_filp_1080_c2i_L1_VQ-16 \
    --flip \
    --image-size 384 > $log_dir/${current_date}/${current_date}_gen1080_dataoutput.log 2>&1 &
echo "Process started with PID $!"